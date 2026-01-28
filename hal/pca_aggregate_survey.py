import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# ==========================================================================
# CONFIGURATION
# ==========================================================================
TEST_SIZE = 0.1
RANDOM_SEED = 42

# CACHE SETTINGS
CACHE_FILE = 'pca_aggregate_survey_cache.pkl'

# MODEL
K_MODEL = 30
USE_PCA = True  # Set to False to use raw embeddings instead of PCA
PCA_COMPONENTS = 48  # Only used if USE_PCA is True

# SPARSITY
LAMBDA_TAU = 0.002
TAU_INIT = 0.5
TAU_WARMUP = 200
RAMP_EPOCHS = 200
SNAPPING_THRESHOLD = 0.005
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# TRAINING
EPOCHS = 1000
EVAL_EVERY = 100  # Reduced frequency to speed up multi-run
PATIENCE = 30
MIN_DELTA = 1e-6

# OPTIMIZATION
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 1e-3
WD_W = 1e-5

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================
def load_cache():
    """Load cached results from disk."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cache from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    """Save cached results to disk."""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {CACHE_FILE}")

def compute_rmse(predictions, targets, mask):
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

class ReluARDModel(nn.Module):
    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)
        self.difficulty_proj = nn.Linear(d, 1)

    def get_tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        W_norm = F.normalize(self.W, dim=1)
        base_loadings = self.x_j @ W_norm.T
        tau = self.get_tau()
        a_j = base_loadings * tau.unsqueeze(0)
        if self.training and self.dropout > 0:
            a_j = F.dropout(a_j, p=self.dropout)
        diff = self.difficulty_proj(self.x_j).squeeze()
        logits = self.theta @ a_j.T + diff.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        return torch.sigmoid(logits)

# ==========================================================================
# DATA LOADING & PREPARATION
# ==========================================================================
resmat_dir = 'resmats'
# Get all files and sort them to ensure consistent incremental batches
all_files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
total_files = len(all_files)

# Pre-load embeddings (do this once to save time)
emb_file = 'result/all_benchmarks_embeddings_4096_8B.pkl'
raw_embs_map = {}
if os.path.exists(emb_file):
    print("Loading embeddings dictionary...")
    emb_df = pd.read_pickle(emb_file)
    # Create a quick lookup map
    for _, r in emb_df.iterrows():
        raw_embs_map[str(r['benchmark.task_id'])] = r['embedding']
        # Handle colbench edge case mapping if needed
        if str(r['benchmark.task_id']).startswith('colbench_backend_programming'):
            suffix = str(r['benchmark.task_id']).split('.')[-1]
            raw_embs_map[f'colbench.{suffix}'] = r['embedding']

# ==========================================================================
# MAIN LOOP: ITERATE OVER BATCH SIZES
# ==========================================================================
# Load existing cache
cache = load_cache()
results = []

print(f"Found {total_files} total matrix files. Starting loop from 2 to {total_files}...")

# Cycle from 2 files up to Total files
for n_current_files in range(2, total_files + 1):
    # Check cache first
    if n_current_files in cache:
        print(f"\n[{n_current_files}/{total_files}] Loading from cache...")
        results.append(cache[n_current_files])
        continue
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print(f"\n[{n_current_files}/{total_files}] Processing batch size: {n_current_files}")
    
    # ------------------------------------------------
    # 1. Load Subset of Data
    # ------------------------------------------------
    current_files = all_files[:n_current_files]
    current_dfs = []
    for f in current_files:
        current_dfs.append(pd.read_csv(os.path.join(resmat_dir, f), index_col=0))
    
    # Intersection of indices
    shared_indices = set(current_dfs[0].index)
    for df in current_dfs[1:]:
        shared_indices = shared_indices.intersection(set(df.index))
    shared_indices = sorted(list(shared_indices))
    
    filtered_dfs = [df.loc[shared_indices] for df in current_dfs]
    stacked_matrix = np.array([df.values for df in filtered_dfs])
    
    # Compute Prob Matrix (Ground Truth for this run)
    prob_matrix = np.nanmean(stacked_matrix, axis=0)
    prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)
    
    # ------------------------------------------------
    # 2. Prepare Embeddings for this Subset
    # ------------------------------------------------
    task_ids = prob_df.columns.tolist()
    current_raw_embs = []
    for task_id in task_ids:
        emb = raw_embs_map.get(str(task_id))
        if emb is None:
             # Try fallback for colbench
            if task_id.startswith('colbench.'):
                number = task_id.split('.')[-1]
                emb = raw_embs_map.get(f'colbench_backend_programming.{number}')
        
        if emb is None:
            emb = np.zeros(4096)
        elif isinstance(emb, str):
            emb = ast.literal_eval(emb)
        current_raw_embs.append(emb)

    # Apply PCA or use raw embeddings
    x_np = np.array(current_raw_embs)
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        x_transformed = pca.fit_transform(x_np)
        embedding_dim = PCA_COMPONENTS
    else:
        x_transformed = x_np
        embedding_dim = x_np.shape[1]
    
    # Normalization
    x_transformed = x_transformed / (np.linalg.norm(x_transformed, axis=1, keepdims=True) + 1e-8)
    x_j = torch.tensor(x_transformed, dtype=torch.float32).to(device)
    
    # ------------------------------------------------
    # 3. Train/Test Split
    # ------------------------------------------------
    N, J = prob_df.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)
    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    y_empirical = torch.from_numpy(prob_df.values.astype(np.float32)).to(device)
    
    # Masks
    train_mask = np.zeros_like(prob_df.values, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(prob_df.values)[:, train_idx]
    test_mask = np.zeros_like(prob_df.values, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(prob_df.values)[:, test_idx]
    
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)
    
    # ------------------------------------------------
    # 4. Run Baseline 1: Global Mean
    # ------------------------------------------------
    mean_val = torch.nanmean(y_empirical[train_mask_t])
    pred_mean = mean_val.expand_as(y_empirical)
    rmse_mean_test = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
    
    # ------------------------------------------------
    # 5. Run Baseline 2: Rasch-IRT
    # ------------------------------------------------
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
    opt_rasch = torch.optim.Adam([theta, beta], lr=0.01)
    
    # Fast training for Rasch
    for _ in range(300):
        opt_rasch.zero_grad()
        loss = F.binary_cross_entropy_with_logits((theta.unsqueeze(1)-beta.unsqueeze(0)), y_empirical, reduction='none')
        (loss * train_mask_t).sum().backward()
        opt_rasch.step()
        
    with torch.no_grad():
        p_rasch = torch.sigmoid(theta.unsqueeze(1)-beta.unsqueeze(0))
        rmse_rasch_test = compute_rmse(p_rasch.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

    # ------------------------------------------------
    # 6. Run Model: Beta-IRT (ReluARDModel)
    # ------------------------------------------------
    model = ReluARDModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.5).to(device)
    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
        {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
        {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
    ])
    
    best_beta_rmse = float('inf')
    patience_counter = 0
    
    # Training Loop
    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        probs = model()
        
        loss_fit = F.binary_cross_entropy(probs[train_mask_t], y_empirical[train_mask_t])
        
        # Annealing
        if epoch < TAU_WARMUP:
            current_lambda = 0.0
        elif epoch < TAU_WARMUP + RAMP_EPOCHS:
            progress = (epoch - TAU_WARMUP) / RAMP_EPOCHS
            current_lambda = LAMBDA_TAU * progress
        else:
            current_lambda = LAMBDA_TAU
            
        tau = model.get_tau()
        loss_sparsity = current_lambda * torch.sum(tau)
        (loss_fit + loss_sparsity).backward()
        optimizer.step()
        
        # Snapping
        if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
            with torch.no_grad():
                active_mask = model.get_tau() > SNAPPING_THRESHOLD
                for k in range(K_MODEL):
                    if not active_mask[k]:
                        model.tau_raw[k] = DEAD_ZONE_VALUE
        
        # Evaluation
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p_test = model()
                curr_rmse = compute_rmse(p_test.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
                
                if curr_rmse < best_beta_rmse - MIN_DELTA:
                    best_beta_rmse = curr_rmse
                    patience_counter = 0
                elif epoch > TAU_WARMUP:
                    patience_counter += 1
    
    print(f"   -> Global Mean: {rmse_mean_test:.4f} | Rasch: {rmse_rasch_test:.4f} | Beta-IRT: {best_beta_rmse:.4f}")
    
    # Store results
    result_entry = {
        'n_samples': n_current_files,
        'rmse_mean': rmse_mean_test,
        'rmse_rasch': rmse_rasch_test,
        'rmse_beta': best_beta_rmse
    }
    results.append(result_entry)
    
    # Cache this result
    cache[n_current_files] = result_entry
    save_cache(cache)

# ==========================================================================
# PLOTTING AND ANALYSIS
# ==========================================================================
print("\n" + "="*60 + "\nGENERATING PLOTS\n" + "="*60)
df_res = pd.DataFrame(results)

# Calculate Improvement over the best baseline (min of Mean or Rasch)
df_res['baseline_min'] = df_res[['rmse_mean', 'rmse_rasch']].min(axis=1)
df_res['improvement_pct'] = ((df_res['baseline_min'] - df_res['rmse_beta']) / df_res['baseline_min']) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- PLOT 1: RMSE COMPARISON ---
ax1.plot(df_res['n_samples'], df_res['rmse_mean'], marker='s', linestyle='--', label='Global Mean Baseline', color='gray', alpha=0.7)
ax1.plot(df_res['n_samples'], df_res['rmse_rasch'], marker='^', linestyle='--', label='Rasch-IRT Baseline', color='tab:red', alpha=0.7)
ax1.plot(df_res['n_samples'], df_res['rmse_beta'], marker='o', linestyle='-', label='Amortised IRT model', color='tab:blue', linewidth=2.5)

ax1.set_xlabel('Number of Response Matrices (Samples)', fontsize=12)
ax1.set_ylabel('Test RMSE (Lower is Better)', fontsize=12)
ax1.set_title('RMSE Convergence Analysis', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xticks(df_res['n_samples'])

# --- PLOT 2: IMPROVEMENT OVER BASELINE ---
bars = ax2.bar(df_res['n_samples'].astype(str), df_res['improvement_pct'], color='tab:green', alpha=0.8, width=0.6)

ax2.set_xlabel('Number of Response Matrices (Samples)', fontsize=12)
ax2.set_ylabel('Improvement over Best Baseline (%)', fontsize=12)
ax2.set_title('Relative Performance Gain', fontsize=14)
ax2.grid(axis='y', linestyle=':', alpha=0.6)

# Add text labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('beta_irt_comparison_analysis.png')
print("Plots saved to 'beta_irt_comparison_analysis.png'")
plt.show()

# Print Data Table
print("\nFinal Statistics Summary:")
print(df_res.to_string(index=False))