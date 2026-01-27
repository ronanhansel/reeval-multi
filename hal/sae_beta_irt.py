# ==========================================================================
# CONFIGURATION - ROBUST SAE-BASED ARCHITECTURE
# ==========================================================================
USE_EMPIRICAL_BASELINE = True
TEST_SIZE = 0.1
RANDOM_SEED = 42

# ARCHITECTURE
K_MODEL = 32            # High capacity, let ARD prune
SAE_FEATURES = 48       # SAE Feature Dimension (M)
SAE_K_SPARSITY = 4      # Active features per item (K)

# SPARSITY (Gentle Annealing)
LAMBDA_TAU = 0.00025     # Low penalty to allow strong factors
TAU_WARMUP = 200
RAMP_EPOCHS = 200
SNAPPING_THRESHOLD = 0.01
DEAD_ZONE_VALUE = -0.5

# TRAINING
EPOCHS = 2000
EVAL_EVERY = 50
PATIENCE = 30
MIN_DELTA = 1e-6

# OPTIMIZATION (High-Gain / Robust Mode)
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 0.0          # CRITICAL: Allow agents to diverge (No regularization)
WD_W = 0.0              # Allow sharp projections
DROPOUT_RATE = 0.0      # No dropout for small N
INIT_SCALE = 0.5        # Bold initialization
TAU_INIT = 2.0          # Start with strong factors

## 1. Setup and Configuration
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing hypothesaes (from your notebook)
try:
    from hypothesaes.quickstart import train_sae
    HAS_SAE_LIB = True
except ImportError:
    HAS_SAE_LIB = False
    print("WARNING: 'hypothesaes' not found. Please install it or the code will fail at SAE step.")

warnings.filterwarnings('ignore')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

def compute_rmse(predictions, targets, mask):
    # Ensure mask is boolean
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    # Calculate MSE only on valid (observed) entries
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

## 2. Load Response Matrices
resmat_dir = 'resmats'
files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
all_dfs = []
for f in files:
    df = pd.read_csv(os.path.join(resmat_dir, f), index_col=0)
    all_dfs.append(df)

shared_indices = set(all_dfs[0].index)
for df in all_dfs[1:]:
    shared_indices = shared_indices.intersection(set(df.index))
shared_indices = sorted(list(shared_indices))

filtered_dfs = [df.loc[shared_indices] for df in all_dfs]
stacked_matrix = np.array([df.values for df in filtered_dfs])
prob_matrix = np.nanmean(stacked_matrix, axis=0)
prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

print(f"Loaded {len(files)} matrices. Shared models: {len(shared_indices)}, Tasks: {prob_df.shape[1]}")

## 3. Prepare Embeddings & SAE
emb_file = 'result/all_benchmarks_embeddings_4096_8B.pkl'
if os.path.exists(emb_file):
    emb_df = pd.read_pickle(emb_file)
    task_ids = prob_df.columns.tolist()
    emb_map = {str(r['benchmark.task_id']): r['embedding'] for _, r in emb_df.iterrows()}
    
    raw_embs = []
    for task_id in task_ids:
        emb = emb_map.get(task_id)
        if emb is None and task_id.startswith('colbench.'):
            number = task_id.split('.')[-1]
            emb = emb_map.get(f'colbench_backend_programming.{number}')
        if emb is None:
            emb = np.zeros(4096)
        elif isinstance(emb, str):
            emb = ast.literal_eval(emb)
        raw_embs.append(emb)
    
    # Normalize raw embeddings before SAE
    x_raw_np = np.array(raw_embs, dtype=np.float32)
    norms = np.linalg.norm(x_raw_np, axis=1, keepdims=True)
    x_raw_np = x_raw_np / (norms + 1e-8)
    
    print(f"Embeddings loaded: {len(raw_embs)}")
else:
    raise FileNotFoundError("Embedding file not found!")

# --- SAE IMPLEMENTATION (Replaces PCA) ---
print(f"\n[Preprocessing] Training Sparse Autoencoder (M={SAE_FEATURES}, K={SAE_K_SPARSITY})...")

if HAS_SAE_LIB:
    # Train SAE using the library from the notebook
    sae = train_sae(
        embeddings=x_raw_np,
        M=SAE_FEATURES,
        K=SAE_K_SPARSITY,
        batch_size=512,
        n_epochs=100,
        learning_rate=5e-4,
        checkpoint_dir='checkpoints/hal_sae_temp'
    )
    
    print("Transforming embeddings to SAE activations...")
    x_sae_activations = sae.get_activations(x_raw_np)
    x_j = torch.tensor(x_sae_activations, dtype=torch.float32).to(device)
    print(f"SAE Feature Shape: {x_j.shape}")
else:
    # Fallback to random projection if library missing (for testing flow)
    print("!! SAE Library missing - Using Random Projection Fallback !!")
    x_j = torch.randn(len(prob_df.columns), SAE_FEATURES).to(device)

# Split Train/Test
N, J = prob_df.shape
np.random.seed(RANDOM_SEED)
J_indices = np.arange(J)
np.random.shuffle(J_indices)
n_test = int(TEST_SIZE * J)
test_idx = J_indices[:n_test]
train_idx = J_indices[n_test:]

y_empirical = torch.from_numpy(prob_df.values.astype(np.float32)).to(device)
train_mask = np.zeros_like(prob_df.values, dtype=bool)
train_mask[:, train_idx] = ~np.isnan(prob_df.values)[:, train_idx]
test_mask = np.zeros_like(prob_df.values, dtype=bool)
test_mask[:, test_idx] = ~np.isnan(prob_df.values)[:, test_idx]
train_mask_t = torch.from_numpy(train_mask).to(device)
test_mask_t = torch.from_numpy(test_mask).to(device)

## 4. Baselines
print("\n" + "="*60 + "\nBASELINES\n" + "="*60)
mean_val = torch.nanmean(y_empirical[train_mask_t])
pred_mean = mean_val.expand_as(y_empirical)
train_rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
test_rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
print(f"1. Global Mean: {mean_val:.4f} | Test RMSE: {test_rmse_mean:.4f}")

## 5. Robust Amortized Model (High-Gain)
print("\n" + "="*60 + "\nAMORTIZED ARD MODEL (SAE + High Gain)\n" + "="*60)

class ReluARDModel(nn.Module):
    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        
        # 1. User Abilities (Theta) - High Gain Init
        self.theta = nn.Parameter(torch.randn(N, K) * INIT_SCALE)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 2. Projection W (Maps SAE Features -> Latent Factors)
        self.W = nn.Parameter(torch.randn(K, d) * INIT_SCALE)
        
        # 3. ARD Scales (Tau) - Start Strong
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)
        
        self.difficulty_proj = nn.Linear(d, 1)

    def get_tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        # Normalize direction (Paper Eq 7)
        W_norm = F.normalize(self.W, dim=1)
        
        # Base Loadings (J, K)
        base_loadings = self.x_j @ W_norm.T
        
        # Apply Sparsity Scale (ARD)
        tau = self.get_tau()
        a_j = base_loadings * tau.unsqueeze(0)
        
        if self.training and self.dropout > 0:
            a_j = F.dropout(a_j, p=self.dropout)
            
        # Predict: Theta * a_j + bias terms
        diff = self.difficulty_proj(self.x_j).squeeze()
        logits = self.theta @ a_j.T + diff.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        
        return torch.sigmoid(logits)

# Initialize with SAE Feature count
model = ReluARDModel(N, J, K_MODEL, SAE_FEATURES, x_j, dropout=DROPOUT_RATE).to(device)

optimizer = optim.AdamW([
    {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
    {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
    {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
])

best_test_rmse = float('inf')
best_active_dims = float('inf')
best_state = None
patience_counter = 0
history = {'epoch': [], 'train': [], 'test': []}

print(f"Starting Training (Features={SAE_FEATURES}, Init={INIT_SCALE})...")

for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    probs = model()
    
    # LOSS 1: BCE (Robust for probabilities)
    loss_fit = F.binary_cross_entropy(probs[train_mask_t], y_empirical[train_mask_t])
    
    # LOSS 2: Annealed Sparsity
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
    
    # SNAPPING
    if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
        with torch.no_grad():
            active_mask = model.get_tau() > SNAPPING_THRESHOLD
            for k in range(K_MODEL):
                if not active_mask[k]:
                    model.tau_raw[k] = DEAD_ZONE_VALUE

    # EVALUATION
    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            p_test = model()
            train_rmse = compute_rmse(p_test.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
            test_rmse = compute_rmse(p_test.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
            
            history['epoch'].append(epoch)
            history['train'].append(train_rmse)
            history['test'].append(test_rmse)
            
            tau_vals = model.get_tau().cpu().numpy()
            active_dims = (tau_vals > SNAPPING_THRESHOLD).sum()
            phase = "WARM" if epoch < TAU_WARMUP else "RUN"
            
            print(f"Epoch {epoch:4d} [{phase}] | Train: {train_rmse:.4f} Test: {test_rmse:.4f} | "
                  f"Active: {active_dims}/{K_MODEL} | MaxTau: {tau_vals.max():.3f}")
            
            is_better = False
            if active_dims < best_active_dims:
                is_better = True
            elif active_dims == best_active_dims and test_rmse < best_test_rmse - MIN_DELTA:
                is_better = True

            if is_better:
                best_test_rmse = test_rmse
                best_active_dims = active_dims
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch > TAU_WARMUP:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f">> Early stopping at epoch {epoch}")
                    break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Restored best model (Active Dims: {best_active_dims}/{K_MODEL}, Test RMSE: {best_test_rmse:.4f})")

## 6. Analysis & Visualization
print("\n" + "="*60 + "\nREGRESSION ANALYSIS & PLOTS\n" + "="*60)

# Collect Data
model.eval()
with torch.no_grad():
    y_pred = model().cpu().numpy()
    y_true = y_empirical.cpu().numpy()

# Masking
flat_pred_train = y_pred[train_mask]
flat_true_train = y_true[train_mask]
flat_pred_test = y_pred[test_mask]
flat_true_test = y_true[test_mask]
res_test = flat_true_test - flat_pred_test

# Metrics
mae_test = mean_absolute_error(flat_true_test, flat_pred_test)
r2_test = r2_score(flat_true_test, flat_pred_test)
pearson_corr, _ = pearsonr(flat_true_test, flat_pred_test)

print(f"METRICS (Test Set):")
print(f"  RMSE    : {best_test_rmse:.4f}")
print(f"  MAE     : {mae_test:.4f}")
print(f"  R^2     : {r2_test:.4f}")
print(f"  Pearson : {pearson_corr:.4f}")

# Plotting
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3)

# 1. Learning Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['epoch'], history['train'], label='Train RMSE')
ax1.plot(history['epoch'], history['test'], label='Test RMSE')
ax1.set_title('Learning Curve')
ax1.legend()

# 2. Test Parity
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(flat_true_test, flat_pred_test, alpha=0.2, color='gray', s=10)
try:
    sns.kdeplot(x=flat_true_test, y=flat_pred_test, levels=5, color='tab:blue', ax=ax2)
except: pass
ax2.plot([0, 1], [0, 1], 'r--')
ax2.set_title(f'Test Parity ($R^2={r2_test:.2f}$)')
ax2.set_xlim(0,1); ax2.set_ylim(0,1)

# 3. Calibration
from sklearn.calibration import calibration_curve
ax3 = fig.add_subplot(gs[0, 2])
prob_true, prob_pred = calibration_curve(flat_true_test > 0.5, flat_pred_test, n_bins=10, strategy='uniform')
ax3.plot(prob_pred, prob_true, marker='o')
ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax3.set_title('Calibration')

# 4. Residuals
ax4 = fig.add_subplot(gs[1, 0])
sns.histplot(res_test, kde=True, ax=ax4)
ax4.set_title('Residuals')

# 5. Latent Factors
ax5 = fig.add_subplot(gs[1, 1])
with torch.no_grad():
    taus = model.get_tau().cpu().numpy()
colors = ['tab:blue' if t > SNAPPING_THRESHOLD else 'lightgray' for t in taus]
ax5.bar(np.arange(len(taus)), taus, color=colors)
ax5.set_title(f'Latent Factor Strength')

plt.tight_layout()
plt.savefig('sae_robust_analysis.png', dpi=150)
print("\nVisualization saved to 'sae_robust_analysis.png'")
plt.show()

# --------------------------------------------------------------------------
# Export metrics for summary
# --------------------------------------------------------------------------
train_rmse_final = history['train'][-1] if history['train'] else float('nan')
test_rmse_final = history['test'][-1] if history['test'] else float('nan')

if flat_pred_train.size >= 2:
    train_corr = float(np.corrcoef(flat_pred_train, flat_true_train)[0, 1])
else:
    train_corr = float('nan')

if flat_pred_test.size >= 2:
    test_corr = float(np.corrcoef(flat_pred_test, flat_true_test)[0, 1])
else:
    test_corr = float('nan')


def get_run_metrics():
    return {
        'model': 'sae_irt',
        'train_rmse': float(train_rmse_final),
        'test_rmse': float(best_test_rmse),
        'train_corr': float(train_corr),
        'test_corr': float(test_corr),
        'mae_test': float(mae_test),
        'r2_test': float(r2_test),
        'pearson_test': float(pearson_corr),
    }