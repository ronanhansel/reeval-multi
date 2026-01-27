# ==========================================================================
# CONFIGURATION - AMORTIZED BETA-IRT (Column-Wise Zero-Shot)
# ==========================================================================
TEST_SIZE = 0.1         # 10% of Tasks held out entirely
RANDOM_SEED = 42

# ARCHITECTURE
K_MODEL = 32            # High capacity allowed (embeddings constrain it)
EMBEDDING_DIM = 4096    # Full Dimension (No PCA)

# SPARSITY (ARD)
LAMBDA_TAU = 0.05       # Strong penalty for amortized model
SNAPPING_THRESHOLD = 0.01 
DEAD_ZONE_VALUE = -0.1

# TRAINING
EPOCHS = 2000
EVAL_EVERY = 50
PATIENCE = 50
MIN_DELTA = 1e-6

# OPTIMIZATION
LR_THETA = 0.02
LR_GLOBAL = 0.005       # W, Tau, Delta
WD_THETA = 0.01
WD_W = 0.001

## 1. Setup
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")

if torch.cuda.is_available(): device = torch.device('cuda')
elif torch.backends.mps.is_available(): device = torch.device('mps')
else: device = torch.device('cpu')
print(f"Using device: {device}")

## 2. Load Data & Embeddings
resmat_dir = 'resmats'
files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
all_dfs = [pd.read_csv(os.path.join(resmat_dir, f), index_col=0) for f in files]

shared_indices = sorted(list(set.intersection(*[set(df.index) for df in all_dfs])))
filtered_dfs = [df.loc[shared_indices] for df in all_dfs]
prob_matrix = np.nanmean(np.array([df.values for df in filtered_dfs]), axis=0)
prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

print(f"Data: {prob_df.shape[0]} Models x {prob_df.shape[1]} Tasks")

# Load Embeddings
emb_file = 'result/all_benchmarks_embeddings_4096_8B.pkl'
if os.path.exists(emb_file):
    emb_df = pd.read_pickle(emb_file)
    emb_map = {str(r['benchmark.task_id']): r['embedding'] for _, r in emb_df.iterrows()}
    
    raw_embs = []
    for task_id in prob_df.columns:
        emb = emb_map.get(str(task_id))
        if emb is None and str(task_id).startswith('colbench.'):
             emb = emb_map.get(f'colbench_backend_programming.{str(task_id).split(".")[-1]}')
        
        if isinstance(emb, str): emb = ast.literal_eval(emb)
        if emb is None: emb = np.zeros(EMBEDDING_DIM)
        raw_embs.append(emb)
    
    # Normalize Full Embeddings (No PCA)
    x_np = np.array(raw_embs, dtype=np.float32)
    norms = np.linalg.norm(x_np, axis=1, keepdims=True)
    x_np = x_np / (norms + 1e-8)
    
    x_j_all = torch.tensor(x_np).to(device)
    EMBEDDING_DIM = x_j_all.shape[1]
    print(f"Embeddings: {x_j_all.shape}")
else:
    raise FileNotFoundError("Embeddings required for Column-Wise Holdout!")

# --- COLUMN-WISE SPLIT ---
print(f"\n[Preprocessing] Generating {int(TEST_SIZE*100)}% Column-Wise Split...")
N, J = prob_df.shape
J_indices = np.arange(J)
np.random.shuffle(J_indices)

n_test = int(J * TEST_SIZE)
test_col_idx = J_indices[:n_test]
train_col_idx = J_indices[n_test:]

# Create Masks (Rectangular blocks)
y_empirical = torch.from_numpy(prob_df.values.astype(np.float32)).to(device)
train_mask = torch.zeros_like(y_empirical, dtype=torch.bool)
test_mask = torch.zeros_like(y_empirical, dtype=torch.bool)

train_mask[:, train_col_idx] = ~torch.isnan(y_empirical[:, train_col_idx])
test_mask[:, test_col_idx] = ~torch.isnan(y_empirical[:, test_col_idx])

print(f"Train Columns: {len(train_col_idx)} | Test Columns: {len(test_col_idx)}")

## 3. Baselines (Computed on Train Columns only)
mean_val = torch.nanmean(y_empirical[train_mask])
pred_mean = mean_val.expand_as(y_empirical)
# RMSE only on test columns
valid_test = test_mask.cpu().numpy()
if valid_test.sum() > 0:
    test_rmse_mean = np.sqrt(mean_squared_error(
        y_empirical.cpu().numpy()[valid_test], 
        pred_mean.cpu().numpy()[valid_test]
    ))
    print(f"Baseline Mean RMSE (Zero-Shot): {test_rmse_mean:.4f}")
else:
    print("Baseline: No valid test data.")

## 4. Amortized Beta-IRT Model
class AmortizedBetaIRT(nn.Module):
    def __init__(self, N, K, d_emb, x_j_emb):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        
        # Free Parameters (Users)
        self.theta = nn.Parameter(torch.randn(N, K) * 0.1)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Amortized Parameters (Items)
        self.W = nn.Parameter(torch.randn(K, d_emb) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K) * 0.5)
        self.diff_proj = nn.Linear(d_emb, 1)

    @property
    def tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        # Normalize W (Paper Eq 7)
        W_norm = F.normalize(self.W, p=2, dim=1)
        
        # Predict parameters for ALL columns (Train + Test) using embeddings
        # Zero-shot capability comes from here: x_j contains info for test cols
        base_loadings = self.x_j @ W_norm.T
        a_j = base_loadings * self.tau.unsqueeze(0)
        
        delta = self.diff_proj(self.x_j).squeeze()
        
        # Interaction
        logits = self.theta @ a_j.T + delta.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        return torch.sigmoid(logits)

model = AmortizedBetaIRT(N, K_MODEL, EMBEDDING_DIM, x_j_all).to(device)

# Optimizers
opt_struct = optim.Adam([
    {'params': model.tau_raw, 'lr': 0.005},
    {'params': [model.W, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
    {'params': model.diff_proj.parameters(), 'lr': LR_GLOBAL, 'weight_decay': WD_W}
], lr=LR_GLOBAL)

opt_users = optim.Adam([
    {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA}
], lr=LR_THETA)

## 5. Training Loop
print(f"Starting Training (K={K_MODEL})...")
best_loss = float('inf')
history = {'train': [], 'test': []}

for epoch in range(EPOCHS + 1):
    model.train()
    
    # 1. Update Users (Given fixed structure)
    opt_users.zero_grad()
    probs = model()
    loss_u = F.binary_cross_entropy(probs[train_mask], y_empirical[train_mask])
    loss_u.backward()
    opt_users.step()
    
    # 2. Update Structure (Given fixed users)
    opt_struct.zero_grad()
    probs = model() 
    loss_fit = F.binary_cross_entropy(probs[train_mask], y_empirical[train_mask])
    loss_reg = LAMBDA_TAU * torch.norm(model.tau, 1)
    (loss_fit + loss_reg).backward()
    opt_struct.step()
    
    # 3. Snapping
    with torch.no_grad():
        small_idx = model.tau < SNAPPING_THRESHOLD
        model.tau_raw[small_idx] = DEAD_ZONE_VALUE

    # Eval
    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            p = model()
            
            # Train Error (Fit)
            train_rmse = torch.sqrt(F.mse_loss(p[train_mask], y_empirical[train_mask])).item()
            
            # Test Error (Generalization to new tasks)
            # Only compute if we have valid test entries
            if test_mask.sum() > 0:
                test_rmse = torch.sqrt(F.mse_loss(p[test_mask], y_empirical[test_mask])).item()
            else:
                test_rmse = 0.0
                
            history['train'].append(train_rmse)
            history['test'].append(test_rmse)
            
            active_k = (model.tau > 0).sum().item()
            print(f"Ep {epoch:4d} | Train: {train_rmse:.4f} Test: {test_rmse:.4f} | Active K: {active_k}/{K_MODEL}")
            
            if test_rmse < best_loss - MIN_DELTA:
                best_loss = test_rmse
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Early stopping.")
                    break

## 6. Analysis
print("\n" + "="*40)
print(f"FINAL METRICS (Test Columns={len(test_col_idx)})")
print(f"RMSE: {best_loss:.4f}")

model.eval()
with torch.no_grad():
    y_pred = model().cpu().numpy()
    y_true = y_empirical.cpu().numpy()

# Masking for plots
flat_pred = y_pred[test_mask.cpu().numpy()]
flat_true = y_true[test_mask.cpu().numpy()]

if len(flat_true) > 1:
    r2 = r2_score(flat_true, flat_pred)
    corr = np.corrcoef(flat_true, flat_pred)[0,1]
    print(f"R^2 : {r2:.4f}")
    print(f"Corr: {corr:.4f}")

# Plots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].plot(history['train'], label='Train')
ax[0].plot(history['test'], label='Test (Zero-Shot)')
ax[0].axhline(test_rmse_mean, c='gray', ls='--', label='Baseline')
ax[0].set_title('RMSE Trajectory')
ax[0].legend()

if len(flat_true) > 1:
    ax[1].scatter(flat_true, flat_pred, alpha=0.2, s=10)
    ax[1].plot([0,1], [0,1], 'r--')
    ax[1].set_title(f'Test Parity (Corr={corr:.2f})')
    ax[1].set_xlabel('Empirical Prob')
    ax[1].set_ylabel('Predicted Prob')

with torch.no_grad(): taus = model.tau.cpu().numpy()
ax[2].bar(range(len(taus)), taus)
ax[2].set_title(f'Latent Factor Scales (Active={np.sum(taus>0)})')

plt.tight_layout()
plt.savefig('amortized_column_split.png')
print("Saved plots to amortized_column_split.png")