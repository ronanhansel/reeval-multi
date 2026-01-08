import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import warnings
import argparse
import ast
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.manual_seed(42)
np.random.seed(42)

# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 0. SETUP & ARGS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, nargs='+', default=None)
parser.add_argument('--lambda_tau', type=float, default=10.0)
args = parser.parse_args()

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
y_df = pd.read_csv('data/result_matrix_merged.csv', index_col=0)
emb_df = pd.read_pickle('data/all_benchmarks_embeddings.pkl')

z_names = ['environmentalbarrier', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']
z_df_list = [pd.read_csv(f'data/rubrics_matrix_{z_name}.csv', index_col=0) for z_name in z_names]

# Filter Benchmarks
if args.benchmark:
    print(f"Filtering by: {args.benchmark}")
    cols = [c for c in y_df.columns if any(b in str(c) for b in args.benchmark)]
    y_df = y_df[cols]
    z_df_list = [df[cols] for df in z_df_list]

# Filter Rows/Cols
y_df = y_df[y_df.notna().any(axis=1)]
for i in range(len(z_df_list)): z_df_list[i] = z_df_list[i].loc[y_df.index]

valid_cols = []
for c in y_df.columns:
    valid_cols.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
y_df = y_df.loc[:, valid_cols]
for i in range(len(z_df_list)): z_df_list[i] = z_df_list[i].loc[:, valid_cols]

# ==========================================
# 2. ITEM-WISE SPLIT (COLD START)
# ==========================================
print("\nCreating ITEM-WISE Split (Cold Start)...")
y_vals = y_df.values.astype(np.float32)
N, J = y_vals.shape
J_indices = np.arange(J)
np.random.shuffle(J_indices)

n_test = int(0.1 * J)
test_idx = J_indices[:n_test]
train_idx = J_indices[n_test:]

train_mask = np.zeros_like(y_vals, dtype=bool)
train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]

test_mask = np.zeros_like(y_vals, dtype=bool)
test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.0)).to(device)
train_mask = torch.from_numpy(train_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)

# Z Data
z_data = torch.stack([torch.from_numpy(np.nan_to_num(df.values, nan=0.0)).float() for df in z_df_list], dim=2).to(device)
z_mask = torch.stack([torch.from_numpy((~np.isnan(df.values)).astype(bool)) for df in z_df_list], dim=2).to(device)

# Filter out environmentalbarrier == 1 (treat as missing)
print("\nFiltering environmentalbarrier == 1...")
env_barrier_idx = z_names.index('environmentalbarrier')
env_barrier_mask = z_data[:, :, env_barrier_idx] == 1
print(f"Masking {env_barrier_mask.sum().item()} entries where environmentalbarrier == 1")

# Apply mask to training/test masks and z_mask
train_mask = train_mask & ~env_barrier_mask
test_mask = test_mask & ~env_barrier_mask
z_mask = z_mask & ~env_barrier_mask.unsqueeze(2)

# ==========================================
# 3. PREPARE EMBEDDINGS (Dense Only)
# ==========================================
print("\nPreparing Embeddings...")
emb_map = {str(r['benchmark.task_id']): r['embedding'] for _, r in emb_df.iterrows()}
raw_embs = []
for c in y_df.columns:
    e = emb_map.get(str(c), np.zeros(512))
    if isinstance(e, str): e = ast.literal_eval(e)
    raw_embs.append(e)

x_j_input = torch.tensor(np.array(raw_embs), dtype=torch.float32)
x_j_input = F.normalize(x_j_input, p=2, dim=1).to(device)

d_features = x_j_input.shape[1]
M = z_data.shape[2]

# ==========================================
# 4. ROBUST LINEAR MODEL (Paper Aligned)
# ==========================================
class LinearRobustARD(nn.Module):
    def __init__(self, N, J, M, K_model, d_features, x_j_input, dropout_p=0.4):
        super().__init__()
        self.N, self.J, self.M, self.K = N, J, M, K_model
        self.register_buffer('x_j', x_j_input)
        
        # [FEATURE 1] Input Dropout (Regularization)
        # Prevents over-reliance on single embedding dimensions
        self.dropout = nn.Dropout(p=dropout_p)

        # [FEATURE 2] Latent Factors
        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)  # User Ability
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.1) # Feature Projection
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)    # Sparsity Scale
        
        # [FEATURE 3] Subskills
        self.u_logits = nn.Parameter(torch.ones(M, K_model) * 2.0)
        self.delta_m = nn.Parameter(torch.zeros(1, M)) # Global Subskill Bias (Parameter Efficient)

        # [FEATURE 4] Linear Amortized Difficulty 
        self.difficulty_proj = nn.Linear(d_features, 1)

    @property
    def tau(self):
        # Paper Eq (6): ReLU ensures exact zero sparsity
        return F.relu(self.tau_raw)

    def get_gates(self, t):
        # Paper Eq (5): Differentiable relaxation
        return torch.sigmoid(self.u_logits / t)

    def forward(self, temp=1.0):
        # 1. Corrupt Input
        x_dropped = self.dropout(self.x_j)
        
        # 2. Linear Difficulty Projection
        # (J, d) @ (d, 1) -> (1, J)
        pred_delta = self.difficulty_proj(x_dropped).squeeze().unsqueeze(0)
        
        # 3. Linear Loading Projection (Normalized)
        # Paper Eq (7): Normalize W to resolve scale indeterminacy
        W_norm = F.normalize(self.W, dim=1)
        # Paper Eq (7): a_j = tau * (W @ x_j)
        a_j = (x_dropped @ W_norm.T) * self.tau.unsqueeze(0)
        
        # 4. Overall Prediction
        # Paper Eq (1): p = sigma(theta @ a_j + delta)
        logits_y = self.theta @ a_j.T + pred_delta
        
        # 5. Subskill Prediction (Gated)
        g_m = self.get_gates(temp)
        logits_z = []
        for m in range(self.M):
            # z depends on gated loading
            lz = self.theta @ (a_j * g_m[m].unsqueeze(0)).T + self.delta_m[:, m].unsqueeze(1)
            logits_z.append(lz.unsqueeze(2))
            
        return logits_y, torch.cat(logits_z, dim=2)

# ==========================================
# 5. OPTIMIZATION
# ==========================================
K_MODEL = 50
# Dropout 0.5 is standard for linear models to force robustness
model = LinearRobustARD(N, J, M, K_MODEL, d_features, x_j_input, dropout_p=0.5).to(device)

# Optimizer with Paper-Aligned Priors
optimizer = optim.Adam([
    # Tau: No Weight Decay (L1 applied manually below) [cite: 88]
    {'params': model.tau_raw, 'lr': 0.01, 'weight_decay': 0.0},
    
    # Projections (W & Difficulty): Weight Decay = L2/Gaussian Prior [cite: 85, 92]
    {'params': list(model.difficulty_proj.parameters()) + [model.W], 
     'lr': 0.005, 'weight_decay': 1e-2},
    
    # Latents (Theta) & Gates: Standard Decay
    {'params': [model.theta, model.u_logits, model.delta_m], 
     'lr': 0.01, 'weight_decay': 1e-4}
])

hyperparams = {
    'lambda_tau': args.lambda_tau, # L1 Strength for Sparsity [cite: 141]
}

print("\nStarting Linear Robust Training...")
for e in range(2001):
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward Pass
    logits_y, logits_z = model()
    
    # 2. Re-compute gates to apply regularization 
    # (Or return them from forward, but accessing u_logits directly is easier here)
    # Paper Eq (5): g = sigmoid(u / T) [cite: 128]
    current_temp = 1.0 # Or anneal this from 1.0 -> 0.1 over epochs
    gates = torch.sigmoid(model.u_logits / current_temp)

    # 3. Calculate Missing Losses 
    # R_sparse: L1 penalty on gates (push to 0)
    reg_sparse_gates = 0.1 * torch.sum(gates)
    
    # R_beta: Entropy-like penalty (push to 0 or 1)
    # Minimizing g*(1-g) forces g to be near 0 or 1
    reg_beta_gates = 0.1 * torch.sum(gates * (1.0 - gates))

    # 4. Standard Losses (from your code)
    loss_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    loss_z = (F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none') * (z_mask & train_mask.unsqueeze(2))).sum()
    
    # Regularization on Tau and Theta
    reg_tau = hyperparams['lambda_tau'] * torch.norm(model.tau, 1)
    reg_theta = 0.5 * torch.sum(model.theta**2)
    
    # 5. Total Aggregated Loss
    # [FIX] Added gate regularizers
    loss = loss_y + loss_z + reg_tau + reg_theta + reg_sparse_gates + reg_beta_gates
    
    loss.backward()
    optimizer.step()
    
    # Zero-Snapping for Exact Sparsity [cite: 176]
    with torch.no_grad():
        model.tau_raw[model.tau < 0.01] = -0.1

    if e % 100 == 0:
        active_dims = (model.tau > 0.01).sum().item()
        print(f"Ep {e} | Loss {loss.item():.2e} | Active Dims: {active_dims}")

# ==========================================
# 6. EVALUATION
# ==========================================
print("\n=== EVALUATING (LINEAR MODEL) ===")
model.eval()
with torch.no_grad():
    logits_y, _ = model()
    probs = torch.sigmoid(logits_y)
    
    y_test = torch.masked_select(y_data, test_mask).cpu().numpy()
    p_test = torch.masked_select(probs, test_mask).cpu().numpy()
    
    y_train = torch.masked_select(y_data, train_mask).cpu().numpy()
    p_train = torch.masked_select(probs, train_mask).cpu().numpy()
    
    test_auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.0
    test_acc = np.mean((p_test > 0.5) == y_test)
    train_auc = roc_auc_score(y_train, p_train) if len(np.unique(y_train)) > 1 else 0.0
    train_acc = np.mean((p_train > 0.5) == y_train)
    
    print(f"\n[Test Set] N={len(y_test)}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    print(f"\n[Train Set] N={len(y_train)}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    
    # Save results
    import json
    results = {
        'lambda_tau': args.lambda_tau,
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'train_auc': float(train_auc),
        'train_acc': float(train_acc),
        'active_dims': int((model.tau > 0.01).sum().item()),
        'benchmarks': args.benchmark
    }
    
    benchmark_str = '_'.join(args.benchmark) if args.benchmark else 'all'
    output_file = f'lambda_tau_{args.lambda_tau}_{benchmark_str}_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print sparsity only for active (non-zero tau) dims in W after training
    W_all = model.W.detach().cpu().numpy()
    tau_active = model.tau.detach().cpu().numpy() > 0.01
    zero_counts = np.sum(np.abs(W_all) < 1e-3, axis=1)
    print("\nSparsity of ACTIVE W rows (tau > 0.01, number of near-zero elements < 1e-3):")
    for i, count in enumerate(zero_counts):
        if tau_active[i]:
            print(f"W[{i}] zeros: {count}/{W_all.shape[1]}")