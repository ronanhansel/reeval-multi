import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import argparse
import ast
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. SETUP & REPRODUCIBILITY
# ==========================================
# CRITICAL: Keep seed 42 to match train_full.py split exactly
torch.manual_seed(42)
np.random.seed(42)

# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Train Baselines')
parser.add_argument('--benchmark', type=str, nargs='+', default=None,
                    help='Filter by benchmark names (e.g., corebench_hard scienceagentbench assistantbench)')
args = parser.parse_args()

# ==========================================
# 1. LOAD ACTUAL DATA
# ==========================================
print("Loading actual data from CSV files...")

# Load binary success rate (y_data)
print("Loading result matrix...")
y_df = pd.read_csv('data/result_matrix_merged.csv', index_col=0)

# Load embeddings
print("Loading embeddings...")
emb_df = pd.read_pickle('data/all_benchmarks_embeddings.pkl')

# ==========================================
# 2. FILTER BY BENCHMARK IF SPECIFIED
# ==========================================
if args.benchmark:
    print(f"\nFiltering by benchmarks: {args.benchmark}")
    filtered_cols = []
    for col in y_df.columns:
        if any(bench in str(col) for bench in args.benchmark):
            filtered_cols.append(col)
    
    print(f"Found {len(filtered_cols)} columns matching benchmarks")
    y_df = y_df[filtered_cols]

# ==========================================
# 3. FILTER ROWS & COLS
# ==========================================
print("\nFiltering rows with no valid responses...")
valid_rows = y_df.notna().any(axis=1)
y_df = y_df[valid_rows]

print("\nFiltering columns with all false (0) or all null...")
valid_cols_mask = []
for col in y_df.columns:
    col_data = y_df[col]
    has_non_null = col_data.notna().any()
    if has_non_null:
        has_non_zero = (col_data.dropna() != 0).any()
        valid_cols_mask.append(has_non_zero)
    else:
        valid_cols_mask.append(False)

valid_cols_mask = np.array(valid_cols_mask)
y_df = y_df.loc[:, valid_cols_mask]

print(f"Final result matrix shape: {y_df.shape}")

# ==========================================
# 3.5. CREATE TRAIN/TEST SPLIT (ITEM-WISE / COLD START)
# ==========================================
print("\nCreating train/test split (ITEM-WISE / COLD START)...")

y_data = y_df.values.astype(np.float32)
y_mask = ~np.isnan(y_data)

N, J = y_data.shape
all_item_indices = np.arange(J)

# 10% Item Holdout
n_test_items = int(0.1 * J)
n_train_items = J - n_test_items

np.random.shuffle(all_item_indices)
test_item_indices = all_item_indices[:n_test_items]
train_item_indices = all_item_indices[n_test_items:]

print(f"Total Items: {J}. Train Items: {n_train_items}. Test Items: {n_test_items}")

train_mask = np.zeros_like(y_mask, dtype=bool)
train_mask[:, train_item_indices] = y_mask[:, train_item_indices]

test_mask = np.zeros_like(y_mask, dtype=bool)
test_mask[:, test_item_indices] = y_mask[:, test_item_indices]

print(f"Train entries: {train_mask.sum()}")
print(f"Test entries: {test_mask.sum()}")

y_data = np.nan_to_num(y_data, nan=0.0)
y_data = torch.from_numpy(y_data).to(device)
train_mask = torch.from_numpy(train_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)

# ==========================================
# 4. LOAD RUBRICS & FILTER ENVIRONMENTAL BARRIER
# ==========================================
print("\nLoading rubrics data...")
z_names = ['environmentalbarrier', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']
z_df_list = [pd.read_csv(f'data/rubrics_matrix_{z_name}.csv', index_col=0) for z_name in z_names]

# Apply same filtering as y_df
if args.benchmark:
    z_df_list = [df[filtered_cols] for df in z_df_list]

for i in range(len(z_df_list)):
    z_df_list[i] = z_df_list[i][valid_rows]
    z_df_list[i] = z_df_list[i].loc[:, valid_cols_mask]

# Create z_data tensor
z_data = torch.stack([torch.from_numpy(np.nan_to_num(df.values, nan=0.0)).float() for df in z_df_list], dim=2).to(device)

# Filter out environmentalbarrier == 1 (treat as missing)
print("\nFiltering environmentalbarrier == 1...")
env_barrier_idx = z_names.index('environmentalbarrier')
env_barrier_mask = z_data[:, :, env_barrier_idx] == 1
print(f"Masking {env_barrier_mask.sum().item()} entries where environmentalbarrier == 1")

# Apply mask to training/test masks
train_mask = train_mask & ~env_barrier_mask
test_mask = test_mask & ~env_barrier_mask

# ==========================================
# 5. PREPARE EMBEDDINGS
# ==========================================
emb_mapping = {}
for idx, row in emb_df.iterrows():
    key = str(row['benchmark.task_id'])
    emb = row['embedding']
    if isinstance(emb, str): emb = ast.literal_eval(emb)
    emb_mapping[key] = np.array(emb, dtype=np.float32)

embeddings_list = []
for col in y_df.columns:
    col_str = str(col)
    if col_str in emb_mapping:
        embeddings_list.append(emb_mapping[col_str])
    else:
        embeddings_list.append(np.zeros(512, dtype=np.float32))

x_j_input = torch.from_numpy(np.array(embeddings_list, dtype=np.float32))
x_j_input = F.normalize(x_j_input, p=2, dim=1).to(device)
d_features = x_j_input.shape[1]

# Helper function
def evaluate_preds(probs, name):
    print(f"\n=== EVALUATING {name} ===")
    
    y_test_true = torch.masked_select(y_data, test_mask).cpu().numpy()
    y_test_prob = torch.masked_select(probs, test_mask).detach().cpu().numpy()
    
    print(f"\n[Test Set] N={len(y_test_true)}")
    if len(np.unique(y_test_true)) > 1:
        print(f"Test AUC:      {roc_auc_score(y_test_true, y_test_prob):.4f}")
    else:
        print("Test AUC:      Undefined")
    print(f"Test Accuracy: {np.mean((y_test_prob > 0.5).astype(int) == y_test_true):.4f}")

    y_train_true = torch.masked_select(y_data, train_mask).cpu().numpy()
    y_train_prob = torch.masked_select(probs, train_mask).detach().cpu().numpy()
    
    print(f"\n[Train Set] N={len(y_train_true)}")
    if len(np.unique(y_train_true)) > 1:
        print(f"Train AUC:     {roc_auc_score(y_train_true, y_train_prob):.4f}")
    print(f"Train Accuracy: {np.mean((y_train_prob > 0.5).astype(int) == y_train_true):.4f}")
    print("-" * 30)

# ==========================================
# BASELINE 1: NAIVE MEAN PREDICTIONS
# ==========================================
print("\n--- BASELINE 1: NAIVE MEANS ---")

y_train_only = y_data * train_mask.float()

# 1. Global Mean
global_mean = y_train_only.sum() / train_mask.float().sum()
pred_global = torch.ones_like(y_data) * global_mean
evaluate_preds(pred_global, "Naive Global Mean")

# 2. Item Mean (Will fail on Test)
item_sum = y_train_only.sum(dim=0)
item_count = train_mask.float().sum(dim=0)
item_mean = torch.where(item_count > 0, item_sum / item_count, global_mean)
pred_item = item_mean.unsqueeze(0).expand(N, J)
evaluate_preds(pred_item, "Naive Item Mean")

# 3. User Mean (Ability)
user_sum = y_train_only.sum(dim=1)
user_count = train_mask.float().sum(dim=1)
user_mean = torch.where(user_count > 0, user_sum / user_count, global_mean)
pred_user = user_mean.unsqueeze(1).expand(N, J)
evaluate_preds(pred_user, "Naive User Mean")

# ==========================================
# BASELINE 2: STANDARD RASCH (Free Params)
# ==========================================
print("\n--- BASELINE 2: STANDARD RASCH (Free Params) ---")
# This baseline cannot predict difficulty for unseen items (Cold Start).
# It serves as a control to show why amortization is needed.

class FreeRaschModel(nn.Module):
    def __init__(self, N, J):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(N))
        self.delta = nn.Parameter(torch.zeros(J))
        
    def forward(self):
        return self.theta.unsqueeze(1) + self.delta.unsqueeze(0)

rasch_model = FreeRaschModel(N, J).to(device)
rasch_opt = optim.Adam(rasch_model.parameters(), lr=0.05)

for e in range(1001):
    rasch_opt.zero_grad()
    logits = rasch_model()
    loss = (F.binary_cross_entropy_with_logits(logits, y_data, reduction='none') * train_mask.float()).sum()
    loss.backward()
    rasch_opt.step()

evaluate_preds(torch.sigmoid(rasch_model()), "Standard Rasch (Free)")

# ==========================================
# BASELINE 3: LINEAR AMORTIZED MODEL (K=1)
# ==========================================
print("\n--- BASELINE 3: LINEAR AMORTIZED MODEL (K=1) ---")
# This mimics our 'LinearRobustARD' but with only 1 latent dimension.
# It effectively tests: "Does the linear difficulty projection work?"

class LinearAmortizedBaseline(nn.Module):
    def __init__(self, N, J, K_model, d_features, x_j_input, dropout_p=0.5):
        super().__init__()
        self.register_buffer('x_j', x_j_input)
        
        # [FEATURE 1] Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        
        # User Ability (Theta)
        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)
        
        # Item Loading Projection (W)
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.1)
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 1.0)
        
        # [FEATURE 2] Amortized Difficulty (Linear)
        # Replaces self.delta_j parameter to allow Cold Start prediction
        self.difficulty_proj = nn.Linear(d_features, 1)

    @property
    def tau(self): return F.relu(self.tau_raw)

    def forward(self):
        x_dropped = self.dropout(self.x_j)
        
        # Linear Difficulty
        pred_delta = self.difficulty_proj(x_dropped).squeeze().unsqueeze(0)
        
        # Linear Loadings
        W_norm = F.normalize(self.W, dim=1)
        a_j = (x_dropped @ W_norm.T) * self.tau.unsqueeze(0)
        
        logits = self.theta @ a_j.T + pred_delta
        return logits

# Initialize with K=1 (Scalar Latent Ability)
amortized_model = LinearAmortizedBaseline(N, J, K_model=1, d_features=d_features, x_j_input=x_j_input, dropout_p=0.5).to(device)

# [FEATURE 3] Optimizer with L2 (Weight Decay)
optimizer = optim.Adam([
    # No decay on Tau
    {'params': amortized_model.tau_raw, 'lr': 0.01, 'weight_decay': 0.0},
    # Decay on projections (W and Difficulty)
    {'params': list(amortized_model.difficulty_proj.parameters()) + [amortized_model.W], 
     'lr': 0.005, 'weight_decay': 1e-2},
    # Decay on Theta
    {'params': amortized_model.theta, 'lr': 0.01, 'weight_decay': 1e-4}
])

for e in range(1001):
    amortized_model.train()
    optimizer.zero_grad()
    
    logits = amortized_model()
    
    loss_elem = F.binary_cross_entropy_with_logits(logits, y_data, reduction='none')
    loss = (loss_elem * train_mask.float()).sum()
    
    # Regularization (L1 on Tau, L2 on Theta)
    # L2 on W/Difficulty is handled by optimizer weight_decay
    reg = 0.5 * torch.sum(amortized_model.theta**2)
    
    (loss + reg).backward()
    optimizer.step()
    
    if e % 200 == 0:
        with torch.no_grad():
            print(f"Ep {e} | Loss: {loss.item():.2e}")

amortized_model.eval()
evaluate_preds(torch.sigmoid(amortized_model()), "Amortized Model (K=1)")