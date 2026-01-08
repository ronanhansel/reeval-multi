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
import sys
import os

from hypothesaes.quickstart import train_sae

# Suppress numpy deprecation warnings from pickled DataFrames
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*numpy.core.numeric.*')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. PARSE ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description='Train MIRT model on benchmark data')
parser.add_argument('--benchmark', type=str, nargs='+', default=None,
                    help='Filter by benchmark names (e.g., corebench_hard scienceagentbench assistantbench)')
# [NEW] SAE Hyperparameters
parser.add_argument('--sae_m', type=int, default=2048, help='SAE expansion dimension (Number of sparse features)')
parser.add_argument('--sae_k', type=int, default=32, help='SAE sparsity (Number of active features per item)')
parser.add_argument('--train_sae', action='store_true', default=True, help='Whether to train SAE or use raw embeddings')
args = parser.parse_args()

# ==========================================
# 1. LOAD ACTUAL DATA
# ==========================================
print("Loading actual data from CSV files...")

# Load binary success rate (y_data)
print("Loading result matrix...")
y_df = pd.read_csv('data/result_matrix_merged.csv', index_col=0)
print(f"Original result matrix shape: {y_df.shape}")

# Load embeddings
print("Loading embeddings...")
emb_df = pd.read_pickle('data/all_benchmarks_embeddings.pkl')
print(f"Embeddings shape: {emb_df.shape}")

# Load z_data (behavioral attributes)
z_names = ['environmentalbarrier', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']
z_df_list = []

print("Loading rubrics matrices...")
for z_name in z_names:
    z_df = pd.read_csv(f'data/rubrics_matrix_{z_name}.csv', index_col=0)
    z_df_list.append(z_df)

# ==========================================
# 2. FILTER BY BENCHMARK IF SPECIFIED
# ==========================================
if args.benchmark:
    print(f"\nFiltering by benchmarks: {args.benchmark}")
    filtered_cols = []
    for col in y_df.columns:
        if any(bench in str(col) for bench in args.benchmark):
            filtered_cols.append(col)
    
    y_df = y_df[filtered_cols]
    for i in range(len(z_df_list)):
        z_df_list[i] = z_df_list[i][filtered_cols]

# ==========================================
# 3. FILTER ROWS/COLS (Clean Data)
# ==========================================
print("\nFiltering rows with no valid responses...")
valid_rows = y_df.notna().any(axis=1)
y_df = y_df[valid_rows]
for i in range(len(z_df_list)):
    z_df_list[i] = z_df_list[i][valid_rows]

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
for i in range(len(z_df_list)):
    z_df_list[i] = z_df_list[i].loc[:, valid_cols_mask]

# ==========================================
# 3.5. CREATE TRAIN/TEST SPLIT (ITEM-WISE / COLD START)
# ==========================================
print("\nCreating train/test split (ITEM-WISE / COLD START)...")

# Prepare full data
y_data = y_df.values.astype(np.float32)
y_mask = ~np.isnan(y_data)

N, J = y_data.shape

# 1. Identify all valid items (columns that have at least one response)
# (We already filtered columns in 3.7, so indices 0..J-1 are valid)
all_item_indices = np.arange(J)

# 2. Randomly select 10% of ITEMS to be completely invisible during training
n_test_items = int(0.1 * J)
n_train_items = J - n_test_items

np.random.shuffle(all_item_indices)
test_item_indices = all_item_indices[:n_test_items]
train_item_indices = all_item_indices[n_test_items:]

print(f"Total Items: {J}")
print(f"Training on {n_train_items} items (columns)")
print(f"Testing on {n_test_items} NEW items (columns)")

# 3. Create Masks
# Train Mask: Active ONLY for training items (and only where data exists)
train_mask = np.zeros_like(y_mask, dtype=bool)
train_mask[:, train_item_indices] = y_mask[:, train_item_indices]

# Test Mask: Active ONLY for test items (and only where data exists)
test_mask = np.zeros_like(y_mask, dtype=bool)
test_mask[:, test_item_indices] = y_mask[:, test_item_indices]

# 4. Handle "Cold Start" logic for delta_j
# WARNING: The standard model will learn delta_j=0 for test items (since no grad).
# This is actually correct for evaluation: it forces the model to predict 
# based ONLY on the features (a_j) + average difficulty (0.0).

print(f"Train entries: {train_mask.sum()}")
print(f"Test entries: {test_mask.sum()}")

# Convert to tensors
y_data = np.nan_to_num(y_data, nan=0.0)
y_data = torch.from_numpy(y_data)
train_mask = torch.from_numpy(train_mask)
test_mask = torch.from_numpy(test_mask)

# ==========================================
# 4. PREPARE Z_DATA
# ==========================================
z_data_list = []
z_mask_list = []
for z_df in z_df_list:
    z_matrix = z_df.values.astype(np.float32)
    z_mask_mat = ~np.isnan(z_matrix)
    z_data_list.append(torch.from_numpy(np.nan_to_num(z_matrix, nan=0.0)))
    z_mask_list.append(torch.from_numpy(z_mask_mat))

z_data = torch.stack(z_data_list, dim=2)
z_mask = torch.stack(z_mask_list, dim=2)

# ==========================================
# 5. PREPARE EMBEDDINGS (X_J)
# ==========================================
print("\nPreparing embeddings...")
emb_mapping = {}
for idx, row in emb_df.iterrows():
    key = str(row['benchmark.task_id'])
    emb = row['embedding']
    if isinstance(emb, str): emb = ast.literal_eval(emb)
    emb_mapping[key] = np.array(emb, dtype=np.float32)

embeddings_list = []
matched_count = 0
for col in y_df.columns:
    col_str = str(col)
    if col_str in emb_mapping:
        embeddings_list.append(emb_mapping[col_str])
        matched_count += 1
    else:
        embeddings_list.append(np.zeros(512, dtype=np.float32)) # Assuming 512 dim

# Initial Dense Embeddings
x_j_input = np.array(embeddings_list, dtype=np.float32)
x_j_input = torch.from_numpy(x_j_input)
x_j_input = F.normalize(x_j_input, p=2, dim=1)

print(f"Matched {matched_count}/{len(y_df.columns)} embeddings")

# ==========================================
# [NEW] 5.5. TRAIN SAE & TRANSFORM
# ==========================================
if args.train_sae:
    print(f"\n=== TRAINING SPARSE AUTOENCODER (SAE) ===")
    print(f"Configuration: M={args.sae_m}, K={args.sae_k}")
    
    # 1. Train SAE
    # We pass the numpy array of embeddings. 
    # Checkpoint dir saves the model so you don't retrain every time.
    sae_model = train_sae(
        embeddings=x_j_input.numpy(),
        M=args.sae_m,
        K=args.sae_k,
        batch_size=512,
        n_epochs=100,  # Adjust based on convergence
        learning_rate=5e-4,
        checkpoint_dir='sae_checkpoints',
        show_progress=True
    )
    
    # 2. Get Sparse Activations
    print("Transforming dense embeddings to sparse features...")
    # This returns a numpy array of shape (J, SAE_M)
    sparse_activations = sae_model.get_activations(x_j_input)
    
    # 3. Replace Input Features
    # We replace the dense vectors with the sparse SAE activations
    x_j_input = torch.from_numpy(sparse_activations).float()
    
    # 4. Update Dimensions
    # Your model will now see 'args.sae_m' features instead of 512
    d_features = args.sae_m
    print(f"New Feature Dimension: {d_features} (Sparse)")
    
else:
    d_features = x_j_input.shape[1]
    print(f"Using raw embeddings. Feature Dimension: {d_features}")

# Get other dimensions
N, J = y_data.shape
M = z_data.shape[2]

# ==========================================
# 6. ROBUST MODEL
# ==========================================
class RobustARDModel(nn.Module):
    def __init__(self, N, J, M, K_model, d_features, x_j_input):
        super().__init__()
        self.N, self.J, self.M, self.K = N, J, M, K_model

        # Register fixed item features (These are now SAE features!)
        self.register_buffer('x_j', x_j_input)

        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)
        # W now maps from SAE Concepts -> Latent Skills
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.1)

        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)
        self.u_logits = nn.Parameter(torch.ones(M, K_model) * 2.0)
        self.delta_j = nn.Parameter(torch.zeros(J))
        self.delta_zm = nn.Parameter(torch.zeros(J, M))

    @property
    def tau(self):
        return F.relu(self.tau_raw)

    def get_gates(self, temp):
        return torch.sigmoid(self.u_logits / temp)

    def forward(self, temp=1.0):
        # Normalize W so scale is handled purely by tau
        W_norm = F.normalize(self.W, dim=1)

        # Amortized loadings (Concepts * Weights)
        base_loadings = self.x_j @ W_norm.T
        a_j = base_loadings * self.tau.unsqueeze(0)

        g_m = self.get_gates(temp)

        logits_y = self.theta @ a_j.T + self.delta_j.unsqueeze(0)
        
        logits_z_list = []
        for m in range(self.M):
            a_masked = a_j * g_m[m].unsqueeze(0)
            l_z = self.theta @ a_masked.T + self.delta_zm[:, m].unsqueeze(0)
            logits_z_list.append(l_z.unsqueeze(2))

        return logits_y, torch.cat(logits_z_list, dim=2)

# ==========================================
# 7. OPTIMIZATION LOOP
# ==========================================

K_MODEL = 50
model = RobustARDModel(N, J, M, K_MODEL, d_features, x_j_input)

# You may need to tune 'lr' since features changed scale/sparsity
optimizer = optim.Adam([
    {'params': model.tau_raw, 'lr': 0.005},
    {'params': [p for n, p in model.named_parameters() if 'tau' not in n], 'lr': 0.01}
])

hyperparams = {'lambda_tau': 50.0} 

print(f"\nStarting Robust ARD with K_model={K_MODEL}...")

for e in range(1001):
    optimizer.zero_grad()
    logits_y, logits_z = model(temp=1.0)

    # Losses
    loss_y_elements = F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none')
    loss_lik_y = (loss_y_elements * train_mask.float()).sum()

    loss_z_elements = F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none')
    z_train_mask = z_mask & train_mask.unsqueeze(2)
    loss_lik_z = (loss_z_elements * z_train_mask.float()).sum()
    
    loss_lik = loss_lik_y + loss_lik_z

    reg_tau = hyperparams['lambda_tau'] * torch.norm(model.tau, 1)
    reg_theta = 0.5 * torch.sum(model.theta ** 2)
    reg_W = 0.5 * torch.sum(model.W ** 2)

    loss = loss_lik + reg_tau + reg_theta + reg_W
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        small_indices = model.tau < 0.01
        model.tau_raw[small_indices] = -0.1

    if e % 100 == 0:
        current_tau = model.tau.detach().numpy()
        print(f"Ep {e} | Loss {loss.item():.2e} | Tau: {np.round(current_tau, 3)}")

# ==========================================
# 8. EVALUATION
# ==========================================
print("\n=== EVALUATING ON TRAIN/TEST SPLIT ===")
model.eval()
with torch.no_grad():
    logits_y, _ = model(temp=1.0)
    probs_y = torch.sigmoid(logits_y)
    
    y_test_true = torch.masked_select(y_data, test_mask).cpu().numpy()
    y_test_prob = torch.masked_select(probs_y, test_mask).cpu().numpy()
    
    y_train_true = torch.masked_select(y_data, train_mask).cpu().numpy()
    y_train_prob = torch.masked_select(probs_y, train_mask).cpu().numpy()
    
    print(f"\n[Test Set] N={len(y_test_true)}")
    if len(np.unique(y_test_true)) > 1:
        print(f"Test AUC:      {roc_auc_score(y_test_true, y_test_prob):.4f}")
        
    print(f"Test Accuracy: {np.mean((y_test_prob > 0.5).astype(int) == y_test_true):.4f}")
    print(f"Train Accuracy: {np.mean((y_train_prob > 0.5).astype(int) == y_train_true):.4f}")