import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
from sklearn.decomposition import PCA
import warnings

# Suppress numpy deprecation warnings from pickled DataFrames
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*numpy.core.numeric.*')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. LOAD ACTUAL DATA
# ==========================================
print("Loading actual data from pickles...")

# Load binary success rate (y_data)
with open('data/resmat_binary_success_rate.pkl', 'rb') as f:
    y_data = pickle.load(f)
    # Convert DataFrame to numpy if needed
    if hasattr(y_data, 'values'):
        y_data = y_data.values
    # Ensure numeric type and convert to float32
    y_data = np.array(y_data, dtype=np.float32)
    
    # Create mask for valid (non-NaN) entries
    y_mask = ~np.isnan(y_data)
    
    # Fill NaN with 0 (value doesn't matter since we'll mask it out)
    y_data = np.nan_to_num(y_data, nan=0.0)
    
    y_data = torch.from_numpy(y_data)
    y_mask = torch.from_numpy(y_mask)

# Load z_data (behavioral attributes)
z_data_list = []
z_mask_list = []
z_names = ['environmentalbarrier', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']

for z_name in z_names:
    with open(f'data/resmat_{z_name}.label.pkl', 'rb') as f:
        z_matrix = pickle.load(f)
        # Convert DataFrame to numpy if needed
        if hasattr(z_matrix, 'values'):
            z_matrix = z_matrix.values
        # Ensure numeric type and convert to float32
        z_matrix = np.array(z_matrix, dtype=np.float32)
        
        # Create mask for valid (non-NaN) entries
        z_mask = ~np.isnan(z_matrix)
        
        # Fill NaN with 0 (value doesn't matter since we'll mask it out)
        z_matrix = np.nan_to_num(z_matrix, nan=0.0)
        
        z_data_list.append(torch.from_numpy(z_matrix))
        z_mask_list.append(torch.from_numpy(z_mask))

# Stack z_data: (N, J, M)
z_data = torch.stack(z_data_list, dim=2)
z_mask = torch.stack(z_mask_list, dim=2)

# Load item features from embeddings mapping
# Use (task_id, benchmark) as key since task_ids are not globally unique
with open('data/task_bench_to_embedding.pkl', 'rb') as f:
    task_bench_to_emb = pickle.load(f)
    
print(f"Loaded embeddings for {len(task_bench_to_emb)} (task_id, benchmark) pairs")

# Get column MultiIndex from y_data to match order
with open('data/resmat_binary_success_rate.pkl', 'rb') as f:
    y_df_original = pickle.load(f)
    if hasattr(y_df_original, 'columns'):
        # Extract (task_id, benchmark) tuples from MultiIndex levels 0 and 2
        columns_info = [(str(col[0]), str(col[2])) for col in y_df_original.columns]
    else:
        raise ValueError("Cannot extract task IDs from resmat")

# Match embeddings to items in order
embeddings_list = []
matched_count = 0
for task_id, benchmark in columns_info:
    key = (task_id, benchmark)
    if key in task_bench_to_emb:
        embeddings_list.append(task_bench_to_emb[key])
        matched_count += 1
    else:
        # If no embedding found, use zero vector
        print(f"Warning: No embedding for (task_id={task_id}, benchmark={benchmark}), using zero vector")
        embeddings_list.append(np.zeros(2560, dtype=np.float32))

x_j_input = np.array(embeddings_list, dtype=np.float32)
x_j_input = torch.from_numpy(x_j_input)
# Normalize features
x_j_input = F.normalize(x_j_input, p=2, dim=1)

target_d = 10 

print(f"Reducing features from {x_j_input.shape[1]} to {target_d} via PCA...")
pca = PCA(n_components=target_d)
x_j_numpy = x_j_input.numpy()
x_j_reduced = pca.fit_transform(x_j_numpy)

# 2. Re-normalize after PCA (Crucial for the model assumption)
x_j_input = torch.from_numpy(x_j_reduced).float()
x_j_input = F.normalize(x_j_input, p=2, dim=1)

d_features = target_d # Update d_features

# Get dimensions from data
N, J = y_data.shape
M = z_data.shape[2]
d_features = x_j_input.shape[1]

print(f"Data loaded: N={N} (models), J={J} (items), M={M} (behaviors), d_features={d_features}")
print(f"Matched {matched_count}/{J} embeddings")
print(f"y_data shape: {y_data.shape}")
print(f"z_data shape: {z_data.shape}")
print(f"x_j shape: {x_j_input.shape}")
print(f"y_mask: {y_mask.sum().item()}/{y_mask.numel()} valid entries ({100*y_mask.float().mean():.1f}%)")
print(f"z_mask: {z_mask.sum().item()}/{z_mask.numel()} valid entries ({100*z_mask.float().mean():.1f}%)")

# ==========================================
# 2. ROBUST MODEL (ReLU + Normalized W)
# ==========================================
class RobustARDModel(nn.Module):
    def __init__(self, N, J, M, K_model, d_features, x_j_input):
        super().__init__()
        self.N, self.J, self.M, self.K = N, J, M, K_model

        # Register fixed item features
        self.register_buffer('x_j', x_j_input)

        # Parameters
        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.1)

        # Tau: Initialize to 0.5 so they start alive
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)

        self.u_logits = nn.Parameter(torch.ones(M, K_model) * 2.0)
        self.delta_j = nn.Parameter(torch.zeros(J))
        self.delta_zm = nn.Parameter(torch.zeros(J, M))

    @property
    def tau(self):
        # ReLU ensures exact zeros (sparsity)
        return F.relu(self.tau_raw)

    def get_gates(self, temp):
        return torch.sigmoid(self.u_logits / temp)

    def forward(self, temp=1.0):
        # Normalize W so scale is handled purely by tau
        W_norm = F.normalize(self.W, dim=1)

        # Amortized loadings
        base_loadings = self.x_j @ W_norm.T
        a_j = base_loadings * self.tau.unsqueeze(0)

        g_m = self.get_gates(temp)

        # Overall prediction
        logits_y = self.theta @ a_j.T + self.delta_j.unsqueeze(0)

        # Subskill prediction
        logits_z_list = []
        for m in range(self.M):
            a_masked = a_j * g_m[m].unsqueeze(0)
            l_z = self.theta @ a_masked.T + self.delta_zm[:, m].unsqueeze(0)
            logits_z_list.append(l_z.unsqueeze(2))

        return logits_y, torch.cat(logits_z_list, dim=2)

# ==========================================
# 3. OPTIMIZATION LOOP
# ==========================================

K_MODEL = 100
model = RobustARDModel(N, J, M, K_MODEL, d_features, x_j_input)

# Separate parameter groups:
# We generally want a smaller LR for the structure (tau) to prevent oscillation
optimizer = optim.Adam([
    {'params': model.tau_raw, 'lr': 0.005},  # Slower learning for ARD
    {'params': [p for n, p in model.named_parameters() if 'tau' not in n], 'lr': 0.01}
])

# CRITICAL: Higher Lambda to overcome N*J likelihood sum
# Rule of thumb: Lambda ~= 1.5 * N often works for factor models
# Adjusted for smaller dataset size (N=46)
hyperparams = {'lambda_tau': 50.0}

print(f"\nStarting Robust ARD with K_model={K_MODEL}...")
print(f"Using Lambda Tau: {hyperparams['lambda_tau']}")

for e in range(1001):
    optimizer.zero_grad()

    # Forward
    logits_y, logits_z = model(temp=1.0)

    # Loss: Likelihood with masking (only compute loss on valid entries)
    # Compute element-wise loss
    loss_y_elements = F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none')
    loss_z_elements = F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none')
    
    # Apply masks and sum only valid entries
    loss_lik = (loss_y_elements * y_mask.float()).sum() + (loss_z_elements * z_mask.float()).sum()

    # Loss: L1 Penalty on Tau
    reg_tau = hyperparams['lambda_tau'] * torch.norm(model.tau, 1)

    # Loss: Regularization on other params
    reg_theta = 0.5 * torch.sum(model.theta ** 2)
    reg_W = 0.5 * torch.sum(model.W ** 2)

    loss = loss_lik + reg_tau + reg_theta + reg_W
    loss.backward()
    optimizer.step()

    # --- SNAPPING TRICK ---
    # If a dimension is very small, kill it to allow ReLU to keep it dead.
    with torch.no_grad():
        # If tau < 0.01, set the raw parameter to -0.1 (dead ReLU region)
        small_indices = model.tau < 0.01
        model.tau_raw[small_indices] = -0.1

    if e % 100 == 0:
        current_tau = model.tau.detach().numpy()
        print(f"Ep {e} | Loss {loss.item():.2e} | Tau: {np.round(current_tau, 3)}")

# ==========================================
# 4. FINAL RESULTS
# ==========================================
print("\n--- FINAL SCALES ---")
final_tau = model.tau.detach().numpy()
sorted_tau = np.sort(final_tau)[::-1]

print(f"Sorted Tau: {np.round(sorted_tau, 3)}")

# Count effective dimensions (strictly positive)
effective_k = np.sum(final_tau > 0.0)
print(f"Effective Dimension: {effective_k}")
print(f"\nBehavior names: {z_names}")
print(f"Gates (u_logits) shape: {model.u_logits.shape}")
