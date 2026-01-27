# ==========================================================================
# CONFIGURATION - FREE PARAMETER BETA-IRT (Standard Normal Prior)
# ==========================================================================
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ARCHITECTURE
# With Free Parameters on N=8, we must keep K small to avoid overfitting.
# If K is too large (e.g. 32), the model will memorize noise.
K_MODEL = 10

# TRAINING
EPOCHS = 2000
EVAL_EVERY = 50
PATIENCE = 50
MIN_DELTA = 1e-5        # Fixed: Defined here

# SPARSITY (ARD)
LAMBDA_TAU = 0.005       # Penalty on Tau to induce sparsity
SNAPPING_THRESHOLD = 0.01 
DEAD_ZONE_VALUE = -0.1

# OPTIMIZATION & PRIORS
# Standard Normal Prior implies L2 Regularization (Weight Decay)
# Strength of prior ~ Weight Decay value
LR_THETA = 0.05
LR_ITEMS = 0.02
WD_THETA = 0.01         # Prior on Users: Theta ~ N(0, 1)
WD_ITEMS = 0.01         # Prior on Items: A ~ N(0, 1)

## 1. Setup
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")

if torch.cuda.is_available(): device = torch.device('cuda')
elif torch.backends.mps.is_available(): device = torch.device('mps')
else: device = torch.device('cpu')
print(f"Using device: {device}")

def compute_rmse(predictions, targets, mask):
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

## 2. Load Data
resmat_dir = 'resmats'
files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
all_dfs = [pd.read_csv(os.path.join(resmat_dir, f), index_col=0) for f in files]

# Align Indices
shared_indices = sorted(list(set.intersection(*[set(df.index) for df in all_dfs])))
filtered_dfs = [df.loc[shared_indices] for df in all_dfs]
prob_matrix = np.nanmean(np.array([df.values for df in filtered_dfs]), axis=0)
prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

print(f"Data: {prob_df.shape[0]} Models x {prob_df.shape[1]} Tasks")

# Cell-wise Split
y_np = prob_df.values.astype(np.float32)
valid_idx = np.argwhere(~np.isnan(y_np))
np.random.shuffle(valid_idx)
n_test = int(len(valid_idx) * TEST_SIZE)
test_pairs = valid_idx[:n_test]
train_pairs = valid_idx[n_test:]

y_true = torch.from_numpy(y_np).to(device)
train_mask = torch.zeros_like(y_true, dtype=torch.bool)
test_mask = torch.zeros_like(y_true, dtype=torch.bool)
train_mask[train_pairs[:,0], train_pairs[:,1]] = True
test_mask[test_pairs[:,0], test_pairs[:,1]] = True

print(f"Train N: {len(train_pairs)} | Test N: {len(test_pairs)}")

## 3. Baselines
mean_val = torch.nanmean(y_true[train_mask])
pred_mean = mean_val.expand_as(y_true)
test_rmse_mean = np.sqrt(mean_squared_error(y_true[test_mask].cpu(), pred_mean[test_mask].cpu()))
print(f"Baseline Mean RMSE: {test_rmse_mean:.4f}")

## 4. Free Parameter Beta-IRT Model
class FreeBetaIRT(nn.Module):
    def __init__(self, N, J, K):
        super().__init__()
        
        # 1. User Parameters (Theta)
        # N(0,1) Prior initialization
        self.theta = nn.Parameter(torch.randn(N, K) * 0.1)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 2. Item Parameters (Free A)
        # N(0,1) Prior initialization
        self.A = nn.Parameter(torch.randn(J, K) * 0.1)
        self.delta = nn.Parameter(torch.zeros(J))
        
        # 3. Sparsity Scales (Tau)
        # Initialize at 1.0 to give free params a chance
        self.tau_raw = nn.Parameter(torch.ones(K) * 1.0)

    @property
    def tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        # Apply ARD: a_j = A_j * tau
        tau = self.tau
        a_j = self.A * tau.unsqueeze(0)
        
        # Interaction: Theta * A^T
        logits = self.theta @ a_j.T + self.delta.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        
        return torch.sigmoid(logits)

model = FreeBetaIRT(prob_df.shape[0], prob_df.shape[1], K_MODEL).to(device)

# Optimizers with Weight Decay (Implements Gaussian Prior)
opt_items = optim.AdamW([
    {'params': [model.A, model.delta, model.global_bias], 'weight_decay': WD_ITEMS},
    {'params': [model.tau_raw], 'weight_decay': 0.0} # Don't L2 decay Tau, we L1 it manually
], lr=LR_ITEMS)

opt_users = optim.AdamW([
    {'params': [model.theta, model.theta_bias], 'weight_decay': WD_THETA}
], lr=LR_THETA)

## 5. Training Loop
print(f"Starting Free-Param Training (K={K_MODEL})...")
best_loss = float('inf')
history = {'train_rmse': [], 'test_rmse': []}

for epoch in range(EPOCHS + 1):
    model.train()
    
    # 1. Update Users
    opt_users.zero_grad()
    probs = model()
    loss_u = F.binary_cross_entropy(probs[train_mask], y_true[train_mask])
    loss_u.backward()
    opt_users.step()
    
    # 2. Update Items
    opt_items.zero_grad()
    probs = model()
    loss_fit = F.binary_cross_entropy(probs[train_mask], y_true[train_mask])
    
    # L1 Penalty on Tau (Laplace Prior for Sparsity)
    loss_sparsity = LAMBDA_TAU * torch.norm(model.tau, 1)
    
    (loss_fit + loss_sparsity).backward()
    opt_items.step()
    
    # 3. Snapping
    with torch.no_grad():
        small_idx = model.tau < SNAPPING_THRESHOLD
        model.tau_raw[small_idx] = DEAD_ZONE_VALUE

    # Evaluation
    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            p = model()
            t_rmse = np.sqrt(mean_squared_error(y_true[train_mask].cpu(), p[train_mask].cpu()))
            v_rmse = np.sqrt(mean_squared_error(y_true[test_mask].cpu(), p[test_mask].cpu()))
            
            history['train_rmse'].append(t_rmse)
            history['test_rmse'].append(v_rmse)
            
            active_k = (model.tau > 0).sum().item()
            print(f"Ep {epoch:4d} | Train: {t_rmse:.4f} Test: {v_rmse:.4f} | Active K: {active_k}/{K_MODEL}")
            
            if v_rmse < best_loss - MIN_DELTA:
                best_loss = v_rmse
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Early stopping.")
                    break

## 6. Analysis
model.eval()
with torch.no_grad():
    final_p = model().cpu().numpy()
    target = y_true.cpu().numpy()

# Extract test values
y_pred_test = final_p[test_mask.cpu().numpy()]
y_true_test = target[test_mask.cpu().numpy()]

rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
r2 = r2_score(y_true_test, y_pred_test)
corr = np.corrcoef(y_true_test, y_pred_test)[0,1]

print("\n" + "="*40)
print(f"FINAL METRICS (Test N={len(y_true_test)})")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 : {r2:.4f}")
print(f"Corr: {corr:.4f}")
print("="*40)

# Plots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Learning Curve
ax[0].plot(history['train_rmse'], label='Train')
ax[0].plot(history['test_rmse'], label='Test')
ax[0].axhline(test_rmse_mean, c='gray', ls='--', label='Baseline')
ax[0].set_title('RMSE Trajectory')
ax[0].legend()

# Parity
ax[1].scatter(y_true_test, y_pred_test, alpha=0.2, s=10)
ax[1].plot([0,1], [0,1], 'r--')
ax[1].set_title(f'Test Parity (Corr={corr:.2f})')
ax[1].set_xlabel('Empirical Prob')
ax[1].set_ylabel('Predicted Prob')

# Factors
with torch.no_grad(): taus = model.tau.cpu().numpy()
ax[2].bar(range(len(taus)), taus)
ax[2].set_title(f'Latent Factor Scales (Active={np.sum(taus>0)})')

plt.tight_layout()
plt.savefig('free_beta_irt.png')
print("Saved plots to free_beta_irt.png")