# ==========================================================================
# CONFIGURATION
# ==========================================================================
USE_EMPIRICAL_BASELINE = True
TEST_SIZE = 0.1
RANDOM_SEED = 42

# MODEL
K_MODEL = 30
PCA_COMPONENTS = 48

# SPARSITY
LAMBDA_TAU = 0.002       # Reduced from 0.005 to stop mass extinction
TAU_INIT = 0.5           # Start alive
TAU_WARMUP = 200         # Epoch 0-200: No penalty
RAMP_EPOCHS = 200        # Epoch 200-400: Slowly increase penalty
SNAPPING_THRESHOLD = 0.005
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# TRAINING
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 30
MIN_DELTA = 1e-6

# OPTIMIZATION
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 1e-3          # Higher regularization on agents to prevent "chasing" noise
WD_W = 1e-5              # Keep projection free

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

warnings.filterwarnings('ignore')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")  # Better plot styling

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
print(f"Baseline mode: {'EMPIRICAL' if USE_EMPIRICAL_BASELINE else 'SINGLE BINARY'}")
print(f"Train/Test split: {int((1-TEST_SIZE)*100)}%/{int(TEST_SIZE*100)}%")
print("="*60)

# Utility functions
def compute_rmse(predictions, targets, mask):
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

## 2. Load Response Matrices
resmat_dir = 'resmats'
files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
n_samples = len(files)

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

print(f"Loaded {n_samples} response matrices")
print(f"Shared models: {len(shared_indices)}, Tasks: {prob_df.shape[1]}")

if n_samples > 1:
    per_entry_var = np.nanvar(stacked_matrix, axis=0, ddof=1)
    noise_floor_mse = np.nanmean(per_entry_var) / n_samples
    noise_floor_rmse = float(np.sqrt(noise_floor_mse))
    print(f"Empirical noise floor (RMSE): {noise_floor_rmse:.4f}")

## 3. Prepare Embeddings
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
    
    x_j_dense = torch.tensor(np.array(raw_embs), dtype=torch.float32)
    x_j_dense = F.normalize(x_j_dense, p=2, dim=1).to(device)
    print(f"Embeddings loaded: {len(raw_embs)}")
else:
    x_j_dense = torch.randn(len(prob_df.columns), 4096).to(device)
    print("Using random embeddings")

# PCA compression
print(f"\n[Preprocessing] Compressing embeddings -> {PCA_COMPONENTS} via PCA...")
x_np = x_j_dense.cpu().numpy()
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
x_pca = pca.fit_transform(x_np)
x_pca = x_pca / (np.linalg.norm(x_pca, axis=1, keepdims=True) + 1e-8)
x_j = torch.tensor(x_pca, dtype=torch.float32).to(device)
print(f"Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

# Split
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
print(f"1. Global Mean: {mean_val:.4f} | Train: {train_rmse_mean:.4f} | Test: {test_rmse_mean:.4f}")

theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
opt_rasch = torch.optim.Adam([theta, beta], lr=0.01)
for _ in range(500):
    opt_rasch.zero_grad()
    loss = F.binary_cross_entropy_with_logits((theta.unsqueeze(1)-beta.unsqueeze(0)), y_empirical, reduction='none')
    (loss * train_mask_t).sum().backward()
    opt_rasch.step()
with torch.no_grad():
    p_rasch = torch.sigmoid(theta.unsqueeze(1)-beta.unsqueeze(0))
    train_rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
    test_rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
print(f"2. Rasch-IRT  | Train: {train_rmse_rasch:.4f} | Test: {test_rmse_rasch:.4f}")

class ReluARDModel(nn.Module):
    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Projection
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        
        # ARD Scales (Initialized high to survive warmup)
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)
        
        self.difficulty_proj = nn.Linear(d, 1)

    def get_tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        # 1. Normalize Projection
        W_norm = F.normalize(self.W, dim=1)
        
        # 2. Compute Loadings
        base_loadings = self.x_j @ W_norm.T
        
        # 3. Apply Sparsity
        tau = self.get_tau()
        a_j = base_loadings * tau.unsqueeze(0)
        
        if self.training and self.dropout > 0:
            a_j = F.dropout(a_j, p=self.dropout)
            
        # 4. Predict
        diff = self.difficulty_proj(self.x_j).squeeze()
        # Note: Bias terms help absorb "easy" variance so factors focus on real structure
        logits = self.theta @ a_j.T + diff.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        
        return torch.sigmoid(logits)

# ==========================================
# TRAINING LOOP (Annealed)
# ==========================================
model = ReluARDModel(N, J, K_MODEL, PCA_COMPONENTS, x_j, dropout=0.5).to(device)

optimizer = optim.AdamW([
    {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
    {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
    {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
])

best_test_rmse = float('inf')
best_state = None
patience_counter = 0
history = {'epoch': [], 'train': [], 'test': []}

print(f"Starting Training (Warmup={TAU_WARMUP}, Ramp={RAMP_EPOCHS}, Lambda={LAMBDA_TAU})...")

for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    probs = model()
    
    # 1. Fit Loss
    loss_fit = F.binary_cross_entropy(probs[train_mask_t], y_empirical[train_mask_t])
    
    # 2. Annealed Sparsity Penalty
    if epoch < TAU_WARMUP:
        current_lambda = 0.0
    elif epoch < TAU_WARMUP + RAMP_EPOCHS:
        # Linearly ramp from 0.0 to LAMBDA_TAU
        progress = (epoch - TAU_WARMUP) / RAMP_EPOCHS
        current_lambda = LAMBDA_TAU * progress
    else:
        current_lambda = LAMBDA_TAU
        
    tau = model.get_tau()
    loss_sparsity = current_lambda * torch.sum(tau)
    
    (loss_fit + loss_sparsity).backward()
    optimizer.step()
    
    # 3. Safe Snapping
    # Only snap to dead zone if we are deep into the penalty phase
    if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
        with torch.no_grad():
            active_mask = model.get_tau() > SNAPPING_THRESHOLD
            for k in range(K_MODEL):
                if not active_mask[k]:
                    model.tau_raw[k] = DEAD_ZONE_VALUE

    # --- EVALUATION ---
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
            # Strict counting for display
            active_dims = (tau_vals > SNAPPING_THRESHOLD).sum()
            
            phase = "WARM" if epoch < TAU_WARMUP else ("RAMP" if epoch < TAU_WARMUP + RAMP_EPOCHS else "FULL")
            
            print(f"Epoch {epoch:4d} [{phase}] | Train: {train_rmse:.4f} Test: {test_rmse:.4f} | "
                  f"Active: {active_dims}/{K_MODEL} | Lambda: {current_lambda:.2e}")
            
            # Save Best Logic
            if test_rmse < best_test_rmse - MIN_DELTA:
                best_test_rmse = test_rmse
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch > TAU_WARMUP:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(">> Early stopping.")
                    break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Restored best model (Test RMSE: {best_test_rmse:.4f})")
model.eval()
with torch.no_grad():
    final_p = model()
    final_test = compute_rmse(final_p.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

print(f"\nFINAL TEST RMSE: {final_test:.4f}")
print(f"Improvement over Baseline: {((test_rmse_mean - final_test)/test_rmse_mean)*100:.2f}%")

## 6. Analysis & Visualization
print("\n" + "="*60 + "\nREGRESSION ANALYSIS & PLOTS\n" + "="*60)

# 1. Collect Data
model.eval()
with torch.no_grad():
    y_pred = model().cpu().numpy()
    y_true = y_empirical.cpu().numpy()

# Flatten and mask for Train/Test sets
flat_pred_train = y_pred[train_mask]
flat_true_train = y_true[train_mask]
flat_pred_test = y_pred[test_mask]
flat_true_test = y_true[test_mask]

# Calculate Residuals
res_train = flat_true_train - flat_pred_train
res_test = flat_true_test - flat_pred_test

# Metrics
mae_test = mean_absolute_error(flat_true_test, flat_pred_test)
r2_test = r2_score(flat_true_test, flat_pred_test)

print(f"METRICS (Test Set):")
print(f"  RMSE : {final_test:.4f}")
print(f"  MAE  : {mae_test:.4f}")
print(f"  R^2  : {r2_test:.4f}")

# --- PLOTTING ---
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 2)

# Plot A: Learning Curves
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(history['epoch'], history['train'], label='Train RMSE', linewidth=2)
ax1.plot(history['epoch'], history['test'], label='Test RMSE', linewidth=2, linestyle='--')
ax1.axhline(y=test_rmse_mean, color='r', linestyle=':', label='Baseline Mean')
ax1.axhline(y=test_rmse_rasch, color='g', linestyle=':', label='Baseline Rasch')
ax1.set_title(f'Learning Trajectory (Best Test RMSE: {final_test:.4f})', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('RMSE')
ax1.legend()

# Plot B: Parity Plot (Train)
ax2 = fig.add_subplot(gs[1, 0])
sns.scatterplot(x=flat_true_train, y=flat_pred_train, alpha=0.1, ax=ax2, color='blue', s=10)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=1.5)
ax2.set_title('Train: Predicted vs Empirical', fontsize=12)
ax2.set_xlabel('Empirical Probability')
ax2.set_ylabel('Predicted Probability')

# Plot C: Parity Plot (Test)
ax3 = fig.add_subplot(gs[1, 1])
sns.scatterplot(x=flat_true_test, y=flat_pred_test, alpha=0.5, ax=ax3, color='orange', s=15)
ax3.plot([0, 1], [0, 1], 'r--', linewidth=1.5)
ax3.set_title(f'Test: Predicted vs Empirical ($R^2={r2_test:.2f}$)', fontsize=12)
ax3.set_xlabel('Empirical Probability')
ax3.set_ylabel('Predicted Probability')

# Plot D: Residual Histogram
ax4 = fig.add_subplot(gs[2, 0])
sns.histplot(res_train, color='blue', alpha=0.3, label='Train', kde=True, ax=ax4, stat='density')
sns.histplot(res_test, color='orange', alpha=0.3, label='Test', kde=True, ax=ax4, stat='density')
ax4.axvline(0, color='black', linestyle='--')
ax4.set_title('Residual Distribution (Error)', fontsize=12)
ax4.set_xlabel('Residual ($y - \hat{y}$)')
ax4.legend()

# Plot E: Latent Factors (Tau)
ax5 = fig.add_subplot(gs[2, 1])
with torch.no_grad():
    taus = model.get_tau().cpu().numpy()
sns.barplot(x=np.arange(len(taus)), y=taus, ax=ax5, palette='viridis')
ax5.set_title('Learned Latent Factor Strength (Tau)', fontsize=12)
ax5.set_xlabel('Latent Dimension')
ax5.set_ylabel('Active Scale')

plt.tight_layout()
plt.savefig('beta_irt_regression_analysis.png', dpi=150)
print("\nPlot saved to 'beta_irt_regression_analysis.png'")
plt.show()

## 6. Analysis & Visualization (Enhanced)
print("\n" + "="*60 + "\nREGRESSION ANALYSIS & PLOTS\n" + "="*60)

from sklearn.calibration import calibration_curve

# 1. Collect Data
model.eval()
with torch.no_grad():
    y_pred = model().cpu().numpy()
    y_true = y_empirical.cpu().numpy()

# Masking
flat_pred_train = y_pred[train_mask]
flat_true_train = y_true[train_mask]
flat_pred_test = y_pred[test_mask]
flat_true_test = y_true[test_mask]

# Residuals
res_test = flat_true_test - flat_pred_test

# Metrics
mae_test = mean_absolute_error(flat_true_test, flat_pred_test)
r2_test = r2_score(flat_true_test, flat_pred_test)
pearson_corr, _ = pearsonr(flat_true_test, flat_pred_test)

print(f"METRICS (Test Set):")
print(f"  RMSE    : {final_test:.4f}")
print(f"  MAE     : {mae_test:.4f}")
print(f"  R^2     : {r2_test:.4f}")
print(f"  Pearson : {pearson_corr:.4f}")

# --- PLOTTING CONFIG ---
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3) # 2 rows, 3 columns

# 1. Learning Trajectory (Top Left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['epoch'], history['train'], label='Train RMSE', color='tab:blue', alpha=0.7)
ax1.plot(history['epoch'], history['test'], label='Test RMSE', color='tab:orange', linewidth=2)
# Mark Best Epoch
best_epoch_idx = np.argmin(history['test'])
best_val = history['test'][best_epoch_idx]
best_ep = history['epoch'][best_epoch_idx]
ax1.scatter(best_ep, best_val, c='red', s=50, zorder=5, label=f'Best ({best_val:.4f})')
ax1.axvline(best_ep, color='red', linestyle=':', alpha=0.5)
ax1.axhline(test_rmse_mean, color='gray', linestyle='--', label='Baseline')
ax1.set_title(f'Learning Curve (Best Epoch: {best_ep})', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('RMSE')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Test Parity + Density (Top Middle)
# Scatter points can hide density. We add a KDE contour to show where the mass is.
ax2 = fig.add_subplot(gs[0, 1])
# Scatter background
ax2.scatter(flat_true_test, flat_pred_test, alpha=0.2, color='gray', s=10, label='Data Points')
# Density contours
try:
    sns.kdeplot(x=flat_true_test, y=flat_pred_test, levels=5, color='tab:blue', linewidths=1.5, ax=ax2)
except:
    pass # Skip if data is too sparse for KDE
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Fit')
ax2.set_title(f'Test Parity ($R^2={r2_test:.2f}$)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Empirical Probability (Target)')
ax2.set_ylabel('Predicted Probability')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.legend(loc='upper left', fontsize=9)

# 3. Calibration Curve (Top Right)
# Does a prediction of 0.6 actually correspond to 60% empirical success?
ax3 = fig.add_subplot(gs[0, 2])
# We bin predictions into 10 buckets
prob_true, prob_pred = calibration_curve(flat_true_test > 0.5, flat_pred_test, n_bins=10, strategy='uniform')
ax3.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
ax3.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
ax3.set_title('Calibration (Reliability Diagram)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Mean Predicted Probability')
ax3.set_ylabel('Fraction of Positives (Proxy)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Residual Distribution (Bottom Left)
ax4 = fig.add_subplot(gs[1, 0])
sns.histplot(res_test, kde=True, color='tab:orange', bins=30, ax=ax4, stat='density')
ax4.axvline(0, color='black', linestyle='--')
ax4.set_title(f'Residuals ($\mu={np.mean(res_test):.3f}, \sigma={np.std(res_test):.3f}$)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Error ($y_{true} - y_{pred}$)')

# 5. Residuals vs Predicted (Bottom Middle)
# Checks for Heteroscedasticity (Does model fail at extremes?)
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(flat_pred_test, res_test, alpha=0.2, color='purple', s=10)
ax5.axhline(0, color='black', linestyle='--')
ax5.set_title('Bias Check: Residuals vs Predictions', fontsize=11, fontweight='bold')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Residual')
# Add a rolling mean line to see systematic bias
try:
    sns.regplot(x=flat_pred_test, y=res_test, scatter=False, lowess=True, color='red', line_kws={'alpha':0.8}, ax=ax5)
except:
    pass

# 6. Latent Factors (Bottom Right)
ax6 = fig.add_subplot(gs[1, 2])
with torch.no_grad():
    taus = model.get_tau().cpu().numpy()
colors = ['tab:blue' if t > TAU_THRESHOLD else 'lightgray' for t in taus]
ax6.bar(np.arange(len(taus)), taus, color=colors)
ax6.axhline(TAU_THRESHOLD, color='red', linestyle=':', label='Threshold')
ax6.set_title(f'Latent Factor Strength ({np.sum(taus > TAU_THRESHOLD)} Active)', fontsize=11, fontweight='bold')
ax6.set_xlabel('Latent Dimension')
ax6.legend()

plt.tight_layout()
plt.savefig('beta_irt_final_analysis.png', dpi=150)
print("\nVisualization saved to 'beta_irt_final_analysis.png'")
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
        'model': 'pca_irt',
        'train_rmse': float(train_rmse_final),
        'test_rmse': float(final_test),
        'train_corr': float(train_corr),
        'test_corr': float(test_corr),
        'mae_test': float(mae_test),
        'r2_test': float(r2_test),
        'pearson_test': float(pearson_corr),
    }