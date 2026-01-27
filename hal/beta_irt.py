# ==========================================================================
# CONFIGURATION
# ==========================================================================
USE_EMPIRICAL_BASELINE = True  # True: use full empirical, False: use 1 random binary
TEST_SIZE = 0.1  # 10% holdout
RANDOM_SEED = 42
USE_SAE = False
K_MODEL = 12
TAU_THRESHOLD = 0.03  # Automatic sparsity threshold on tau
TAU_TEMPERATURE = 0.015  # Soft gating sharpness (lower = harder)
EPOCHS = 800
EVAL_EVERY = 50
PATIENCE = 8
MIN_DELTA = 1e-4
SPARSITY_TOL = 1.5e-3  # Allow slight RMSE tradeoff for fewer active dims
MIN_ACTIVE_DIMS = 1
LR_THETA = 0.01
LR_GLOBAL = 0.003
WD_THETA = 1e-4
WD_W = 1e-3
LAMBDA_TAU = 1e-2
TAU_WARMUP = 50
TAU_INIT = -2.0
NORMALIZE_THETA = True

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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

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

print("="*60)

# Utility functions
def compute_rmse(predictions, targets, mask):
    """Compute RMSE on masked data"""
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

def compute_correlation(predictions, targets, mask):
    """Compute Pearson correlation on masked data"""
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    valid_preds, valid_targets = predictions[valid], targets[valid]
    return pearsonr(valid_preds, valid_targets)[0] if len(valid_preds) > 1 else 0.0

def beta_nll_loss(mu, y, mask, phi, eps=1e-4):
    """Negative log-likelihood for Beta regression with fixed concentration phi."""
    y = torch.clamp(y, eps, 1.0 - eps)
    mu = torch.clamp(mu, eps, 1.0 - eps)
    alpha = mu * phi
    beta = (1.0 - mu) * phi
    log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    loglik = (alpha - 1.0) * torch.log(y) + (beta - 1.0) * torch.log(1.0 - y) - log_beta
    return (-loglik)[mask].mean()

## 2. Load Response Matrices
resmat_dir = 'resmats'
files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
n_samples = len(files)

all_dfs = []
for f in files:
    df = pd.read_csv(os.path.join(resmat_dir, f), index_col=0)
    all_dfs.append(df)

# Find shared models across all matrices
shared_indices = set(all_dfs[0].index)
for df in all_dfs[1:]:
    shared_indices = shared_indices.intersection(set(df.index))
shared_indices = sorted(list(shared_indices))

# Filter to shared rows
filtered_dfs = [df.loc[shared_indices] for df in all_dfs]

# Calculate empirical probability matrix (element-wise mean across 9 samples)
prob_matrix = np.nanmean([df.values for df in filtered_dfs], axis=0)
prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

print(f"Loaded {n_samples} response matrices")
print(f"Shared models: {len(shared_indices)}, Tasks: {prob_df.shape[1]}")
print(f"Empirical prob range: [{np.nanmin(prob_df.values):.3f}, {np.nanmax(prob_df.values):.3f}]")

# Quick data diagnostics
resmat_stats = []
for i, df in enumerate(filtered_dfs):
    vals = df.values
    resmat_stats.append({
        'idx': i,
        'mean': np.nanmean(vals),
        'std': np.nanstd(vals),
        'nan_rate': np.isnan(vals).mean()
    })
stats_df = pd.DataFrame(resmat_stats)
print("Resmat summary (mean/std/nan_rate):")
print(stats_df[['mean', 'std', 'nan_rate']].describe().round(4))

## 3. Prepare Embeddings and Train/Test Split
# Load embeddings
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
    matched = sum(1 for e in raw_embs if not np.allclose(e, 0))
    print(f"Embeddings: {matched}/{len(task_ids)} matched ({matched/len(task_ids):.1%})")
else:
    x_j_dense = torch.randn(len(prob_df.columns), 4096, dtype=torch.float32)
    x_j_dense = F.normalize(x_j_dense, p=2, dim=1).to(device)
    print("Using random embeddings")

# Create train/test split (10% test, seed 42)
N, J = prob_df.shape
np.random.seed(RANDOM_SEED)
J_indices = np.arange(J)
np.random.shuffle(J_indices)

n_test = int(TEST_SIZE * J)
test_idx = J_indices[:n_test]
train_idx = J_indices[n_test:]

# Convert to tensors
y_empirical = torch.from_numpy(prob_df.values.astype(np.float32)).to(device)
x_j = x_j_dense

# Create masks
train_mask = np.zeros_like(prob_df.values, dtype=bool)
train_mask[:, train_idx] = ~np.isnan(prob_df.values)[:, train_idx]
test_mask = np.zeros_like(prob_df.values, dtype=bool)
test_mask[:, test_idx] = ~np.isnan(prob_df.values)[:, test_idx]

train_mask_t = torch.from_numpy(train_mask).to(device)
test_mask_t = torch.from_numpy(test_mask).to(device)

print(f"Split: {len(train_idx)} train items, {len(test_idx)} test items")
print(f"Entries: {train_mask.sum()} train, {test_mask.sum()} test")

## 4. Baseline Models
print("\n" + "="*60)
print("BASELINES")
print("="*60)

if USE_EMPIRICAL_BASELINE:
    # Use full empirical probabilities (9 samples)
    y_baseline = y_empirical
    baseline_type = f"Empirical ({n_samples} samples)"
else:
    # Use single random binary matrix
    random_idx = np.random.randint(0, len(filtered_dfs))
    single_df = filtered_dfs[random_idx]
    y_baseline = torch.from_numpy(single_df.values.astype(np.float32)).to(device)
    baseline_type = f"Single Binary (matrix {random_idx+1}/{n_samples})"

print(f"Baseline data: {baseline_type}")

# Baseline 1: Global Mean
mean_val = torch.nanmean(y_baseline[train_mask_t])
pred_mean = mean_val.expand_as(y_baseline)

train_rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
test_rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

print(f"\n1. Global Mean: {mean_val:.4f}")
print(f"   Train RMSE: {train_rmse_mean:.4f} | Test RMSE: {test_rmse_mean:.4f}")

baseline_results = {
    'mean': {'train_rmse': train_rmse_mean, 'test_rmse': test_rmse_mean}
}

# Baseline 2: Rasch-IRT
theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
optimizer = torch.optim.Adam([theta, beta], lr=0.01, weight_decay=1e-5)

print(f"\n2. Rasch-IRT (training 500 epochs...)")
for epoch in range(500):
    optimizer.zero_grad()
    logits = theta.unsqueeze(1) - beta.unsqueeze(0)
    loss = F.binary_cross_entropy_with_logits(logits, y_baseline, reduction='none')
    loss = (loss * train_mask_t).sum() / train_mask_t.sum()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logits = theta.unsqueeze(1) - beta.unsqueeze(0)
    probs_rasch = torch.sigmoid(logits)
    train_rmse_rasch = compute_rmse(probs_rasch.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
    test_rmse_rasch = compute_rmse(probs_rasch.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

print(f"   Train RMSE: {train_rmse_rasch:.4f} | Test RMSE: {test_rmse_rasch:.4f}")

baseline_results['rasch'] = {'train_rmse': train_rmse_rasch, 'test_rmse': test_rmse_rasch}

## 5. Beta-IRT Model
print("\n" + "="*60)
print("BETA-IRT MODEL")
print("="*60)

class BetaIRTLinear(nn.Module):
    """Linear Beta-IRT with learned tau sparsity (automatic dimensionality)."""
    def __init__(self, N, J, K_model, d_features, x_j_emb, tau_threshold=None, tau_temperature=0.02):
        super().__init__()
        self.N, self.J, self.K = N, J, K_model
        self.tau_threshold = tau_threshold
        self.tau_temperature = tau_temperature
        self.register_buffer('x_j', x_j_emb)

        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.empty(K_model, d_features))
        nn.init.xavier_uniform_(self.W)
        self.tau_raw = nn.Parameter(torch.ones(K_model) * TAU_INIT)
        self.difficulty_proj = nn.Linear(d_features, 1)
        nn.init.xavier_uniform_(self.difficulty_proj.weight)

    def _gated_tau(self):
        tau = F.softplus(self.tau_raw)
        if self.tau_threshold is None:
            return tau
        gate = torch.sigmoid((tau - self.tau_threshold) / self.tau_temperature)
        return tau * gate

    def forward(self, return_logits=False):
        pred_delta = self.difficulty_proj(self.x_j).squeeze().unsqueeze(0)
        tau = self._gated_tau()
        W_norm = F.normalize(self.W, dim=1)
        a_j = (self.x_j @ W_norm.T) * tau.unsqueeze(0)
        logits_y = (
            self.theta @ a_j.T
            + pred_delta
            + self.theta_bias.unsqueeze(1)
            + self.global_bias
        )
        if return_logits:
            return logits_y
        return torch.sigmoid(logits_y)

d_features = x_j.shape[1]
model = BetaIRTLinear(
    N, J, K_MODEL, d_features, x_j,
    tau_threshold=TAU_THRESHOLD,
    tau_temperature=TAU_TEMPERATURE
).to(device)

print(f"K={K_MODEL} latent factors, d={d_features} features")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

if USE_SAE:
    print("\n" + "="*60)
    print("SPARSE AUTOENCODER")
    print("="*60)
    from hypothesaes.quickstart import train_sae

    embeddings_np = x_j_dense.cpu().numpy()
    print(f"Training SAE on {embeddings_np.shape[0]} embeddings of dimension {embeddings_np.shape[1]}...")

    M_features = 1024
    K_active = 64
    sae = train_sae(
        embeddings=embeddings_np,
        M=M_features,
        K=K_active,
        batch_size=512,
        n_epochs=50,
        learning_rate=5e-4,
        checkpoint_dir='checkpoints/irt_sae'
    )

    embeddings_torch = torch.tensor(embeddings_np, dtype=torch.float32)
    with torch.no_grad():
        sae_output = sae(embeddings_torch)
        if isinstance(sae_output, tuple):
            reconstruction, details = sae_output
            if isinstance(details, dict):
                if 'latent' in details:
                    x_j_sparse = details['latent']
                elif 'activations' in details:
                    x_j_sparse = details['activations']
                else:
                    x_j_sparse = reconstruction
            else:
                x_j_sparse = reconstruction
        else:
            x_j_sparse = sae_output

    x_j_sparse = x_j_sparse.to(device)
    x_j_sparse = F.normalize(x_j_sparse, p=2, dim=1)
    x_j = x_j_sparse
    d_features = x_j.shape[1]
    print(f"Using {d_features}D SAE sparse features for Beta-IRT")
    model = BetaIRTLinear(
        N, J, K_MODEL, d_features, x_j,
        tau_threshold=TAU_THRESHOLD,
        tau_temperature=TAU_TEMPERATURE
    ).to(device)

# Fixed concentration for Beta likelihood
phi = max(int(n_samples) - 1, 1)
phi = torch.tensor(float(phi), device=device)

optimizer = optim.AdamW([
    {'params': model.theta, 'lr': LR_THETA, 'weight_decay': WD_THETA},
    {'params': model.theta_bias, 'lr': LR_THETA, 'weight_decay': WD_THETA},
    {'params': model.global_bias, 'lr': LR_THETA, 'weight_decay': 0.0},
    {'params': model.W, 'lr': LR_GLOBAL, 'weight_decay': WD_W},
    {'params': model.tau_raw, 'lr': LR_GLOBAL, 'weight_decay': 0.0},
    {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL, 'weight_decay': WD_W}
])

train_losses = []
test_losses = []
best_test_rmse = float('inf')
best_active_dims = K_MODEL
epochs_without_improvement = 0
best_model_state = None

for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(return_logits=True)
    loss_fit = F.binary_cross_entropy_with_logits(logits, y_empirical, reduction='none')
    loss_fit = loss_fit[train_mask_t].mean()
    tau = model._gated_tau()
    tau_weight = LAMBDA_TAU * min(1.0, epoch / max(TAU_WARMUP, 1))
    loss_tau = tau_weight * torch.sum(tau)
    loss = loss_fit + loss_tau
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    if NORMALIZE_THETA:
        with torch.no_grad():
            theta_mean = model.theta.mean(dim=0, keepdim=True)
            theta_std = model.theta.std(dim=0, keepdim=True).clamp_min(1e-4)
            model.theta.data = (model.theta.data - theta_mean) / theta_std

    probs = torch.sigmoid(logits)
    train_rmse = torch.sqrt(((probs - y_empirical) ** 2)[train_mask_t].mean()).item()
    train_losses.append(train_rmse)

    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            p_test = model()
            test_rmse = torch.sqrt(((p_test - y_empirical) ** 2)[test_mask_t].mean()).item()
            test_losses.append(test_rmse)
            active_dims = (model._gated_tau() > TAU_THRESHOLD).sum().item()
            pred_std = p_test[test_mask_t].std().item()
            print(f"Epoch {epoch:4d} | Train: {train_rmse:.4f} | Test: {test_rmse:.4f} | Active: {active_dims}/{K_MODEL} | PredSTD: {pred_std:.4f}")

            if active_dims >= MIN_ACTIVE_DIMS and (
                (test_rmse < best_test_rmse - MIN_DELTA) or (
                    test_rmse <= best_test_rmse + SPARSITY_TOL and active_dims < best_active_dims
                )
            ):
                best_test_rmse = test_rmse
                best_active_dims = active_dims
                epochs_without_improvement = 0
                best_model_state = {
                    'theta': model.theta.data.clone(),
                    'W': model.W.data.clone(),
                    'tau_raw': model.tau_raw.data.clone(),
                    'difficulty_proj_weight': model.difficulty_proj.weight.data.clone(),
                    'difficulty_proj_bias': model.difficulty_proj.bias.data.clone()
                }
                print(f"         → New best model (RMSE: {best_test_rmse:.4f}, Active: {best_active_dims}/{K_MODEL})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                    print(f"   No improvement for {PATIENCE * EVAL_EVERY} epochs (patience={PATIENCE})")
                    print(f"   Best test RMSE: {best_test_rmse:.4f}")
                    break

# Restore best model
if best_model_state is not None:
    print(f"\n✓ Restoring best model from early stopping")
    model.theta.data.copy_(best_model_state['theta'])
    model.W.data.copy_(best_model_state['W'])
    model.tau_raw.data.copy_(best_model_state['tau_raw'])
    model.difficulty_proj.weight.data.copy_(best_model_state['difficulty_proj_weight'])
    model.difficulty_proj.bias.data.copy_(best_model_state['difficulty_proj_bias'])

print(f"\nTraining complete | Best test RMSE: {best_test_rmse:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    probs_final = model()
    train_rmse_betairt = compute_rmse(probs_final.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
    test_rmse_betairt = compute_rmse(probs_final.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
    active_dims = (model._gated_tau() > TAU_THRESHOLD).sum().item()

print(f"Active dimensions: {active_dims}/{K_MODEL}")
print(f"Train RMSE: {train_rmse_betairt:.4f} | Test RMSE: {test_rmse_betairt:.4f}")

## 6. Final Comparison
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"\n{baseline_type} Baselines:")
print(f"  Global Mean     Train: {baseline_results['mean']['train_rmse']:.4f} | Test: {baseline_results['mean']['test_rmse']:.4f}")
print(f"  Rasch-IRT       Train: {baseline_results['rasch']['train_rmse']:.4f} | Test: {baseline_results['rasch']['test_rmse']:.4f}")

print(f"\nBeta-IRT (Empirical {n_samples} samples, linear):")
print(f"  Beta-IRT        Train: {train_rmse_betairt:.4f} | Test: {test_rmse_betairt:.4f}")
print(f"  Active dims: {active_dims}/{K_MODEL}")
print(f"  Features: {d_features}D | Active dims (tau>{TAU_THRESHOLD}): {active_dims}")

# Improvement calculation
if USE_EMPIRICAL_BASELINE:
    baseline_best = min(baseline_results['mean']['test_rmse'], baseline_results['rasch']['test_rmse'])
    improvement = ((baseline_best - test_rmse_betairt) / baseline_best) * 100
    print(f"\nImprovement over best baseline: {improvement:+.2f}%")
else:
    # Compare cross-dataset: baseline on single binary, Beta-IRT on empirical
    print(f"\nNote: Baselines use {baseline_type}, Beta-IRT uses empirical ({n_samples} samples)")

## 7. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Training curves
test_epochs = [i * EVAL_EVERY for i in range(len(test_losses))]
axes[0, 0].plot(train_losses, label='Train', alpha=0.7, linewidth=0.8)
axes[0, 0].plot(test_epochs, test_losses, label='Test', marker='o', markersize=2, linewidth=1.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].set_title('Beta-IRT Training Curves (with SAE + Early Stopping)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Model comparison
models = ['Global Mean', 'Rasch-IRT', 'Beta-IRT\n(SAE)']
train_rmses = [baseline_results['mean']['train_rmse'], baseline_results['rasch']['train_rmse'], train_rmse_betairt]
test_rmses = [baseline_results['mean']['test_rmse'], baseline_results['rasch']['test_rmse'], test_rmse_betairt]

x = np.arange(len(models))
width = 0.35
axes[0, 1].bar(x - width/2, train_rmses, width, label='Train', alpha=0.8)
axes[0, 1].bar(x + width/2, test_rmses, width, label='Test', alpha=0.8)
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('Model Comparison')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim(0.1, max(max(train_rmses), max(test_rmses)) * 1.1)

# 3. Predicted vs Actual (Test set)
probs_np = probs_final.cpu().numpy()
targets_np = y_empirical.cpu().numpy()
test_preds = probs_np[test_mask]
test_targets = targets_np[test_mask]

if len(test_preds) > 5000:
    idx = np.random.choice(len(test_preds), 5000, replace=False)
    test_preds, test_targets = test_preds[idx], test_targets[idx]

axes[1, 0].scatter(test_targets, test_preds, alpha=0.3, s=8)
axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect', linewidth=1.5)
axes[1, 0].set_xlabel('Actual Probability')
axes[1, 0].set_ylabel('Predicted Probability')
axes[1, 0].set_title(f'Test Set Predictions (RMSE={test_rmse_betairt:.4f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(0, 1)

# 4. Difficulty distribution
with torch.no_grad():
    difficulties = model.difficulty_proj(x_j).squeeze().cpu().numpy()

axes[1, 1].hist(difficulties, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 1].set_xlabel('Difficulty (β)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Task Difficulty Distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = 'beta_irt_results_sae.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved: {output_file}")
plt.show()

## 8. Save Results
results_dict = {
    'config': {
        'n_samples': int(n_samples),
        'n_models': int(N),
        'n_tasks': int(J),
        'train_test_split': f'{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}',
        'random_seed': int(RANDOM_SEED),
        'use_empirical_baseline': USE_EMPIRICAL_BASELINE,
        'baseline_type': baseline_type,
        'embedding_features': int(d_features),
        'sae_features': 1024 if USE_SAE else 0,
        'sae_sparsity': 64 if USE_SAE else 0,
        'latent_factors': int(K_MODEL),
        'active_factors': int(active_dims),
        'early_stopping_patience': PATIENCE,
        'total_epochs': len(train_losses)
    },
    'baselines': {
        'mean': {'train_rmse': float(baseline_results['mean']['train_rmse']), 
                 'test_rmse': float(baseline_results['mean']['test_rmse'])},
        'rasch': {'train_rmse': float(baseline_results['rasch']['train_rmse']), 
                  'test_rmse': float(baseline_results['rasch']['test_rmse'])}
    },
    'beta_irt': {
        'train_rmse': float(train_rmse_betairt),
        'test_rmse': float(test_rmse_betairt),
        'best_test_rmse': float(best_test_rmse)
    }
}

output_json = 'beta_irt_results_sae.json'
with open(output_json, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"Results saved: {output_json}")
print("\n" + "="*60)
print("COMPLETE")
print("="*60)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Data & Calculate Empirical P
# ==========================================
resmat_dir = 'resmats'

if os.path.exists(resmat_dir):
    files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
    print(f"Found {len(files)} response matrices in '{resmat_dir}'")
    
    # Load all CSVs
    all_dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(resmat_dir, f), index_col=0)
        all_dfs.append(df)
    
    # Find shared indices (Agents) to ensure alignment
    shared_indices = set(all_dfs[0].index)
    for df in all_dfs[1:]:
        shared_indices = shared_indices.intersection(set(df.index))
    shared_indices = sorted(list(shared_indices))
    
    # Filter and stack data
    filtered_vals = [df.loc[shared_indices].values for df in all_dfs]
    stacked_matrix = np.array(filtered_vals)  # Shape: (n_samples, n_agents, n_tasks)
    
    # Calculate Empirical Matrix P (Mean across samples)
    p_matrix = np.nanmean(stacked_matrix, axis=0)
    print(f"Empirical Matrix P shape: {p_matrix.shape}")
    
    # ==========================================
    # 2. Plot Histogram
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Flatten matrix to 1D array and drop NaNs
    p_values = p_matrix.flatten()
    p_values = p_values[~np.isnan(p_values)]
    
    # Plot histogram
    plt.hist(p_values, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
    
    # Add statistics lines
    mean_val = np.mean(p_values)
    median_val = np.median(p_values)
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.3f}')
    
    # Formatting
    plt.title('Distribution of Empirical Probabilities (Matrix P)', fontsize=14, pad=15)
    plt.xlabel('Probability of Success', fontsize=12)
    plt.ylabel('Count of Agent-Task Pairs', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

else:
    print(f"Error: Directory '{resmat_dir}' not found. Please run this where your data resides.")