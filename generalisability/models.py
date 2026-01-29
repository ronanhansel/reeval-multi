"""
Latent Variable Models for IRT

Contains two main model families:
1. Bernoulli IRT - For binary response data (correct/incorrect)
2. Beta IRT - For continuous response data (e.g., averaged success rates)

Both support:
- Amortized item parameters via embeddings
- Automatic Relevance Determination (ARD) for dimension discovery
- Multiple baseline comparisons (Global Mean, Rasch)

Usage:
    python models.py --model bernoulli --embedding-type pca
    python models.py --model beta --embedding-type sae
    python models.py --benchmark helm --model bernoulli
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_error
from huggingface_hub import snapshot_download

from embeddings import get_embeddings, align_embeddings_to_tasks

warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data-reeval-multi')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')

# HuggingFace dataset
HF_REPO_ID = "ronanhansel/data-reeval-multi"


def ensure_data_downloaded():
    """Download data from HuggingFace if not present locally."""
    if not os.path.exists(DATA_DIR):
        print(f"Data not found locally. Downloading from HuggingFace ({HF_REPO_ID})...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print("Download complete.")

# Data split settings
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Model architecture
K_MODEL = 30

# Tau sparsity settings (tuned for ColBench/Beta IRT)
LAMBDA_TAU = 0.0005  # Lower regularization to preserve dimensions
TAU_INIT = 0.5
TAU_WARMUP = 300     # Longer warmup before applying sparsity
RAMP_EPOCHS = 300    # Longer ramp to final sparsity
SNAPPING_THRESHOLD = 0.001  # Less aggressive snapping
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# Training settings
EPOCHS = 1001  # Longer training for better convergence
EVAL_EVERY = 100

# Learning rates
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 1e-3
WD_W = 1e-5

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# ==============================================================================
# Model Definitions
# ==============================================================================

class BernoulliIRT(nn.Module):
    """
    Amortized IRT model for binary responses.

    P(y_ij = 1) = sigmoid(theta_i @ a_j + beta_j + bias_i)

    where:
        - theta_i: latent ability vector for user i
        - a_j = tau * (W @ x_j): amortized item loadings from embeddings
        - beta_j: amortized item difficulty
        - tau: ARD scales for automatic dimension discovery
    """

    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        self.K = K

        # User parameters
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))

        # Global parameters
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)

        # Item difficulty projection
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

    def get_active_dims(self):
        return (self.get_tau() > TAU_THRESHOLD).sum().item()


class BetaIRT(nn.Module):
    """
    Amortized IRT model for continuous [0, 1] responses using Beta distribution.

    E[y_ij] = sigmoid(theta_i @ a_j + beta_j + bias_i)

    Same structure as BernoulliIRT but trained with MSE loss for continuous targets.
    Suitable for aggregated response matrices (averaged across runs).
    """

    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        self.K = K

        # User parameters
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))

        # Global parameters
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)

        # Item difficulty projection
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

    def get_active_dims(self):
        return (self.get_tau() > TAU_THRESHOLD).sum().item()


# ==============================================================================
# Baseline Models
# ==============================================================================

def train_rasch_baseline(N, J, y_train, train_mask, n_iter=100):
    """Train a basic Rasch IRT model as baseline."""
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)

    optimizer = torch.optim.LBFGS(
        [theta, beta], lr=0.1, max_iter=20,
        history_size=10, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        probs = torch.sigmoid(theta.unsqueeze(1) - beta.unsqueeze(0))
        loss = F.binary_cross_entropy(probs, y_train, reduction='none')
        total_loss = (loss * train_mask).sum() / train_mask.sum()
        total_loss.backward()
        return total_loss

    for _ in range(n_iter):
        optimizer.step(closure)

    with torch.no_grad():
        probs = torch.sigmoid(theta.unsqueeze(1) - beta.unsqueeze(0))

    return probs


def compute_global_mean_baseline(y_train, train_mask, shape):
    """Compute global mean baseline prediction."""
    mean_val = y_train[train_mask].mean()
    return mean_val.expand(shape)


# ==============================================================================
# Training Functions
# ==============================================================================

def train_model(model, y_train, train_mask, y_eval=None, eval_mask=None,
                epochs=EPOCHS, model_type='bernoulli'):
    """
    Train IRT model with ARD sparsity regularization.

    Args:
        model: BernoulliIRT or BetaIRT model
        y_train: training targets
        train_mask: boolean mask for training entries
        y_eval: evaluation targets (optional)
        eval_mask: evaluation mask (optional)
        epochs: number of training epochs
        model_type: 'bernoulli' or 'beta'

    Returns:
        best_metric: best evaluation metric (RMSE for beta, BCE for bernoulli)
        history: training history dict
    """
    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
        {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
        {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
    ])

    history = {'loss': [], 'eval_metric': [], 'active_dims': []}
    best_metric = float('inf')

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        probs = model()

        # Loss computation
        if model_type == 'bernoulli':
            loss_fit = F.binary_cross_entropy(probs[train_mask], y_train[train_mask])
        else:  # beta
            loss_fit = F.mse_loss(probs[train_mask], y_train[train_mask])

        # ARD sparsity schedule
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

        # Tau snapping
        if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
            with torch.no_grad():
                active_mask = model.get_tau() > SNAPPING_THRESHOLD
                for k in range(model.K):
                    if not active_mask[k]:
                        model.tau_raw[k] = DEAD_ZONE_VALUE

        # Evaluation
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p_eval = model()

                if y_eval is not None and eval_mask is not None:
                    if model_type == 'bernoulli':
                        metric = F.binary_cross_entropy(p_eval[eval_mask], y_eval[eval_mask]).item()
                    else:
                        metric = np.sqrt(F.mse_loss(p_eval[eval_mask], y_eval[eval_mask]).item())
                    best_metric = min(best_metric, metric)
                else:
                    metric = loss_fit.item()

                active_dims = model.get_active_dims()

                history['loss'].append(loss_fit.item())
                history['eval_metric'].append(metric)
                history['active_dims'].append(active_dims)

                if epoch % 500 == 0:
                    print(f"Ep {epoch} | Loss: {loss_fit.item():.4f} | Metric: {metric:.4f} | Active: {active_dims}")

    return best_metric, history


# ==============================================================================
# Data Loading
# ==============================================================================

def load_helm_data():
    """Load HELM benchmark data."""
    # Ensure data is downloaded
    ensure_data_downloaded()

    resmat_file = os.path.join(DATA_DIR, 'resmat.pkl')
    print(f"Loading HELM response matrix from {resmat_file}...")

    y_df = pd.read_pickle(resmat_file)

    # Filter valid rows/cols
    y_df = y_df[y_df.notna().any(axis=1)]
    valid_cols = [c for c in y_df.columns if y_df[c].notna().any() and (y_df[c].dropna() != 0).any()]
    y_df = y_df[valid_cols]

    # Get question text for alignment
    try:
        if isinstance(y_df.columns, pd.MultiIndex) and 'input.text' in y_df.columns.names:
            task_ids = y_df.columns.get_level_values('input.text').tolist()
        else:
            task_ids = [str(c) for c in y_df.columns]
    except:
        task_ids = [str(c) for c in y_df.columns]

    print(f"HELM data shape: {y_df.shape}")
    return y_df, task_ids


def load_colbench_data():
    """Load ColBench benchmark data (aggregated across response matrices)."""
    # Ensure data is downloaded
    ensure_data_downloaded()

    resmat_dir = os.path.join(DATA_DIR, 'colbench')
    all_files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])

    print(f"Loading {len(all_files)} ColBench response matrices...")

    all_dfs = [pd.read_csv(os.path.join(resmat_dir, f), index_col=0) for f in all_files]

    # Find shared indices
    shared_indices = set(all_dfs[0].index)
    for df in all_dfs[1:]:
        shared_indices = shared_indices.intersection(set(df.index))
    shared_indices = sorted(list(shared_indices))

    # Create oracle (averaged) matrix
    filtered_dfs = [df.loc[shared_indices] for df in all_dfs]
    stacked = np.array([df.values for df in filtered_dfs])
    oracle_matrix = np.nanmean(stacked, axis=0)
    oracle_df = pd.DataFrame(oracle_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

    task_ids = oracle_df.columns.tolist()

    print(f"ColBench oracle shape: {oracle_df.shape}")
    return oracle_df, task_ids, all_dfs


# ==============================================================================
# Experiment Runners
# ==============================================================================

def run_helm_experiment(embedding_type='pca', embedding_dim=48, k_sparsity=4):
    """
    Run HELM benchmark evaluation (Bernoulli IRT).

    Returns results DataFrame with model comparisons.
    """
    print("=" * 60)
    print("HELM BENCHMARK EXPERIMENT")
    print("=" * 60)

    # Load data
    y_df, task_ids = load_helm_data()
    N, J = y_df.shape

    # Load and align embeddings
    embeddings, emb_task_ids, emb_meta = get_embeddings(
        embedding_type=embedding_type,
        dim=embedding_dim,
        k_sparsity=k_sparsity,
        benchmark='helm'
    )

    aligned_emb = align_embeddings_to_tasks(embeddings, emb_task_ids, task_ids, 'helm')
    x_j = torch.tensor(aligned_emb, dtype=torch.float32).to(device)

    # Train/test split
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_vals = y_df.values.astype(np.float32)
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    train_mask = np.zeros_like(y_vals, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]
    test_mask = np.zeros_like(y_vals, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

    y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.0)).to(device)
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    results = []

    # 1. Global Mean
    print("\n--- Global Mean Baseline ---")
    pred_mean = compute_global_mean_baseline(y_data, train_mask_t, y_data.shape)
    y_te = y_data[test_mask_t].cpu().numpy()
    p_te = pred_mean[test_mask_t].cpu().numpy()
    auc_mean = roc_auc_score(y_te, p_te)
    print(f"Test AUC: {auc_mean:.4f}")
    results.append({'Model': 'Average', 'AUC': auc_mean, 'Type': 'baseline'})

    # 2. Rasch IRT
    print("\n--- Rasch IRT Baseline ---")
    p_rasch = train_rasch_baseline(N, J, y_data, train_mask_t)
    p_te = p_rasch[test_mask_t].cpu().numpy()
    auc_rasch = roc_auc_score(y_te, p_te)
    print(f"Test AUC: {auc_rasch:.4f}")
    results.append({'Model': 'Rasch-IRT', 'AUC': auc_rasch, 'Type': 'baseline'})

    # 3. Amortized Difficulty (theta only, no latent factors)
    print("\n--- Amortized Difficulty ---")

    class AmortizedDifficulty(nn.Module):
        def __init__(self, N, d, x_j):
            super().__init__()
            self.register_buffer('x_j', x_j)
            self.theta = nn.Parameter(torch.zeros(N))
            self.diff_proj = nn.Linear(d, 1)

        def forward(self):
            pred_beta = self.diff_proj(self.x_j).squeeze().unsqueeze(0)
            return torch.sigmoid(self.theta.unsqueeze(1) + pred_beta)

    model_ad = AmortizedDifficulty(N, x_j.shape[1], x_j).to(device)
    opt_ad = optim.Adam(model_ad.parameters(), lr=0.01)

    for e in range(1001):
        model_ad.train()
        opt_ad.zero_grad()
        probs = model_ad()
        loss = F.binary_cross_entropy(probs[train_mask_t], y_data[train_mask_t])
        loss.backward()
        opt_ad.step()

    model_ad.eval()
    with torch.no_grad():
        p_ad = model_ad()
        p_te = p_ad[test_mask_t].cpu().numpy()
        auc_ad = roc_auc_score(y_te, p_te)
    print(f"Test AUC: {auc_ad:.4f}")
    results.append({'Model': 'Amortised Difficulty', 'AUC': auc_ad, 'Type': 'amortized'})

    # 4. Sub-Amortized IRT (no SAE)
    print("\n--- Sub-Amortized IRT ---")
    model_sub = BernoulliIRT(N, J, K_MODEL, x_j.shape[1], x_j, dropout=0.5).to(device)
    _, _ = train_model(model_sub, y_data, train_mask_t, y_data, test_mask_t,
                       epochs=1500, model_type='bernoulli')

    model_sub.eval()
    with torch.no_grad():
        p_sub = model_sub()
        p_te = p_sub[test_mask_t].cpu().numpy()
        auc_sub = roc_auc_score(y_te, p_te)
    print(f"Test AUC: {auc_sub:.4f}")
    results.append({'Model': 'Sub-Amortised IRT', 'AUC': auc_sub, 'Type': 'amortized'})

    # 5. Full Amortized IRT (with SAE if available)
    if embedding_type == 'sae':
        print("\n--- Amortized IRT (SAE) ---")
        model_full = BernoulliIRT(N, J, 100, x_j.shape[1], x_j, dropout=0.5).to(device)
        _, _ = train_model(model_full, y_data, train_mask_t, y_data, test_mask_t,
                           epochs=2000, model_type='bernoulli')

        model_full.eval()
        with torch.no_grad():
            p_full = model_full()
            p_te = p_full[test_mask_t].cpu().numpy()
            auc_full = roc_auc_score(y_te, p_te)
        print(f"Test AUC: {auc_full:.4f}")
        results.append({'Model': 'Amortised IRT', 'AUC': auc_full, 'Type': 'amortized'})

    return pd.DataFrame(results)


def run_colbench_experiment(embedding_dim=48, k_sparsity=4, model_type='beta'):
    """
    Run ColBench benchmark evaluation with both PCA and SAE embeddings.

    Args:
        embedding_dim: dimension for PCA/SAE embeddings
        k_sparsity: sparsity parameter for SAE
        model_type: 'bernoulli' or 'beta'

    Returns results DataFrame.
    """
    print("=" * 60)
    print(f"COLBENCH BENCHMARK EXPERIMENT ({model_type.upper()} IRT)")
    print("=" * 60)

    # Load data
    oracle_df, task_ids, all_dfs = load_colbench_data()
    N, J = oracle_df.shape

    # Train/test split (do this BEFORE loading embeddings to ensure consistency)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_vals = oracle_df.values.astype(np.float32)
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    # Create masks - NaN entries are excluded via mask, not imputed
    train_mask = np.zeros_like(y_vals, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]
    test_mask = np.zeros_like(y_vals, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

    # Keep original values, masked entries won't be used in loss
    y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.0)).to(device)
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    results = []

    # Compute metrics helper
    def compute_metrics(probs, y_true, mask):
        p_flat = probs[mask].cpu().numpy()
        y_flat = y_true[mask].cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_flat, p_flat))
        y_bin = (y_flat > 0.5).astype(int)
        auc = roc_auc_score(y_bin, p_flat) if len(np.unique(y_bin)) > 1 else 0.5
        return rmse, auc

    # Load PCA embeddings for baselines and PCA model
    print("\nLoading PCA embeddings...")
    pca_emb, pca_task_ids, _ = get_embeddings(
        embedding_type='pca', dim=embedding_dim, benchmark='colbench'
    )
    pca_aligned = align_embeddings_to_tasks(pca_emb, pca_task_ids, task_ids, 'colbench')
    x_j_pca = torch.tensor(pca_aligned, dtype=torch.float32).to(device)

    # 1. Global Mean
    print("\n--- Global Mean Baseline ---")
    pred_mean = compute_global_mean_baseline(y_data, train_mask_t, y_data.shape)
    rmse_mean, auc_mean = compute_metrics(pred_mean, y_data, test_mask_t)
    print(f"Test RMSE: {rmse_mean:.4f}, AUC: {auc_mean:.4f}")
    results.append({'Model': 'Global Mean', 'RMSE': rmse_mean, 'AUC': auc_mean})

    # 2. Rasch IRT
    print("\n--- Rasch IRT Baseline ---")
    p_rasch = train_rasch_baseline(N, J, y_data, train_mask_t)
    rmse_rasch, auc_rasch = compute_metrics(p_rasch, y_data, test_mask_t)
    print(f"Test RMSE: {rmse_rasch:.4f}, AUC: {auc_rasch:.4f}")
    results.append({'Model': 'Rasch-IRT', 'RMSE': rmse_rasch, 'AUC': auc_rasch})

    # 3. Amortised Difficulty (theta + amortized difficulty, no latent factors)
    print("\n--- Amortised Difficulty ---")

    class AmortizedDifficulty(nn.Module):
        def __init__(self, N, d, x_j):
            super().__init__()
            self.register_buffer('x_j', x_j)
            self.theta = nn.Parameter(torch.zeros(N))
            self.diff_proj = nn.Linear(d, 1)

        def forward(self):
            pred_beta = self.diff_proj(self.x_j).squeeze().unsqueeze(0)
            return torch.sigmoid(self.theta.unsqueeze(1) + pred_beta)

    model_ad = AmortizedDifficulty(N, x_j_pca.shape[1], x_j_pca).to(device)
    opt_ad = optim.Adam(model_ad.parameters(), lr=0.01)

    for e in range(1001):
        model_ad.train()
        opt_ad.zero_grad()
        probs = model_ad()
        loss = F.mse_loss(probs[train_mask_t], y_data[train_mask_t])
        loss.backward()
        opt_ad.step()

    model_ad.eval()
    with torch.no_grad():
        p_ad = model_ad()
        rmse_ad, auc_ad = compute_metrics(p_ad, y_data, test_mask_t)
    print(f"Test RMSE: {rmse_ad:.4f}, AUC: {auc_ad:.4f}")
    results.append({'Model': 'Amortised Difficulty', 'RMSE': rmse_ad, 'AUC': auc_ad})

    # 4. Amortized IRT (PCA)
    print(f"\n--- Amortised IRT (PCA) ---")
    torch.manual_seed(RANDOM_SEED)  # Reset seed for reproducibility
    if model_type == 'bernoulli':
        model_pca = BernoulliIRT(N, J, K_MODEL, x_j_pca.shape[1], x_j_pca, dropout=0.5).to(device)
    else:
        model_pca = BetaIRT(N, J, K_MODEL, x_j_pca.shape[1], x_j_pca, dropout=0.5).to(device)

    _, _ = train_model(model_pca, y_data, train_mask_t, y_data, test_mask_t,
                       epochs=EPOCHS, model_type=model_type)

    model_pca.eval()
    with torch.no_grad():
        probs = model_pca()
        rmse_pca, auc_pca = compute_metrics(probs, y_data, test_mask_t)
        active_dims_pca = model_pca.get_active_dims()

    print(f"Test RMSE: {rmse_pca:.4f}, AUC: {auc_pca:.4f}, Active dims: {active_dims_pca}")
    results.append({'Model': 'Amortised IRT (PCA)', 'RMSE': rmse_pca, 'AUC': auc_pca,
                    'Active_Dims': active_dims_pca})

    # 5. Amortized IRT (SAE)
    print(f"\n--- Amortised IRT (SAE) ---")
    print("Loading SAE embeddings...")
    sae_emb, sae_task_ids, _ = get_embeddings(
        embedding_type='sae', dim=embedding_dim, k_sparsity=k_sparsity, benchmark='colbench'
    )
    sae_aligned = align_embeddings_to_tasks(sae_emb, sae_task_ids, task_ids, 'colbench')
    x_j_sae = torch.tensor(sae_aligned, dtype=torch.float32).to(device)

    torch.manual_seed(RANDOM_SEED)  # Reset seed for reproducibility
    if model_type == 'bernoulli':
        model_sae = BernoulliIRT(N, J, K_MODEL, x_j_sae.shape[1], x_j_sae, dropout=0.5).to(device)
    else:
        model_sae = BetaIRT(N, J, K_MODEL, x_j_sae.shape[1], x_j_sae, dropout=0.5).to(device)

    _, _ = train_model(model_sae, y_data, train_mask_t, y_data, test_mask_t,
                       epochs=EPOCHS, model_type=model_type)

    model_sae.eval()
    with torch.no_grad():
        probs = model_sae()
        rmse_sae, auc_sae = compute_metrics(probs, y_data, test_mask_t)
        active_dims_sae = model_sae.get_active_dims()

    print(f"Test RMSE: {rmse_sae:.4f}, AUC: {auc_sae:.4f}, Active dims: {active_dims_sae}")
    results.append({'Model': 'Amortised IRT (SAE)', 'RMSE': rmse_sae, 'AUC': auc_sae,
                    'Active_Dims': active_dims_sae})

    return pd.DataFrame(results)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train IRT models')
    parser.add_argument('--benchmark', type=str, default='colbench', choices=['helm', 'colbench'],
                        help='Benchmark to evaluate (default: colbench)')
    parser.add_argument('--model', type=str, default='beta', choices=['bernoulli', 'beta'],
                        help='Model type (default: beta)')
    parser.add_argument('--embedding-type', type=str, default='pca', choices=['raw', 'pca', 'sae'],
                        help='Embedding type for HELM (default: pca). ColBench runs both PCA and SAE.')
    parser.add_argument('--embedding-dim', type=int, default=48,
                        help='Embedding dimension (default: 48)')
    parser.add_argument('--k-sparsity', type=int, default=4,
                        help='SAE sparsity (default: 4)')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Using device: {device}")

    if args.benchmark == 'helm':
        results_df = run_helm_experiment(
            embedding_type=args.embedding_type,
            embedding_dim=args.embedding_dim,
            k_sparsity=args.k_sparsity
        )
        output_file = os.path.join(RESULT_DIR, 'helm_results.csv')
    else:
        # ColBench runs both PCA and SAE embeddings
        results_df = run_colbench_experiment(
            embedding_dim=args.embedding_dim,
            k_sparsity=args.k_sparsity,
            model_type=args.model
        )
        output_file = os.path.join(RESULT_DIR, 'colbench_results.csv')

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\n[OUTPUT] Results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
