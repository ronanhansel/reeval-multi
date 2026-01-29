#!/usr/bin/env python3
"""
HELM Analysis: Baseline Fitting (Naive, Rasch, Amortized IRT)

Merges HELM analysis into a single workflow, similar to hal/amortized_irt.py.
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import ast
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from huggingface_hub import snapshot_download
from sklearn.metrics import mean_squared_error, roc_auc_score

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

# Data paths
HF_REPO_ID = "ronanhansel/data-reeval-multi"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-reeval-multi')

# Data split
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Model architecture (Amortized IRT)
K_MODEL = 30

# Tau sparsity settings
LAMBDA_TAU = 0.002
TAU_INIT = 0.5
TAU_WARMUP = 200
RAMP_EPOCHS = 200
SNAPPING_THRESHOLD = 0.005
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# Training settings
EPOCHS = 1000
EVAL_EVERY = 100

# Learning rates
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 1e-3
WD_W = 1e-5

warnings.filterwarnings('ignore')

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# ══════════════════════════════════════════════════════════════════════════════
# Utils
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmse(predictions, targets, mask):
    """Compute RMSE over masked entries."""
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))


def evaluate_auc(y_pred_tensor, y_true_tensor, mask_tensor):
    """Compute AUC over masked entries."""
    y_true_flat = y_true_tensor[mask_tensor].detach().cpu().numpy()
    y_pred_flat = y_pred_tensor[mask_tensor].detach().cpu().numpy()
    y_true_binary = (y_true_flat > 0.5).astype(int)
    if len(np.unique(y_true_binary)) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true_binary, y_pred_flat)
    except ValueError:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Model Definition
# ══════════════════════════════════════════════════════════════════════════════

class AmortizedIRTModel(nn.Module):
    """
    Amortized IRT model with automatic relevance determination (ARD).
    """

    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)  # Pre-computed embeddings (J x d)
        self.dropout = dropout

        # User (model) parameters
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)  # Latent abilities
        self.theta_bias = nn.Parameter(torch.zeros(N))

        # Global parameters
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)  # Projection matrix
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)  # ARD scales

        # Item difficulty projection
        self.difficulty_proj = nn.Linear(d, 1)

    def get_tau(self):
        """Get non-negative tau values (ReLU ensures exact sparsity)."""
        return F.relu(self.tau_raw)

    def forward(self):
        # Normalize W for scale invariance
        W_norm = F.normalize(self.W, dim=1)

        # Amortized item loadings: a_j = tau * (W @ x_j)
        base_loadings = self.x_j @ W_norm.T  # (J, K)
        tau = self.get_tau()
        a_j = base_loadings * tau.unsqueeze(0)

        # Dropout for regularization
        if self.training and self.dropout > 0:
            a_j = F.dropout(a_j, p=self.dropout)

        # Item difficulty
        diff = self.difficulty_proj(self.x_j).squeeze()

        # Response probability: sigmoid(theta @ a_j + difficulty + biases)
        logits = self.theta @ a_j.T + diff.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        return torch.sigmoid(logits)


# ══════════════════════════════════════════════════════════════════════════════
# Training Functions
# ══════════════════════════════════════════════════════════════════════════════

def train_rasch(N, J, y_train, train_mask_t, n_outer_iter=100):
    """Train a basic Rasch IRT model (baseline)."""
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
        total_loss = (loss * train_mask_t).sum() / train_mask_t.sum()
        total_loss.backward()
        return total_loss

    for _ in range(n_outer_iter):
        optimizer.step(closure)

    with torch.no_grad():
        probs = torch.sigmoid(theta.unsqueeze(1) - beta.unsqueeze(0))

    return probs


def train_amortized_irt(model, y_train, train_mask_t, y_oracle, test_mask_oracle, epochs=EPOCHS):
    """Train amortized IRT model with ARD sparsity regularization."""
    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
        {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
        {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
    ])

    best_rmse = float('inf')

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        probs = model()

        # Reconstruction loss
        loss_fit = F.binary_cross_entropy(probs[train_mask_t], y_train[train_mask_t])

        # ARD sparsity schedule (warmup -> ramp -> full)
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

        # Tau snapping (enforce exact zeros)
        if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
            with torch.no_grad():
                active_mask = model.get_tau() > SNAPPING_THRESHOLD
                for k in range(K_MODEL):
                    if not active_mask[k]:
                        model.tau_raw[k] = DEAD_ZONE_VALUE

        # Evaluation
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p_eval = model()
                curr_rmse = compute_rmse(p_eval.cpu().numpy(), y_oracle.cpu().numpy(), test_mask_oracle)
                best_rmse = min(best_rmse, curr_rmse)

    return best_rmse


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load response matrices and embeddings for HELM."""
    resmat_path = os.path.join(DATA_DIR, 'resmat.pkl')
    emb_path = os.path.join(DATA_DIR, 'embed_meta-llama_Llama-3.1-8B-Instruct.pkl')

    # Download from HuggingFace if data not present
    if not os.path.exists(resmat_path) or not os.path.exists(emb_path):
        print(f"Data not found locally. Downloading from HuggingFace ({HF_REPO_ID})...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print("Download complete.")

    print(f"Loading HELM data from {resmat_path}...")
    y_df = pd.read_pickle(resmat_path)
    emb_df = pd.read_pickle(emb_path)

    # Filter Valid Rows/Cols
    y_df = y_df[y_df.notna().any(axis=1)]
    valid_cols_list = []
    for c in y_df.columns:
        valid_cols_list.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
    y_df = y_df.iloc[:, valid_cols_list]

    # Align Embeddings
    print("Aligning Embeddings...")
    if 'question' not in emb_df.columns:
        text_col = [c for c in emb_df.columns if 'text' in str(c) or 'question' in str(c)][0]
        emb_df = emb_df.rename(columns={text_col: 'question'})

    emb_map = {}
    for _, row in emb_df.iterrows():
        q_text = row['question']
        emb = row['embedding']
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)
        emb_map[q_text] = emb

    # Get Questions
    try:
        if isinstance(y_df.columns, pd.MultiIndex) and 'input.text' in y_df.columns.names:
            questions = y_df.columns.get_level_values('input.text').tolist()
        else:
            questions = [str(c) for c in y_df.columns]
    except:
        questions = [str(c) for c in y_df.columns]

    # Build Aligned Tensors
    aligned_embs = []
    valid_indices = []
    for i, q in enumerate(questions):
        if q in emb_map:
            aligned_embs.append(emb_map[q])
            valid_indices.append(True)
        else:
            valid_indices.append(False)

    valid_indices = np.array(valid_indices)
    y_df = y_df.iloc[:, valid_indices]
    
    x_j_dense = torch.tensor(np.array(aligned_embs), dtype=torch.float32).to(device)
    x_j_dense = F.normalize(x_j_dense, p=2, dim=1)

    print(f"Final Data Shape: {y_df.shape}")
    print(f"Embeddings Shape: {x_j_dense.shape}")

    return y_df, x_j_dense


def prepare_experiment_data(y_df, x_j):
    """Prepare train/test splits."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_vals = y_df.values.astype(np.float32)
    N, J = y_vals.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    print(f"Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    train_mask = np.zeros_like(y_vals, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]

    test_mask = np.zeros_like(y_vals, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

    y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.5)).to(device)
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    return {
        'y_data': y_data,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'train_mask_t': train_mask_t,
        'test_mask_t': test_mask_t,
        'x_j': x_j,
        'N': N,
        'J': J,
        'embedding_dim': x_j.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(data):
    """Run experiment for all baselines."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N = data['N']
    J = data['J']
    y_data = data['y_data']
    train_mask_t = data['train_mask_t']
    test_mask = data['test_mask']
    test_mask_t = data['test_mask_t']
    x_j = data['x_j']
    embedding_dim = data['embedding_dim']

    print("\nRunning Model 1: Global Mean (Naive)...")
    mean_val = y_data[train_mask_t].mean()
    pred_mean = mean_val.expand_as(y_data)
    rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_data.cpu().numpy(), test_mask)
    auc_mean = evaluate_auc(pred_mean, y_data, test_mask_t)
    print(f"  -> RMSE: {rmse_mean:.4f} | AUC: {auc_mean:.4f}")

    print("\nRunning Model 2: Rasch IRT...")
    p_rasch = train_rasch(N, J, y_data, train_mask_t)
    rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_data.cpu().numpy(), test_mask)
    auc_rasch = evaluate_auc(p_rasch, y_data, test_mask_t)
    print(f"  -> RMSE: {rmse_rasch:.4f} | AUC: {auc_rasch:.4f}")

    print("\nRunning Model 3: Amortized IRT...")
    model = AmortizedIRTModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.5).to(device)
    best_rmse_amortized = train_amortized_irt(model, y_data, train_mask_t, y_data, test_mask)
    
    model.eval()
    with torch.no_grad():
        p_amortized = model()
        auc_amortized = evaluate_auc(p_amortized, y_data, test_mask_t)
    print(f"  -> RMSE: {best_rmse_amortized:.4f} | AUC: {auc_amortized:.4f}")

    return {
        'rmse_mean': rmse_mean,
        'rmse_rasch': rmse_rasch,
        'rmse_amortized': best_rmse_amortized,
        'auc_mean': auc_mean,
        'auc_rasch': auc_rasch,
        'auc_amortized': auc_amortized,
    }


def main():
    print(f"Using device: {device}")

    # Load data
    y_df, x_j = load_data()

    # Prepare experiment data
    data = prepare_experiment_data(y_df, x_j)

    # Run experiment
    results = run_experiment(data)

    # Save results
    results_df = pd.DataFrame([results])
    out_path = os.path.join(RESULT_DIR, 'helm_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n[OUTPUT] Saved results to: {out_path}")


if __name__ == '__main__':
    main()
