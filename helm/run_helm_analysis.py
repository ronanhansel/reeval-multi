#!/usr/bin/env python3
"""
HELM Analysis: Baseline Fitting (Naive, Rasch, Amortized IRT with Raw/PCA/SAE)

Merges HELM analysis into a single workflow.
Runs 5 models:
1. Global Mean
2. Rasch IRT
3. Amortized IRT (Raw Embeddings)
4. Amortized IRT (PCA Embeddings)
5. Amortized IRT (SAE Embeddings)
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
from sklearn.decomposition import PCA

try:
    from hypothesaes.quickstart import train_sae
    HAS_SAE_LIB = True
except ImportError:
    HAS_SAE_LIB = False
    print("WARNING: 'hypothesaes' library not found. SAE embeddings will not be available.")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
EMB_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings_cache')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(EMB_CACHE_DIR, exist_ok=True)

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

# Embedding Config
PCA_DIM = 48
SAE_FEATURES = 48
SAE_K = 4

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

def normalize_embeddings(embeddings):
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return embeddings / norms

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
# Data Loading & Processing
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
    
    # Raw embeddings (numpy array)
    raw_embs_np = np.array(aligned_embs, dtype=np.float32)
    print(f"Final Data Shape: {y_df.shape}")
    print(f"Embeddings Shape: {raw_embs_np.shape}")

    return y_df, raw_embs_np

def get_pca_embeddings(raw_embs_np):
    """Generate or load PCA embeddings."""
    cache_path = os.path.join(EMB_CACHE_DIR, f'embeddings_pca_{PCA_DIM}.npy')
    
    if os.path.exists(cache_path):
        print(f"Loading cached PCA embeddings from {cache_path}")
        return np.load(cache_path)
    
    print(f"Generating PCA embeddings (dim={PCA_DIM})...")
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    raw_norm = normalize_embeddings(raw_embs_np)
    pca_embs = pca.fit_transform(raw_norm)
    pca_embs = normalize_embeddings(pca_embs)
    
    np.save(cache_path, pca_embs)
    print(f"Saved PCA embeddings to {cache_path}")
    return pca_embs

def get_sae_embeddings(raw_embs_np):
    """Generate or load SAE embeddings."""
    cache_path = os.path.join(EMB_CACHE_DIR, f'embeddings_sae_{SAE_FEATURES}.npy')
    
    if os.path.exists(cache_path):
        print(f"Loading cached SAE embeddings from {cache_path}")
        return np.load(cache_path)
    
    if not HAS_SAE_LIB:
        print("SAE library not available, returning None")
        return None

    print(f"Generating SAE embeddings (dim={SAE_FEATURES}, k={SAE_K})...")
    raw_norm = normalize_embeddings(raw_embs_np)
    
    sae = train_sae(
        embeddings=raw_norm,
        M=SAE_FEATURES,
        K=SAE_K,
        batch_size=512,
        n_epochs=100,
        learning_rate=5e-4,
        checkpoint_dir='/tmp/_helm_sae_ckpt'
    )
    
    sae_embs = sae.get_activations(raw_norm)
    sae_embs = normalize_embeddings(sae_embs)
    
    np.save(cache_path, sae_embs)
    print(f"Saved SAE embeddings to {cache_path}")
    return sae_embs

def prepare_experiment_data(y_df, x_j_raw, x_j_pca, x_j_sae):
    """Prepare train/test splits and tensors."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_vals = y_df.values.astype(np.float32)
    N, J = y_vals.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    train_mask = np.zeros_like(y_vals, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]

    test_mask = np.zeros_like(y_vals, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

    y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.5)).to(device)
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)
    
    # Process embeddings to tensors
    def to_tensor(arr):
        if arr is None: return None
        t = torch.tensor(arr, dtype=torch.float32).to(device)
        return F.normalize(t, p=2, dim=1)

    return {
        'y_data': y_data,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'train_mask_t': train_mask_t,
        'test_mask_t': test_mask_t,
        'x_j_raw': to_tensor(x_j_raw),
        'x_j_pca': to_tensor(x_j_pca),
        'x_j_sae': to_tensor(x_j_sae),
        'N': N,
        'J': J
    }

# ══════════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(data):
    """Run experiment for all baselines."""
    N = data['N']
    J = data['J']
    y_data = data['y_data']
    train_mask_t = data['train_mask_t']
    test_mask = data['test_mask']
    test_mask_t = data['test_mask_t']
    
    results = {}

    # 1. Global Mean
    print("\nRunning Model 1: Global Mean (Naive)...")
    mean_val = y_data[train_mask_t].mean()
    pred_mean = mean_val.expand_as(y_data)
    results['rmse_mean'] = compute_rmse(pred_mean.cpu().numpy(), y_data.cpu().numpy(), test_mask)
    results['auc_mean'] = evaluate_auc(pred_mean, y_data, test_mask_t)
    print(f"  -> RMSE: {results['rmse_mean']:.4f} | AUC: {results['auc_mean']:.4f}")

    # 2. Rasch IRT
    print("\nRunning Model 2: Rasch IRT...")
    p_rasch = train_rasch(N, J, y_data, train_mask_t)
    results['rmse_rasch'] = compute_rmse(p_rasch.cpu().numpy(), y_data.cpu().numpy(), test_mask)
    results['auc_rasch'] = evaluate_auc(p_rasch, y_data, test_mask_t)
    print(f"  -> RMSE: {results['rmse_rasch']:.4f} | AUC: {results['auc_rasch']:.4f}")

    # Helper for Amortized Models
    def run_amortized(name, x_j):
        if x_j is None:
            print(f"\nSkipping Model: Amortized IRT ({name}) - Embeddings missing")
            return float('nan'), float('nan')
        
        print(f"\nRunning Model: Amortized IRT ({name})...")
        emb_dim = x_j.shape[1]
        model = AmortizedIRTModel(N, J, K_MODEL, emb_dim, x_j, dropout=0.5).to(device)
        best_rmse = train_amortized_irt(model, y_data, train_mask_t, y_data, test_mask)
        
        model.eval()
        with torch.no_grad():
            p_out = model()
            auc = evaluate_auc(p_out, y_data, test_mask_t)
        
        print(f"  -> RMSE: {best_rmse:.4f} | AUC: {auc:.4f}")
        return best_rmse, auc

    # 3. Amortized (Raw)
    r, a = run_amortized("Raw", data['x_j_raw'])
    results['rmse_amortized_raw'] = r
    results['auc_amortized_raw'] = a

    # 4. Amortized (PCA)
    r, a = run_amortized("PCA", data['x_j_pca'])
    results['rmse_amortized_pca'] = r
    results['auc_amortized_pca'] = a

    # 5. Amortized (SAE)
    r, a = run_amortized("SAE", data['x_j_sae'])
    results['rmse_amortized_sae'] = r
    results['auc_amortized_sae'] = a

    return results


def main():
    print(f"Using device: {device}")

    # Load data
    y_df, raw_embs_np = load_data()
    
    # Get/Generate Embeddings
    pca_embs_np = get_pca_embeddings(raw_embs_np)
    sae_embs_np = get_sae_embeddings(raw_embs_np)

    # Prepare experiment data
    data = prepare_experiment_data(y_df, raw_embs_np, pca_embs_np, sae_embs_np)

    # Run experiment
    results = run_experiment(data)

    # Save results
    results_df = pd.DataFrame([results])
    out_path = os.path.join(RESULT_DIR, 'helm_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n[OUTPUT] Saved results to: {out_path}")


if __name__ == '__main__':
    main()