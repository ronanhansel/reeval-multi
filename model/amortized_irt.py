#!/usr/bin/env python3
"""
Amortized IRT Experiment (Contribution 1)

Evaluates the Amortized Factor Model with pre-computed embeddings.
Compares Global Mean, Rasch-IRT, and Amortized IRT across different
numbers of response matrix samples.

Supports both Beta IRT (train on averaged probabilities P̂) and
Bernoulli IRT (train on binarized responses Y).

Results are saved to CSV for separate plotting.

Usage:
    python amortized_irt.py --embedding-type pca        # Use PCA embeddings (default)
    python amortized_irt.py --embedding-type sae        # Use SAE embeddings
    python amortized_irt.py --embedding-type raw        # Use raw 4096-dim embeddings
    python amortized_irt.py --model-type beta           # Train on P̂ (default)
    python amortized_irt.py --model-type bernoulli      # Train on binary Y
    python amortized_irt.py --n-samples 22              # Run only n=22 (fast)
    python amortized_irt.py --n-samples 1,22            # Run n=1 and n=22
"""

import argparse
import ast
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from huggingface_hub import snapshot_download

from utils import compute_rmse, evaluate_auc

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

# Data paths
HF_REPO_ID = "ronanhansel/data-reeval-multi"

# Data split
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Model architecture
K_MODEL = 30

# Tau sparsity settings
LAMBDA_TAU = 1.38
TAU_INIT = 0.2
TAU_WARMUP = 20
RAMP_EPOCHS = 50
SNAPPING_THRESHOLD = 0.01
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# Training settings
EPOCHS = 250
EVAL_EVERY = 25

# Learning rates
LR_THETA = 0.01
LR_GLOBAL = 0.002
WD_THETA = 5.0
WD_W = 0.1

# Beta distribution precision parameter
BETA_PHI = 10.0

warnings.filterwarnings('ignore')

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# ══════════════════════════════════════════════════════════════════════════════
# Model Definition
# ══════════════════════════════════════════════════════════════════════════════

class AmortizedIRTModel(nn.Module):
    """
    Amortized IRT model with automatic relevance determination (ARD).

    Item loadings are amortized from pre-computed embeddings via learned projection W.
    Tau parameters enable automatic dimensionality discovery through sparsity.
    """

    def __init__(self, N, J, K, d, x_j_emb, dropout=0.7):
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


def train_amortized_irt(model, y_train, train_mask_t, y_oracle, test_mask_oracle,
                        model_type='beta', beta_phi=BETA_PHI, epochs=EPOCHS):
    """Train amortized IRT model with ARD sparsity regularization.

    Args:
        model_type: 'beta' uses Beta distribution NLL for continuous targets,
                    'bernoulli' uses Bernoulli NLL for binary targets
        beta_phi: Precision parameter for Beta distribution (only used when model_type='beta')
    """
    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
        {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
        {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
    ])

    best_rmse = float('inf')

    eps = 1e-6

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        probs = model()

        # Reconstruction loss using torch.distributions
        p = probs[train_mask_t].clamp(eps, 1 - eps)
        
        if model_type == 'beta':
            # Beta NLL: α = μφ, β = (1-μ)φ
            y = y_train[train_mask_t].clamp(eps, 1 - eps)
            dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
            loss_fit = -dist.log_prob(y).mean()
        else:
            # Bernoulli NLL
            y = y_train[train_mask_t]
            dist = torch.distributions.Bernoulli(probs=p)
            loss_fit = -dist.log_prob(y).mean()

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
        if epoch % EVAL_EVERY == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                p_eval = model()
                
                # Train metrics
                train_rmse = compute_rmse(p_eval.cpu().numpy(), y_train.cpu().numpy(), train_mask_t.cpu().numpy())
                train_auc = evaluate_auc(p_eval, y_train, train_mask_t)
                
                # Test metrics
                curr_rmse = compute_rmse(p_eval.cpu().numpy(), y_oracle.cpu().numpy(), test_mask_oracle)
                curr_auc = evaluate_auc(p_eval, y_oracle, torch.from_numpy(test_mask_oracle).to(device))
                best_rmse = min(best_rmse, curr_rmse)
                
                tau_vals = model.get_tau()
                print(f"Epoch {epoch:4d} | Loss: {loss_fit.item():.4f} | Train RMSE: {train_rmse:.4f} AUC: {train_auc:.4f} | Test RMSE: {curr_rmse:.4f} AUC: {curr_auc:.4f} | Tau (min/mean/max): {tau_vals.min().item():.3f}/{tau_vals.mean().item():.3f}/{tau_vals.max().item():.3f}")

    return best_rmse


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(embedding_type='pca', embedding_dim=48):
    """
    Load response matrices and pre-computed embeddings.

    Args:
        embedding_type: 'raw', 'pca', or 'sae'
        embedding_dim: dimension for pca/sae embeddings (ignored for raw)
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resmat_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix')
    post_rev_dir = os.path.join(resmat_dir, 'post-revision')
    
    # Load from local directory instead of huggingface cache
    colbench_dir = os.path.join(post_rev_dir, 'colbench_backend_programming', 'resmat')
    processed_emb_dir = os.path.join(repo_root, 'model', 'processed_embeddings')
    raw_emb_file = os.path.join(resmat_dir, 'all_benchmarks_embeddings_4096_8B.pkl')

    # 1. Load ColBench response matrices
    colbench_files = sorted([f for f in os.listdir(colbench_dir) if f.startswith('resmat')])
        
    colbench_dfs = []
    for f in colbench_files:
        df = pd.read_csv(os.path.join(colbench_dir, f), index_col=0)
        df.columns = [f"colbench_backend_programming.{c}" if not str(c).startswith("colbench") else c for c in df.columns]
        colbench_dfs.append(df)
    
    # 2. Load other benchmarks response matrices
    other_benchmarks = [b for b in os.listdir(post_rev_dir) if b != 'colbench_backend_programming' and os.path.isdir(os.path.join(post_rev_dir, b))]
    
    # Find maximum number of runs across other benchmarks
    max_other_runs = 0
    for benchmark in other_benchmarks:
        b_resmat_dir = os.path.join(post_rev_dir, benchmark, 'resmat')
        if os.path.exists(b_resmat_dir):
            b_files = [f for f in os.listdir(b_resmat_dir) if f.startswith('resmat')]
            max_other_runs = max(max_other_runs, len(b_files))
            
    other_dfs = []
    for i in range(max_other_runs):
        combined_df = None
        for benchmark in other_benchmarks:
            b_resmat_dir = os.path.join(post_rev_dir, benchmark, 'resmat')
            if not os.path.exists(b_resmat_dir): continue
            
            b_files = sorted([f for f in os.listdir(b_resmat_dir) if f.startswith('resmat')])
            if i < len(b_files):
                df = pd.read_csv(os.path.join(b_resmat_dir, b_files[i]), index_col=0)
                # Ensure unique columns by prefixing with benchmark name if not already
                df.columns = [f"{benchmark}.{c}" if not str(c).startswith(benchmark) else c for c in df.columns]
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], axis=1, join='outer')
        
        if combined_df is not None:
            other_dfs.append(combined_df)
            
    max_runs = max(len(colbench_dfs), len(other_dfs))
    all_dfs = []
    for i in range(max_runs):
        dfs_to_concat = []
        if i < len(colbench_dfs):
            dfs_to_concat.append(colbench_dfs[i])
        else:
            dfs_to_concat.append(colbench_dfs[i % len(colbench_dfs)])
            
        if i < len(other_dfs):
            dfs_to_concat.append(other_dfs[i])
        elif len(other_dfs) > 0:
            dfs_to_concat.append(other_dfs[-1])
            
        if dfs_to_concat:
            combined = pd.concat(dfs_to_concat, axis=1, join='outer')
            all_dfs.append(combined)

    # Find global shared indices (models present in all matrices)
    global_shared_indices = set(all_dfs[0].index)
    for df in all_dfs[1:]:
        global_shared_indices = global_shared_indices.intersection(set(df.index))
    global_shared_indices = sorted(list(global_shared_indices))

    # Load embeddings based on type
    if embedding_type == 'raw':
        emb_file = raw_emb_file
    elif embedding_type == 'pca':
        emb_file = os.path.join(processed_emb_dir, f'embeddings_pca_{embedding_dim}.pkl')
    elif embedding_type == 'sae':
        emb_file = os.path.join(processed_emb_dir, f'embeddings_sae_{embedding_dim}.pkl')
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    # Fall back to raw if processed embeddings don't exist
    if not os.path.exists(emb_file):
        print(f"Warning: {emb_file} not found. Falling back to raw embeddings.")
        print("Run 'python generate_embeddings.py' to generate processed embeddings.")
        emb_file = raw_emb_file
        embedding_type = 'raw'

    print(f"Loading {embedding_type} embeddings from {emb_file}...")
    emb_df = pd.read_pickle(emb_file)

    # Build embedding map
    raw_embs_map = {}
    id_col = 'task_id' if 'task_id' in emb_df.columns else 'benchmark.task_id'

    for _, r in emb_df.iterrows():
        task_id = str(r[id_col])
        raw_embs_map[task_id] = r['embedding']

        # Handle colbench naming variations
        if task_id.startswith('colbench_backend_programming'):
            suffix = task_id.split('.')[-1]
            raw_embs_map[f'colbench.{suffix}'] = r['embedding']

    return all_dfs, global_shared_indices, raw_embs_map, embedding_type


def prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map):
    """Prepare oracle ground truth, train/test splits, and embedding tensors."""
    print("=" * 60)
    print("PREPARING EXPERIMENT DATA")
    print("=" * 60)

    # Find the union of all columns across all dfs to handle potential missing items in some runs
    all_columns = sorted(list(set().union(*[df.columns for df in all_dfs])))
    
    # Create oracle matrix (average across all response matrices)
    oracle_dfs_filtered = [df.loc[global_shared_indices].reindex(columns=all_columns) for df in all_dfs]
    oracle_stacked = np.array([df.values for df in oracle_dfs_filtered], dtype=float)
    oracle_matrix = np.nanmean(oracle_stacked, axis=0)
    oracle_df = pd.DataFrame(oracle_matrix, index=global_shared_indices, columns=all_columns)

    print(f"Total matrices: {len(all_dfs)}")
    print(f"Global user intersection: {len(global_shared_indices)} users")
    print(f"Oracle matrix shape: {oracle_df.shape}")

    # Train/test split (by items/columns)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N, J = oracle_df.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    print(f"Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    # Prepare tensors
    oracle_values_clean = np.nan_to_num(oracle_df.values, nan=0.5)
    y_oracle = torch.from_numpy(oracle_values_clean.astype(np.float32)).to(device)

    train_mask = np.zeros_like(oracle_df.values, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(oracle_df.values)[:, train_idx]

    test_mask = np.zeros_like(oracle_df.values, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(oracle_df.values)[:, test_idx]

    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    # Align embeddings with oracle columns
    task_ids = oracle_df.columns.tolist()
    embeddings = []
    for task_id in task_ids:
        emb = raw_embs_map.get(str(task_id))
        if emb is None and task_id.startswith('colbench.'):
            number = task_id.split('.')[-1]
            emb = raw_embs_map.get(f'colbench_backend_programming.{number}')
        if emb is None:
            # Use zeros for missing embeddings
            sample_emb = next(iter(raw_embs_map.values()))
            emb = np.zeros(len(sample_emb) if hasattr(sample_emb, '__len__') else 4096)
        elif isinstance(emb, str):
            emb = ast.literal_eval(emb)
        embeddings.append(np.array(emb, dtype=np.float32))

    embeddings = np.stack(embeddings)

    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    x_j = torch.tensor(embeddings, dtype=torch.float32).to(device)

    print(f"Embeddings shape: {x_j.shape}")

    return {
        'oracle_df': oracle_df,
        'y_oracle': y_oracle,
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

def run_experiment(n_files, all_dfs, global_shared_indices, data, model_type='beta',
                   beta_phi=BETA_PHI):
    """Run experiment for a specific number of sample files.

    Args:
        model_type: 'beta' to train on averaged probabilities P̂,
                    'bernoulli' to train on binarized responses Y
        beta_phi: Precision parameter for Beta distribution
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N = data['N']
    J = data['J']
    y_oracle = data['y_oracle']
    train_idx = data['train_idx']
    test_mask = data['test_mask']
    test_mask_t = data['test_mask_t']
    x_j = data['x_j']
    embedding_dim = data['embedding_dim']

    # Prepare training data from n_files samples
    current_dfs = [all_dfs[i].loc[global_shared_indices] for i in range(n_files)]
    all_columns = sorted(list(set().union(*[df.columns for df in current_dfs])))
    current_dfs = [df.reindex(columns=all_columns) for df in current_dfs]
    current_stacked = np.array([df.values for df in current_dfs], dtype=float)
    train_target_matrix = np.nanmean(current_stacked, axis=0)
    train_target_df = pd.DataFrame(train_target_matrix, index=global_shared_indices, columns=all_columns)

    train_values = train_target_df.values.copy()

    if model_type == 'bernoulli':
        # Binarize: convert averaged probabilities to binary responses
        train_values = (train_values > 0.5).astype(np.float32)

    train_values = np.nan_to_num(train_values, nan=0.5)
    y_train = torch.from_numpy(train_values.astype(np.float32)).to(device)

    train_mask_current = np.zeros_like(train_target_df.values, dtype=bool)
    train_mask_current[:, train_idx] = ~np.isnan(train_target_df.values)[:, train_idx]
    train_mask_current_t = torch.from_numpy(train_mask_current).to(device)

    # 1. Global Mean baseline
    mean_val = y_train[train_mask_current_t].mean()
    pred_mean = mean_val.expand_as(y_oracle)
    rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_mean = evaluate_auc(pred_mean, y_oracle, test_mask_t)

    # 2. Rasch IRT baseline
    p_rasch = train_rasch(N, J, y_train, train_mask_current_t)
    rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_rasch = evaluate_auc(p_rasch, y_oracle, test_mask_t)

    # 3. Amortized IRT (our method)
    model = AmortizedIRTModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.5).to(device)
    best_rmse = train_amortized_irt(model, y_train, train_mask_current_t, y_oracle, test_mask,
                                     model_type=model_type, beta_phi=beta_phi)

    model.eval()
    with torch.no_grad():
        p_amortized = model()
        auc_amortized = evaluate_auc(p_amortized, y_oracle, test_mask_t)
        
        tau_val = model.get_tau()
        active_mask = tau_val > TAU_THRESHOLD
        active_dims = active_mask.sum().item()
        
        active_dim_indices = torch.nonzero(active_mask).squeeze().cpu().tolist()
        if isinstance(active_dim_indices, int):
            active_dim_indices = [active_dim_indices]

    return {
        'n_samples': n_files,
        'model_type': model_type,
        'rmse_mean': rmse_mean,
        'rmse_rasch': rmse_rasch,
        'rmse_amortized': best_rmse,
        'auc_mean': auc_mean,
        'auc_rasch': auc_rasch,
        'auc_amortized': auc_amortized,
        'active_dims': active_dims,
        'active_indices': str(active_dim_indices)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_n_samples(arg, total_files):
    """Parse --n-samples argument into list of integers."""
    if arg == 'all':
        return list(range(1, total_files + 1))

    result = []
    for part in arg.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))

    result = [n for n in result if 1 <= n <= total_files]
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description='Amortized IRT Experiment')
    parser.add_argument(
        '--embedding-type', type=str, default='pca',
        choices=['raw', 'pca', 'sae'],
        help='Type of embeddings to use (default: pca)'
    )
    parser.add_argument(
        '--embedding-dim', type=int, default=48,
        help='Embedding dimension for pca/sae (default: 48)'
    )
    parser.add_argument(
        '--model-type', type=str, default='beta',
        choices=['beta', 'bernoulli'],
        help='IRT model type: beta (train on P̂) or bernoulli (train on binary Y) (default: beta)'
    )
    parser.add_argument(
        '--beta-phi', type=float, default=BETA_PHI,
        help=f'Beta distribution precision parameter φ (default: {BETA_PHI}). Higher = more concentrated.'
    )
    parser.add_argument(
        '--n-samples', type=str, default='all',
        help='Which n values to run. Examples: "all", "22", "1,22", "1-5,22"'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: result/amortized_irt_{embedding_type}_{model_type}.csv)'
    )
    parser.add_argument('--lambda-tau', type=float, default=None, help='Override LAMBDA_TAU')
    parser.add_argument('--wd-theta', type=float, default=None, help='Override WD_THETA')
    parser.add_argument('--epochs', type=int, default=None, help='Override EPOCHS')
    parser.add_argument('--snapping-threshold', type=float, default=None, help='Override SNAPPING_THRESHOLD')
    args = parser.parse_args()

    global LAMBDA_TAU, WD_THETA, EPOCHS, SNAPPING_THRESHOLD
    if args.lambda_tau is not None:
        LAMBDA_TAU = args.lambda_tau
    if args.wd_theta is not None:
        WD_THETA = args.wd_theta
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.snapping_threshold is not None:
        SNAPPING_THRESHOLD = args.snapping_threshold

    print(f"Using device: {device}")
    print(f"Embedding type: {args.embedding_type}")
    if args.model_type == 'beta':
        print(f"Model type: beta (Beta distribution NLL on P̂, φ={args.beta_phi})")
    else:
        print(f"Model type: bernoulli (Bernoulli NLL on binary Y)")
    
    print(f"Hyperparameters -> LAMBDA_TAU: {LAMBDA_TAU} | WD_THETA: {WD_THETA} | EPOCHS: {EPOCHS} | SNAPPING: {SNAPPING_THRESHOLD}")

    # Load data
    all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data(
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim
    )
    total_files = len(all_dfs)

    # Parse n_samples argument
    n_values = parse_n_samples(args.n_samples, total_files)
    print(f"\nWill run experiments for n = {n_values}")

    # Prepare experiment data
    data = prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map)

    # Run experiments
    print("\n" + "=" * 60)
    print(f"STARTING EXPERIMENTS ({actual_emb_type.upper()} embeddings, {args.model_type.upper()} model)")
    print("=" * 60)

    results = []
    for i, n in enumerate(n_values):
        print(f"\n[{i+1}/{len(n_values)}] Processing with n={n} sample(s)...")
        result = run_experiment(n, all_dfs, global_shared_indices, data,
                                model_type=args.model_type, beta_phi=args.beta_phi)
        result['embedding_type'] = actual_emb_type
        results.append(result)

        print(f"   -> RMSE | Mean: {result['rmse_mean']:.4f} | Rasch: {result['rmse_rasch']:.4f} | "
              f"Amortized: {result['rmse_amortized']:.4f}")
        print(f"   -> AUC  | Mean: {result['auc_mean']:.4f} | Rasch: {result['auc_rasch']:.4f} | "
              f"Amortized: {result['auc_amortized']:.4f} | Active dims: {result['active_dims']}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    output_path = args.output or os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}.csv')
    df_results.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
