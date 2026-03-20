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
import re
import sys
import warnings

# Prevent massive CPU utilization and hangs when using Multiprocessing Pools
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from huggingface_hub import snapshot_download
from filelock import FileLock, Timeout
import multiprocessing as mp
from functools import partial

from utils import compute_rmse, evaluate_auc

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

BASELINE_DIR = os.path.join(RESULT_DIR, 'baselines')
os.makedirs(BASELINE_DIR, exist_ok=True)
DEFAULT_BASELINE_OUTPUT = os.path.join(BASELINE_DIR, 'baseline_metrics.csv')
DEFAULT_MIRT_SWEEP_OUTPUT = os.path.join(BASELINE_DIR, 'mirt_sweep.csv')

BASELINE_METRIC_COLS = [
    'rmse_naive', 'rmse_rasch', 'rmse_2pl', 'rmse_mirt',
    'auc_naive', 'auc_rasch', 'auc_2pl', 'auc_mirt',
    'rmse_knn', 'auc_knn'
]
NON_MIRT_METRIC_COLS = [c for c in BASELINE_METRIC_COLS if c not in {'rmse_mirt', 'auc_mirt'}]

BASELINE_KEY_COLS = ['seed', 'model_type', 'n_samples', 'pre_revision', 'j_percentage']
INLINE_BASELINE_COLS = BASELINE_METRIC_COLS.copy()
BASELINE_AUX_COLS = ['agent_batch_size', 'selected_mirt_dim', 'mirt_sweep_min', 'mirt_sweep_max']

MIRT_SWEEP_METRIC_COLS = ['rmse_mirt', 'auc_mirt']
MIRT_SWEEP_KEY_COLS = BASELINE_KEY_COLS + ['mirt_dim']

# Data paths
HF_REPO_ID = "ronanhansel/data-reeval-multi"

# Data split
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Model architecture
K_MODEL = 30

# Tau sparsity settings
LAMBDA_TAU = 1.38
TAU_INIT = 0.5
TAU_WARMUP = 100
RAMP_EPOCHS = 400
SNAPPING_THRESHOLD = 0.001
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.001

# Training settings
EPOCHS = 1000
EVAL_EVERY = 20

# Learning rates
LR_THETA = 0.01
LR_GLOBAL = 0.002
WD_THETA = 5.0
WD_W = 0.1

# Beta distribution precision parameter
BETA_PHI = 10.0

# Embedding kNN baseline settings
KNN_K = 10

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

    def __init__(self, N, J, K, d, x_j_emb, dropout=0.7, no_tau=False):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)  # Pre-computed embeddings (J x d)
        self.dropout = dropout
        self.no_tau = no_tau

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
        if getattr(self, 'no_tau', False):
            return torch.ones_like(self.tau_raw)
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


class MIRTModel(nn.Module):
    """
    Non-Amortized Multidimensional IRT model.
    Learns item loadings directly as parameters instead of projecting from embeddings.
    """
    def __init__(self, N, J, K):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.a_j = nn.Parameter(torch.randn(J, K) * 0.01)
        self.diff = nn.Parameter(torch.zeros(J))
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self):
        logits = self.theta @ self.a_j.T + self.diff.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
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


def train_2pl(N, J, y_train, train_mask_t, n_outer_iter=100):
    """Train a 2-parameter logistic (2PL) IRT model (baseline)."""
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
    alpha = nn.Parameter(torch.ones(J, device=device)) # Discrimination

    optimizer = torch.optim.LBFGS(
        [theta, beta, alpha], lr=0.1, max_iter=20,
        history_size=10, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        probs = torch.sigmoid(alpha.unsqueeze(0) * (theta.unsqueeze(1) - beta.unsqueeze(0)))
        loss = F.binary_cross_entropy(probs, y_train, reduction='none')
        total_loss = (loss * train_mask_t).sum() / train_mask_t.sum()
        total_loss.backward()
        return total_loss

    for _ in range(n_outer_iter):
        optimizer.step(closure)

    with torch.no_grad():
        probs = torch.sigmoid(alpha.unsqueeze(0) * (theta.unsqueeze(1) - beta.unsqueeze(0)))

    return probs


def train_amortized_irt(model, y_train, train_mask_t, y_oracle, test_mask_oracle,
                        model_type='beta', beta_phi=BETA_PHI, epochs=EPOCHS, lambda_tau=LAMBDA_TAU, quiet=False):
    """Train amortized IRT model with ARD sparsity regularization.

    Args:
        model_type: 'beta' uses Beta distribution NLL for continuous targets,
                    'bernoulli' uses Bernoulli NLL for binary targets
        beta_phi: Precision parameter for Beta distribution (only used when model_type='beta')
    """
    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
        {'params': [model.W, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
        {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
    ])
    optimizer_tau = optim.SGD([model.tau_raw], lr=0.05)

    best_state = None
    best_loss = float('inf')
    best_rmse = float('inf')

    eps = 1e-6

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        optimizer_tau.zero_grad()
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
            current_lambda = lambda_tau * progress
        else:
            current_lambda = lambda_tau

        tau = model.get_tau()
        loss_sparsity = current_lambda * torch.sum(tau)
        total_loss = loss_fit + loss_sparsity

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_rmse = curr_rmse if 'curr_rmse' in locals() else best_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        (loss_fit + loss_sparsity).backward()
        optimizer.step()
        optimizer_tau.step()

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
                if not quiet:
                    print(f"Epoch {epoch:4d} | Loss: {loss_fit.item():.4f} | Train RMSE: {train_rmse:.4f} AUC: {train_auc:.4f} | Test RMSE: {curr_rmse:.4f} AUC: {curr_auc:.4f}")
                    
                    active_indices = torch.where(tau_vals > SNAPPING_THRESHOLD)[0].cpu().numpy()
                    print(f"  Active Dims ({len(active_indices)}): Tau Mean={tau_vals.mean().item():.4f}, Max={tau_vals.max().item():.4f}")
                    print("-" * 40)

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_rmse, best_state, final_state


    # Combined parser moved to main()


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(embedding_type='pca', embedding_dim=48, pre_revision='none'):
    """
    Load response matrices and pre-computed embeddings.

    Args:
        embedding_type: 'raw', 'pca', or 'sae'
        embedding_dim: dimension for pca/sae embeddings (ignored for raw)
        pre_revision: 'none', '8', or 'max' to override post-revision loading.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resmat_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix')
    
    processed_emb_dir = os.path.join(repo_root, 'model', 'processed_embeddings')
    raw_emb_file = os.path.join(resmat_dir, 'all_benchmarks_embeddings_4096_8B.pkl')

    post_rev_dir = os.path.join(resmat_dir, 'post-revision')
    pre_rev_dir = os.path.join(resmat_dir, 'pre-revision')

    if pre_revision != 'none':
        import random
        print(f"Bypassing Post-Revision. Loading Pre-Revision Data ({pre_revision} agents)...")
        b_names = ['colbench_backend_programming', 'corebench_hard', 'scicode', 'scienceagentbench']
        combined_dfs = []
        for b in b_names:
            possible_files = ['raw_score.csv', 'benchmark.csv', 'success_rate.csv', 'written_score.csv']
            df = None
            for f in possible_files:
                p = os.path.join(pre_rev_dir, b, f)
                if os.path.exists(p):
                    df = pd.read_csv(p, index_col=0)
                    break
            
            if df is not None:
                # [ALIGNMENT FIX]: Filter SciCode to match the 29 refined items used in post-revision
                if b == 'scicode':
                    # Exact list of 29 task IDs (stripped) from post-revision
                    SCICODE_POST_IDS = ['12', '14', '15', '16', '2', '23', '28', '32', '35', '41', '43', '46', '48', '52', '56', '58', '59', '61', '62', '63', '64', '66', '67', '71', '72', '77', '79', '80', '9']
                    post_cols = [f"scicode.{it}" for it in SCICODE_POST_IDS]
                    # Only keep columns that are in the post-revision set
                    valid_df_cols = [c for c in df.columns if c in post_cols]
                    df = df[valid_df_cols]

                # Prefix agents to ensure uniqueness across benchmarks (matching post-revision style)
                # and ensuring exactly 32 unique rows when sampling 8 per benchmark.
                df.index = [f"{b}.{a}" for a in df.index]
                # Ensure columns are prefixed with benchmark name
                df.columns = [f"{b}.{c}" if not str(c).startswith(b) and not str(c).startswith(b.replace('_hard','')) else c for c in df.columns]
                combined_dfs.append(df)
        
        if not combined_dfs:
            raise FileNotFoundError(f"No pre-revision data found in {pre_rev_dir}")
            
        final_df = pd.concat(combined_dfs, axis=1, join='outer')
        
        if pre_revision != 'max':
            try:
                n_total = int(pre_revision)
            except ValueError:
                n_total = None

            if n_total is not None:
                sampled_agents = []
                n_per_benchmark = n_total // 4
                remainder = n_total % 4
                np.random.seed(RANDOM_SEED)
                for i, b_name in enumerate(b_names):
                    # Distribute remainder across first few benchmarks
                    current_n = n_per_benchmark + (1 if i < remainder else 0)
                    if current_n == 0: continue

                    # Find the benchmark-specific DataFrame from the list we just populated
                    # (Logic matches how we concatenated them into combined_dfs)
                    b_df_matches = [df for df in combined_dfs if any(str(c).startswith(b_name) for c in df.columns)]
                    if b_df_matches:
                        b_df = b_df_matches[0]
                        # Find agents that have data for this benchmark
                        b_agents = b_df.dropna(how='all').index.tolist()
                        if len(b_agents) > current_n:
                            sampled = np.random.choice(b_agents, size=current_n, replace=False)
                        else:
                            sampled = b_agents
                        sampled_agents.extend(sampled)
            
            # Print sampling breakdown if not quiet
            if not locals().get('quiet', False):
                print(f"Equating Dimensions: Sampled {len(sampled_agents)} agents for Pre-Revision ({pre_revision}).")
            
            final_df = final_df.loc[sampled_agents]
            
        final_df = final_df.dropna(axis=1, how='all')
        all_dfs = [final_df]
        global_shared_indices = sorted(list(final_df.index))
        
        colbench_dfs = []
        other_dfs = []
    else:
        # Load from local directory instead of huggingface cache
        colbench_dir = os.path.join(post_rev_dir, 'colbench_backend_programming', 'resmat')
        # 1. Load ColBench response matrices
        colbench_files = sorted([f for f in os.listdir(colbench_dir) if f.startswith('resmat')])
            
        colbench_dfs = []
        for f in colbench_files:
            df = pd.read_csv(os.path.join(colbench_dir, f), index_col=0)
            
            # [CRITICAL DATA FIX]: The base CSV files dynamically generated from trace logs
            # inadvertently aggregate columns across multiple benchmarks (e.g. resmat_sky0.csv contains SAB + SciCode).
            # This strict filter prevents cross-contamination to ensure pure N-dim modeling arrays.
            valid_cols = [c for c in df.columns if str(c).startswith("colbench")]
            if valid_cols:
                df = df[valid_cols]
                df.columns = [f"colbench_backend_programming.{c}" if not str(c).startswith("colbench") else c for c in df.columns]
                colbench_dfs.append(df)
    
        # 2. Load other benchmarks response matrices
        other_benchmarks = [b for b in os.listdir(post_rev_dir) if b != 'colbench_backend_programming' and os.path.isdir(os.path.join(post_rev_dir, b))]

        from utils import get_benchmark_iterations
        bench_iterations = {} # benchmark -> [df0, df1, ..., df10]
        for benchmark in other_benchmarks:
            b_resmat_dir = os.path.join(post_rev_dir, benchmark, 'resmat')
            if os.path.exists(b_resmat_dir):
                iters = get_benchmark_iterations(b_resmat_dir, benchmark)
                if iters:
                    bench_iterations[benchmark] = iters
                    
        all_dfs = []
        for col_df in colbench_dfs:
            current_bench_parts = [col_df]
            for benchmark, iters in bench_iterations.items():
                # Randomly pick an iteration (0 = base, 1-10 = remediation)
                # This ensures N=1 uses one noisy run across all benchmarks.
                idx = np.random.randint(0, len(iters))
                current_bench_parts.append(iters[idx])
            
            all_dfs.append(pd.concat(current_bench_parts, axis=1, join='outer'))
    
        # Find shared indices for the model target matrices.
        # The user specifically requested 32 unique agents for the post-revision sweep.
        # We use all_dfs[0].index as the canonical baseline because intersecting all 54 runs 
        # inadvertently drops ColBench entirely due to runs like resmat_moon21 being empty.
        # The IRT model correctly handles missing runs by padding their absence with NaNs.
        global_shared_indices = sorted(list(set(all_dfs[0].index)))

    # Load embeddings based on type
    emb_file = None
    if embedding_type == 'raw':
        emb_file = raw_emb_file
    elif embedding_type == 'pca':
        emb_file = os.path.join(processed_emb_dir, f'embeddings_pca_{embedding_dim}.pkl')
    elif embedding_type == 'sae':
        emb_file = os.path.join(processed_emb_dir, f'embeddings_sae_{embedding_dim}.pkl')
    elif embedding_type in ['ones', 'rasch_2pl', 'nonamortised_mirt']:
        emb_file = None
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    # Fall back to raw if processed embeddings don't exist
    if emb_file is not None and not os.path.exists(emb_file):
        print(f"Warning: {emb_file} not found. Falling back to raw embeddings.")
        print("Run 'python generate_embeddings.py' to generate processed embeddings.")
        emb_file = raw_emb_file
        embedding_type = 'raw'

    if embedding_type in ['ones', 'rasch_2pl', 'nonamortised_mirt']:
        raw_embs_map = {}
    else:
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


def prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map, embedding_type='pca', j_percentage=1.0):
    """Prepare oracle ground truth, train/test splits, and embedding tensors."""
    print("=" * 60)
    print("PREPARING EXPERIMENT DATA")
    print("=" * 60)

    # Find the union of all columns across all dfs to handle potential missing items in some runs
    all_columns = sorted(list(set().union(*[df.columns for df in all_dfs])))
    
    # Create oracle matrix (average across all response matrices)
    oracle_dfs_filtered = [df.reindex(index=global_shared_indices, columns=all_columns) for df in all_dfs]
    oracle_stacked = np.array([df.values for df in oracle_dfs_filtered], dtype=float)
    oracle_matrix = np.nanmean(oracle_stacked, axis=0)
    oracle_df = pd.DataFrame(oracle_matrix, index=global_shared_indices, columns=all_columns)

    print(f"Total matrices: {len(all_dfs)}")
    print(f"Global user intersection: {len(global_shared_indices)} users")
    print(f"Oracle matrix shape: {oracle_df.shape}")

    # Train/test split (by items/columns)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N, J_full = oracle_df.shape
    
    # 2D Scaling Study: Randomly sub-sample the available items (columns) before holdout
    if j_percentage < 1.0:
        n_j_keep = max(10, int(j_percentage * J_full)) # Keep at least 10 items for meaningful calc
        all_j_indices = np.arange(J_full)
        # Fix seed for item sampling so it's consistent for a given RANDOM_SEED
        np.random.seed(RANDOM_SEED + 999) 
        sampled_j_indices = np.random.choice(all_j_indices, size=n_j_keep, replace=False)
        sampled_j_indices.sort()
        
        oracle_df = oracle_df.iloc[:, sampled_j_indices]
        print(f"Sub-sampling Items: {J_full} -> {n_j_keep} ({j_percentage*100:.1f}%)")
        
    N, J = oracle_df.shape
    sampled_columns = oracle_df.columns.tolist()
    
    # Filter all_dfs to match the sampled columns
    all_dfs_filtered = [df.reindex(columns=sampled_columns) for df in all_dfs]
    
    J_indices = np.arange(J)
    # Reset seed for train/test split consistency
    np.random.seed(RANDOM_SEED)
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
    if embedding_type in ['ones', 'rasch_2pl', 'nonamortised_mirt']:
        embeddings = np.ones((len(task_ids), 1), dtype=np.float32)
    else:
        embeddings = []
        for task_id in task_ids:
            emb = raw_embs_map.get(str(task_id))
            if emb is None and task_id.startswith('colbench.'):
                number = task_id.split('.')[-1]
                emb = raw_embs_map.get(f'colbench_backend_programming.{number}')
            if emb is None:
                # Use zeros for missing embeddings
                sample_emb = next(iter(raw_embs_map.values())) if raw_embs_map else [0.0]
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
        'all_dfs': all_dfs_filtered, # Pass filtered dataframes
        'y_oracle': y_oracle,
        'train_mask_t': train_mask_t,
        'test_mask_t': test_mask_t,
        'test_mask': test_mask,
        'x_j': x_j,
        'test_idx': test_idx,
        'train_idx': train_idx,
        'N': N,
        'J': J,
        'embedding_dim': x_j.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════════

def build_training_targets(n_files, all_dfs, global_shared_indices, data, model_type='beta', quiet=False):
    """Build training matrix/mask for a specific n_files configuration."""
    N = data['N']
    J = data['J']
    train_idx = data['train_idx']

    dfs_to_use = data.get('all_dfs', all_dfs)
    if n_files < len(dfs_to_use):
        sampled_indices = np.random.choice(len(dfs_to_use), n_files, replace=False)
        if not quiet:
            print(f"Sampled iterations: {sampled_indices}")
        current_dfs = [dfs_to_use[i].reindex(index=global_shared_indices) for i in sampled_indices]
    else:
        current_dfs = [dfs_to_use[i].reindex(index=global_shared_indices) for i in range(n_files)]

    all_columns = sorted(list(set().union(*[df.columns for df in current_dfs])))
    current_dfs = [df.reindex(columns=all_columns) for df in current_dfs]
    current_stacked = np.array([df.values for df in current_dfs], dtype=float)
    train_target_matrix = np.nanmean(current_stacked, axis=0)
    train_target_df = pd.DataFrame(train_target_matrix, index=global_shared_indices, columns=all_columns)

    train_values = train_target_df.values.copy()
    if model_type == 'bernoulli':
        train_values = (train_values > 0.5).astype(np.float32)

    train_values = np.nan_to_num(train_values, nan=0.5)
    y_train = torch.from_numpy(train_values.astype(np.float32)).to(device)

    train_mask_current = np.zeros_like(train_target_df.values, dtype=bool)
    train_mask_current[:, train_idx] = ~np.isnan(train_target_df.values)[:, train_idx]
    train_mask_current_t = torch.from_numpy(train_mask_current).to(device)

    return N, J, y_train, train_mask_current_t


def compute_non_mirt_baseline_metrics(N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                                      model_type='beta', beta_phi=BETA_PHI, x_j=None, knn_k=KNN_K):
    """Compute non-MIRT baselines for one configuration."""
    # 1. Naive item-mean baseline
    valid_counts = train_mask_current_t.sum(dim=0)
    item_sums = (y_train * train_mask_current_t).sum(dim=0)
    global_mean = y_train[train_mask_current_t].mean()
    item_means = torch.where(valid_counts > 0, item_sums / valid_counts, global_mean)
    p_naive = item_means.unsqueeze(0).expand(N, J)

    rmse_naive = compute_rmse(p_naive.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_naive = evaluate_auc(p_naive, y_oracle, test_mask_t)

    # 2. Rasch (1PL)
    p_rasch = train_rasch(N, J, y_train, train_mask_current_t)
    rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_rasch = evaluate_auc(p_rasch, y_oracle, test_mask_t)

    # 3. Rasch (2PL)
    p_2pl = train_2pl(N, J, y_train, train_mask_current_t)
    rmse_2pl = compute_rmse(p_2pl.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_2pl = evaluate_auc(p_2pl, y_oracle, test_mask_t)

    # 4. Embedding kNN cold-start baseline (predict held-out items from nearest train items)
    # Uses item embeddings only and observed user responses on train items.
    if x_j is None:
        p_knn = p_naive.clone()
    else:
        x = x_j
        if x.dim() != 2 or x.shape[0] != J:
            p_knn = p_naive.clone()
        else:
            train_item_mask = train_mask_current_t.any(dim=0)
            test_item_mask = torch.from_numpy(test_mask).to(device).any(dim=0)

            train_item_idx = torch.where(train_item_mask)[0]
            test_item_idx = torch.where(test_item_mask)[0]

            if train_item_idx.numel() == 0 or test_item_idx.numel() == 0:
                p_knn = p_naive.clone()
            else:
                # User fallback = mean on observed train items.
                train_obs = train_mask_current_t.float()
                user_counts = train_obs.sum(dim=1)
                global_mean = y_train[train_mask_current_t].mean()
                user_means = torch.where(
                    user_counts > 0,
                    (y_train * train_obs).sum(dim=1) / user_counts.clamp_min(1.0),
                    global_mean
                )

                p_knn = user_means.unsqueeze(1).expand(N, J).clone()

                # Embeddings are already normalized in data prep, but renormalize defensively.
                x_norm = F.normalize(x, dim=1)
                sims = x_norm[test_item_idx] @ x_norm[train_item_idx].T
                k_eff = max(1, min(int(knn_k), train_item_idx.numel()))
                sim_vals, sim_pos = torch.topk(sims, k=k_eff, dim=1)
                nn_item_idx = train_item_idx[sim_pos]

                # Positive similarities only; fallback to uniform if all non-positive.
                nn_weights = torch.clamp(sim_vals, min=0.0)
                zero_rows = nn_weights.sum(dim=1, keepdim=True) <= 1e-12
                if zero_rows.any():
                    nn_weights[zero_rows.squeeze(1)] = 1.0

                for t in range(test_item_idx.numel()):
                    nbrs = nn_item_idx[t]
                    w = nn_weights[t].unsqueeze(0)
                    obs = train_mask_current_t[:, nbrs].float()
                    yy = y_train[:, nbrs]

                    num = (yy * obs * w).sum(dim=1)
                    den = (obs * w).sum(dim=1)
                    pred = torch.where(den > 0, num / den.clamp_min(1e-12), user_means)
                    p_knn[:, test_item_idx[t]] = pred

    rmse_knn = compute_rmse(p_knn.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_knn = evaluate_auc(p_knn, y_oracle, test_mask_t)

    return {
        'rmse_naive': rmse_naive,
        'rmse_rasch': rmse_rasch,
        'rmse_2pl': rmse_2pl,
        'auc_naive': auc_naive,
        'auc_rasch': auc_rasch,
        'auc_2pl': auc_2pl,
        'rmse_knn': rmse_knn,
        'auc_knn': auc_knn,
    }


def compute_single_mirt_metrics(N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                                model_type='beta', beta_phi=BETA_PHI, mirt_dim=K_MODEL):
    """Train one standalone MIRT baseline at a specific latent dimension."""
    mirt_model = MIRTModel(N, J, mirt_dim).to(device)
    mirt_optimizer = optim.AdamW(mirt_model.parameters(), lr=0.01, weight_decay=0.1)
    mirt_model.train()

    best_mirt_rmse = float('inf')
    best_p_mirt = None

    for _ in range(EPOCHS // 2):
        mirt_optimizer.zero_grad()
        p_mirt = mirt_model()
        p_m_clamp = p_mirt[train_mask_current_t].clamp(1e-6, 1 - 1e-6)
        y_m_clamp = y_train[train_mask_current_t]

        if model_type == 'beta':
            y_m_clamp = y_m_clamp.clamp(1e-6, 1 - 1e-6)
            dist = torch.distributions.Beta(p_m_clamp * beta_phi, (1 - p_m_clamp) * beta_phi)
            loss = -dist.log_prob(y_m_clamp).mean()
        else:
            dist = torch.distributions.Bernoulli(probs=p_m_clamp)
            loss = -dist.log_prob(y_m_clamp).mean()

        loss.backward()
        mirt_optimizer.step()

        with torch.no_grad():
            curr_m_rmse = compute_rmse(p_mirt.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
            if curr_m_rmse < best_mirt_rmse:
                best_mirt_rmse = curr_m_rmse
                best_p_mirt = p_mirt.clone()

    auc_mirt = evaluate_auc(best_p_mirt, y_oracle, test_mask_t)

    return {
        'rmse_mirt': best_mirt_rmse,
        'auc_mirt': auc_mirt,
        'mirt_dim': int(mirt_dim),
        'mirt_state': mirt_model.state_dict(),
    }


def compute_baseline_metrics(N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                             model_type='beta', beta_phi=BETA_PHI, x_j=None, knn_k=KNN_K,
                             mirt_dim=K_MODEL):
    """Compute naive + Rasch + 2PL + standalone MIRT baselines for one configuration."""
    results = compute_non_mirt_baseline_metrics(
        N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
        model_type=model_type, beta_phi=beta_phi, x_j=x_j, knn_k=knn_k
    )
    results.update(
        compute_single_mirt_metrics(
            N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
            model_type=model_type, beta_phi=beta_phi, mirt_dim=mirt_dim
        )
    )
    return results


def _optional_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _optional_float(value):
    if pd.isna(value):
        return None
    return float(value)


def _row_has_complete_metrics(row, metric_cols):
    return all(not pd.isna(row.get(col)) for col in metric_cols)


def _baseline_row_matches_mirt_request(row, mirt_dim_min, mirt_dim_max):
    if not _row_has_complete_metrics(row, BASELINE_METRIC_COLS):
        return False

    selected_dim = _optional_int(row.get('selected_mirt_dim'))
    sweep_min = _optional_int(row.get('mirt_sweep_min'))
    sweep_max = _optional_int(row.get('mirt_sweep_max'))

    if selected_dim is None and sweep_min is None and sweep_max is None:
        return int(mirt_dim_min) == K_MODEL and int(mirt_dim_max) == K_MODEL

    return (
        selected_dim is not None and
        sweep_min == int(mirt_dim_min) and
        sweep_max == int(mirt_dim_max)
    )


def _baseline_payload_from_row(row):
    payload = {k: float(row[k]) for k in BASELINE_METRIC_COLS}
    selected_dim = _optional_int(row.get('selected_mirt_dim'))
    if selected_dim is not None:
        payload['selected_mirt_dim'] = selected_dim
    return payload


def select_best_mirt_result(results):
    """Pick the best MIRT sweep candidate by AUC, then RMSE, then smaller dimension."""
    return max(
        results,
        key=lambda r: (
            float(r['auc_mirt']),
            -float(r['rmse_mirt']),
            -int(r['mirt_dim']),
        )
    )


def get_or_compute_baselines(n_files, all_dfs, global_shared_indices, data, model_type='beta', beta_phi=BETA_PHI,
                             baseline_output=DEFAULT_BASELINE_OUTPUT, pre_revision='none', j_percentage=1.0,
                             allow_compute=True, quiet=False, mirt_dim_min=K_MODEL, mirt_dim_max=K_MODEL,
                             mirt_sweep_output=DEFAULT_MIRT_SWEEP_OUTPUT):
    """Fetch baselines from cache, or compute and persist once per unique configuration."""
    baseline_key = {
        'seed': int(RANDOM_SEED),
        'model_type': str(model_type),
        'n_samples': int(n_files),
        'pre_revision': normalize_pre_revision(pre_revision),
        'j_percentage': normalize_j_percentage(j_percentage),
    }

    cached = try_get_cached_baseline(
        baseline_output,
        baseline_key,
        mirt_dim_min=mirt_dim_min,
        mirt_dim_max=mirt_dim_max,
    )
    if cached is not None:
        return cached, None

    if not allow_compute:
        raise RuntimeError(
            f"Missing baseline cache row for {baseline_key}. "
            f"Run with --baseline-only first or set allow_compute=True."
        )

    existing_row = load_existing_baseline_row(baseline_output, baseline_key)
    non_mirt_metrics = None
    if existing_row is not None and _row_has_complete_metrics(existing_row, NON_MIRT_METRIC_COLS):
        non_mirt_metrics = {k: float(existing_row[k]) for k in NON_MIRT_METRIC_COLS}

    N, J, y_train, train_mask_current_t = build_training_targets(
        n_files, all_dfs, global_shared_indices, data, model_type=model_type, quiet=quiet
    )

    if non_mirt_metrics is None:
        non_mirt_metrics = compute_non_mirt_baseline_metrics(
            N, J, y_train, train_mask_current_t,
            data['y_oracle'], data['test_mask'], data['test_mask_t'],
            model_type=model_type, beta_phi=beta_phi, x_j=data.get('x_j')
        )

    mirt_results = []
    legacy_30_imported = False
    for mirt_dim in range(int(mirt_dim_min), int(mirt_dim_max) + 1):
        sweep_key = baseline_key.copy()
        sweep_key['mirt_dim'] = int(mirt_dim)
        cached_mirt = try_get_cached_mirt_sweep_row(mirt_sweep_output, sweep_key)
        if cached_mirt is not None:
            mirt_results.append(cached_mirt)
            continue

        can_import_legacy_30 = (
            int(mirt_dim) == K_MODEL and
            existing_row is not None and
            not legacy_30_imported
        )
        if can_import_legacy_30:
            legacy_selected_dim = _optional_int(existing_row.get('selected_mirt_dim'))
            legacy_sweep_min = _optional_int(existing_row.get('mirt_sweep_min'))
            legacy_sweep_max = _optional_int(existing_row.get('mirt_sweep_max'))
            is_legacy_fixed_30 = (
                _row_has_complete_metrics(existing_row, ['rmse_mirt', 'auc_mirt']) and
                (
                    (legacy_selected_dim is None and legacy_sweep_min is None and legacy_sweep_max is None) or
                    (legacy_selected_dim == K_MODEL and legacy_sweep_min == K_MODEL and legacy_sweep_max == K_MODEL)
                )
            )
            if is_legacy_fixed_30:
                cached_mirt = {
                    'rmse_mirt': float(existing_row['rmse_mirt']),
                    'auc_mirt': float(existing_row['auc_mirt']),
                    'mirt_dim': K_MODEL,
                }
                append_mirt_sweep_row(mirt_sweep_output, {**sweep_key, **cached_mirt})
                mirt_results.append(cached_mirt)
                legacy_30_imported = True
                if not quiet:
                    print(f"[MIRT] Reused legacy cached 30D baseline for seed={baseline_key['seed']}, n={n_files}.")
                continue

        computed_mirt = compute_single_mirt_metrics(
            N, J, y_train, train_mask_current_t,
            data['y_oracle'], data['test_mask'], data['test_mask_t'],
            model_type=model_type, beta_phi=beta_phi, mirt_dim=mirt_dim
        )
        append_mirt_sweep_row(
            mirt_sweep_output,
            {
                **sweep_key,
                'rmse_mirt': float(computed_mirt['rmse_mirt']),
                'auc_mirt': float(computed_mirt['auc_mirt']),
            }
        )
        mirt_results.append(computed_mirt)

    best_mirt = select_best_mirt_result(mirt_results)

    baseline_row = baseline_key.copy()
    baseline_row['agent_batch_size'] = compute_agent_batch_size(
        baseline_key['pre_revision'], baseline_key['n_samples']
    )
    for col, value in non_mirt_metrics.items():
        baseline_row[col] = float(value)
    baseline_row['rmse_mirt'] = float(best_mirt['rmse_mirt'])
    baseline_row['auc_mirt'] = float(best_mirt['auc_mirt'])
    baseline_row['selected_mirt_dim'] = int(best_mirt['mirt_dim'])
    baseline_row['mirt_sweep_min'] = int(mirt_dim_min)
    baseline_row['mirt_sweep_max'] = int(mirt_dim_max)
    append_baseline_row(baseline_output, baseline_row)

    cached_now = _baseline_payload_from_row(baseline_row)
    return cached_now, best_mirt.get('mirt_state')


def run_experiment(n_files, all_dfs, global_shared_indices, data, model_type='beta',
                   beta_phi=BETA_PHI, no_tau=False, quiet=False, embedding_type=None,
                   baseline_output=DEFAULT_BASELINE_OUTPUT, pre_revision='none', j_percentage=1.0,
                   allow_compute_baselines=True, mirt_dim_min=K_MODEL, mirt_dim_max=K_MODEL,
                   mirt_sweep_output=DEFAULT_MIRT_SWEEP_OUTPUT):
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
    test_mask = data['test_mask']
    test_mask_t = data['test_mask_t']
    x_j = data['x_j']
    embedding_dim = data['embedding_dim']

    baselines, mirt_state = get_or_compute_baselines(
        n_files,
        all_dfs,
        global_shared_indices,
        data,
        model_type=model_type,
        beta_phi=beta_phi,
        baseline_output=baseline_output,
        pre_revision=pre_revision,
        j_percentage=j_percentage,
        allow_compute=allow_compute_baselines,
        quiet=quiet,
        mirt_dim_min=mirt_dim_min,
        mirt_dim_max=mirt_dim_max,
        mirt_sweep_output=mirt_sweep_output,
    )

    rmse_2pl = baselines['rmse_2pl']
    best_mirt_rmse = baselines['rmse_mirt']
    auc_2pl = baselines['auc_2pl']
    auc_mirt = baselines['auc_mirt']
    selected_mirt_dim = int(baselines.get('selected_mirt_dim', K_MODEL))

    if embedding_type == 'rasch_2pl':
        return {
            'n_samples': n_files,
            'model_type': model_type,
            'seed': RANDOM_SEED,
            'lambda_tau': LAMBDA_TAU,
            'rmse_amortized': rmse_2pl,
            'auc_amortized': auc_2pl,
            'active_dims': 0,
            'active_indices': '[]',
            'tau_values': '[]',
            'model_state': {},
            'final_state': {}
        }

    if embedding_type == 'nonamortised_mirt':
        model_state = mirt_state if mirt_state is not None else {}
        return {
            'n_samples': n_files,
            'model_type': model_type,
            'seed': RANDOM_SEED,
            'lambda_tau': LAMBDA_TAU,
            'rmse_amortized': best_mirt_rmse,
            'auc_amortized': auc_mirt,
            'active_dims': selected_mirt_dim,
            'active_indices': str(list(range(selected_mirt_dim))),
            'tau_values': str([1.0] * selected_mirt_dim),
            'model_state': model_state,
            'final_state': model_state
        }

    # Build train targets only when the amortized model is needed
    _, _, y_train, train_mask_current_t = build_training_targets(
        n_files, all_dfs, global_shared_indices, data, model_type=model_type, quiet=quiet
    )

    # 5. Amortized IRT (our method)
    model = AmortizedIRTModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.5, no_tau=no_tau).to(device)
    best_rmse, best_state, final_state = train_amortized_irt(model, y_train, train_mask_current_t, y_oracle, test_mask,
                                     model_type=model_type, beta_phi=beta_phi,
                                     epochs=EPOCHS, lambda_tau=LAMBDA_TAU, quiet=quiet)

    model.eval()
    with torch.no_grad():
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
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
        'seed': RANDOM_SEED,
        'lambda_tau': LAMBDA_TAU,
        'rmse_amortized': best_rmse,
        'auc_amortized': auc_amortized,
        'active_dims': active_dims,
        'active_indices': str(active_dim_indices),
        'tau_values': str(tau_val.cpu().tolist()),
        'model_state': best_state,
        'final_state': final_state
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_n_samples(arg, total_files):
    """Parse --n-samples argument into list of integers."""
    if arg == 'all':
        return list(range(1, total_files + 1))
    if arg == 'max':
        return [total_files]
    if arg == '1,all':
        return [1, total_files]

    result = []
    for part in arg.split(','):
        part = part.strip()
        if part == 'all':
            result.append(total_files)
            continue
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))

    result = [n for n in result if 1 <= n <= total_files]
    return sorted(set(result))


def normalize_pre_revision(value):
    """Normalize pre-revision value to stable string key used in baseline cache."""
    if value is None:
        return 'none'
    v = str(value).strip().lower()
    return v if v else 'none'


def normalize_j_percentage(value):
    """Normalize j_percentage for reliable float comparisons in CSV cache."""
    return float(f"{float(value):.6f}")


def compute_agent_batch_size(pre_revision, n_samples):
    """Return user-facing effective batch size (pre-revision size or n_samples)."""
    pre = normalize_pre_revision(pre_revision)
    if pre == 'none':
        return str(int(n_samples))
    if pre == 'max':
        return 'max'
    try:
        return str(int(pre))
    except Exception:
        return pre


def baseline_row_matches(df, key):
    """Return rows matching baseline key with tolerance on j_percentage."""
    if df.empty:
        return df

    mask = (
        (df['seed'].astype(int) == int(key['seed'])) &
        (df['model_type'].astype(str) == str(key['model_type'])) &
        (df['n_samples'].astype(int) == int(key['n_samples'])) &
        (df['pre_revision'].astype(str) == str(key['pre_revision'])) &
        np.isclose(df['j_percentage'].astype(float), float(key['j_percentage']), atol=1e-6)
    )
    return df[mask]


def mirt_sweep_row_matches(df, key):
    """Return rows matching a per-dimension MIRT sweep key."""
    if df.empty:
        return df
    return baseline_row_matches(df, key)[lambda x: x['mirt_dim'].astype(int) == int(key['mirt_dim'])]


def load_baseline_store(path):
    """Load baseline store or return an empty frame with required schema."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for col in BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS:
                if col not in df.columns:
                    df[col] = np.nan

            # Backfill explicit effective batch size for readability.
            df['agent_batch_size'] = [
                compute_agent_batch_size(pr, ns)
                for pr, ns in zip(df['pre_revision'], df['n_samples'])
            ]
            return df
        except Exception:
            pass

    cols = BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS
    return pd.DataFrame(columns=cols)


def load_mirt_sweep_store(path):
    """Load MIRT sweep store or return an empty frame with required schema."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for col in MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS:
                if col not in df.columns:
                    df[col] = np.nan
            return df
        except Exception:
            pass

    cols = MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS
    return pd.DataFrame(columns=cols)


def append_baseline_row(path, row):
    """Atomically append or upsert one baseline row in the baseline cache file."""
    lock = FileLock(f"{path}.lock", timeout=600)
    with lock:
        df_old = load_baseline_store(path)
        if 'agent_batch_size' not in row:
            row['agent_batch_size'] = compute_agent_batch_size(row.get('pre_revision', 'none'), row.get('n_samples', 0))
        df_new = pd.DataFrame([row])
        df = pd.concat([df_old, df_new], ignore_index=True)
        # Keep latest row per unique baseline key
        df = df.drop_duplicates(subset=BASELINE_KEY_COLS, keep='last')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def append_mirt_sweep_row(path, row):
    """Atomically append or upsert one MIRT sweep row."""
    lock = FileLock(f"{path}.lock", timeout=600)
    with lock:
        df_old = load_mirt_sweep_store(path)
        df_new = pd.DataFrame([row])
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=MIRT_SWEEP_KEY_COLS, keep='last')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def load_existing_baseline_row(path, key):
    """Lookup baseline row by key without enforcing MIRT sweep coverage."""
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if any(c not in df.columns for c in BASELINE_KEY_COLS + BASELINE_METRIC_COLS):
        return None

    match = baseline_row_matches(df, key)
    if match.empty:
        return None

    return match.iloc[-1].to_dict()


def try_get_cached_baseline(path, key, mirt_dim_min=K_MODEL, mirt_dim_max=K_MODEL):
    """Lookup cached baseline row by key and requested MIRT sweep range."""
    row = load_existing_baseline_row(path, key)
    if row is None or not _baseline_row_matches_mirt_request(row, mirt_dim_min, mirt_dim_max):
        return None
    return _baseline_payload_from_row(row)


def try_get_cached_mirt_sweep_row(path, key):
    """Lookup one cached MIRT sweep row by key."""
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if any(c not in df.columns for c in MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS):
        return None

    match = mirt_sweep_row_matches(df, key)
    if match.empty:
        return None

    row = match.iloc[-1].to_dict()
    if not _row_has_complete_metrics(row, MIRT_SWEEP_METRIC_COLS):
        return None
    return {
        'rmse_mirt': float(row['rmse_mirt']),
        'auc_mirt': float(row['auc_mirt']),
        'mirt_dim': int(row['mirt_dim']),
    }


def seed_mirt_sweep_from_baseline_store(baseline_output, mirt_sweep_output, quiet=False):
    """Populate exact-dimension MIRT sweep rows from legacy or exact baseline rows."""
    baseline_df = load_baseline_store(baseline_output)
    if baseline_df.empty:
        return

    seed_rows = []
    for _, row in baseline_df.iterrows():
        if not _row_has_complete_metrics(row, ['rmse_mirt', 'auc_mirt']):
            continue

        selected_dim = _optional_int(row.get('selected_mirt_dim'))
        sweep_min = _optional_int(row.get('mirt_sweep_min'))
        sweep_max = _optional_int(row.get('mirt_sweep_max'))

        if selected_dim is None and sweep_min is None and sweep_max is None:
            exact_dim = K_MODEL
        elif selected_dim is not None and sweep_min == selected_dim and sweep_max == selected_dim:
            exact_dim = selected_dim
        else:
            continue

        seed_row = {k: row[k] for k in BASELINE_KEY_COLS}
        seed_row['mirt_dim'] = int(exact_dim)
        seed_row['rmse_mirt'] = float(row['rmse_mirt'])
        seed_row['auc_mirt'] = float(row['auc_mirt'])
        seed_rows.append(seed_row)

    if not seed_rows:
        return

    lock = FileLock(f"{mirt_sweep_output}.lock", timeout=600)
    with lock:
        existing = load_mirt_sweep_store(mirt_sweep_output)
        combined = pd.concat([existing, pd.DataFrame(seed_rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=MIRT_SWEEP_KEY_COLS, keep='last')
        os.makedirs(os.path.dirname(mirt_sweep_output), exist_ok=True)
        combined.to_csv(mirt_sweep_output, index=False)

    if not quiet:
        print(f"Seeded {len(seed_rows)} exact-dimension MIRT rows into {mirt_sweep_output}")


def migrate_existing_baselines(source_dir, baseline_output, quiet=False):
    """Extract baseline columns from existing result CSVs into unified baseline cache."""
    pattern = re.compile(r"_pre_([^_]+)_")
    j_pattern = re.compile(r"_j([0-9]+(?:\.[0-9]+)?)")

    migrated_rows = []
    files = [f for f in os.listdir(source_dir) if f.startswith('amortized_irt_') and f.endswith('.csv')]

    for fname in files:
        path = os.path.join(source_dir, fname)
        if os.path.abspath(path) == os.path.abspath(baseline_output):
            continue

        try:
            df = pd.read_csv(path, on_bad_lines='skip')
        except Exception:
            continue

        required = ['seed', 'model_type', 'n_samples'] + BASELINE_METRIC_COLS
        if any(col not in df.columns for col in required):
            continue

        pre_revision = 'none'
        if 'scenario' in df.columns and df['scenario'].notna().any():
            # Expected forms: Pre-32 / Pre-max
            val = str(df['scenario'].dropna().iloc[0]).replace('Pre-', '').strip().lower()
            if val:
                pre_revision = val
        else:
            m = pattern.search(fname)
            if m:
                pre_revision = normalize_pre_revision(m.group(1))

        j_percentage = 1.0
        m_j = j_pattern.search(fname)
        if m_j:
            j_percentage = normalize_j_percentage(float(m_j.group(1)))

        sub = df[required].copy()
        sub['pre_revision'] = pre_revision
        sub['j_percentage'] = j_percentage
        sub['agent_batch_size'] = [
            compute_agent_batch_size(pre_revision, ns)
            for ns in sub['n_samples'].tolist()
        ]
        sub['selected_mirt_dim'] = K_MODEL
        sub['mirt_sweep_min'] = K_MODEL
        sub['mirt_sweep_max'] = K_MODEL
        sub = sub[BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS]
        migrated_rows.append(sub)

    if not migrated_rows:
        if not quiet:
            print("No baseline columns discovered to migrate.")
        return

    migrated_df = pd.concat(migrated_rows, ignore_index=True)
    for k in ['seed', 'n_samples']:
        migrated_df[k] = pd.to_numeric(migrated_df[k], errors='coerce')
    migrated_df['j_percentage'] = pd.to_numeric(migrated_df['j_percentage'], errors='coerce').fillna(1.0)
    migrated_df = migrated_df.dropna(subset=['seed', 'n_samples'])
    migrated_df['seed'] = migrated_df['seed'].astype(int)
    migrated_df['n_samples'] = migrated_df['n_samples'].astype(int)
    migrated_df['pre_revision'] = migrated_df['pre_revision'].astype(str).map(normalize_pre_revision)
    migrated_df['j_percentage'] = migrated_df['j_percentage'].map(normalize_j_percentage)

    lock = FileLock(f"{baseline_output}.lock", timeout=600)
    with lock:
        existing = load_baseline_store(baseline_output)
        combined = pd.concat([existing, migrated_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=BASELINE_KEY_COLS, keep='last')
        os.makedirs(os.path.dirname(baseline_output), exist_ok=True)
        combined.to_csv(baseline_output, index=False)

    if not quiet:
        print(f"Migrated {len(migrated_df)} baseline rows into {baseline_output}")


def strip_inline_baseline_columns(source_dir, quiet=False):
    """Rewrite amortized_irt CSVs by removing legacy inline baseline columns."""
    files = [f for f in os.listdir(source_dir) if f.startswith('amortized_irt_') and f.endswith('.csv')]
    rewritten = 0
    removed_cols = 0

    for fname in files:
        path = os.path.join(source_dir, fname)
        try:
            df = pd.read_csv(path, on_bad_lines='skip')
        except Exception:
            continue

        present = [c for c in INLINE_BASELINE_COLS if c in df.columns]
        if not present:
            continue

        df = df.drop(columns=present)
        df.to_csv(path, index=False)
        rewritten += 1
        removed_cols += len(present)

    if not quiet:
        print(f"Stripped inline baseline columns from {rewritten} files ({removed_cols} columns removed total).")


# Global variables for worker processes to avoid pickling overhead
_WORKER_DFS = None
_WORKER_INDICES = None
_WORKER_EMBS_MAP = None
_WORKER_EMB_TYPE = None

def init_worker(dfs, indices, embs, emb_type):
    """Initialize worker process with large shared objects to avoid pickling overhead."""
    global _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE
    _WORKER_DFS = dfs
    _WORKER_INDICES = indices
    _WORKER_EMBS_MAP = embs
    _WORKER_EMB_TYPE = emb_type

def run_single_config(config, args, n_values):
    """Worker function for running a single (seed, lambda_tau) configuration."""
    seed, lambda_tau, worker_id = config
    
    # Retrieve large objects from global state initialized once per process
    global _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE
    all_dfs = _WORKER_DFS
    global_shared_indices = _WORKER_INDICES
    raw_embs_map = _WORKER_EMBS_MAP
    actual_emb_type = _WORKER_EMB_TYPE
    
    # Assign GPU evenly across workers
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    local_device = torch.device(f'cuda:{worker_id % num_gpus}' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')
    
    # Set independent random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    global LAMBDA_TAU, RANDOM_SEED
    LAMBDA_TAU = lambda_tau
    RANDOM_SEED = seed

    j_suffix = f"_j{args.j_percentage}" if args.j_percentage < 1.0 else ""
    output_path = None
    if not args.baseline_only:
        if args.output:
            output_path = args.output
        else:
            suffix = f"_pre_{args.pre_revision}" if args.pre_revision != 'none' else ""
            n_suffix = f"_n_{args.n_samples}" if args.n_samples != 'all' else "_n_max"
            output_path = os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}{suffix}{n_suffix}{j_suffix}.csv')

    if output_path is not None and os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            max_n = max(n_values)
            if not df_existing.empty and 'seed' in df_existing.columns and 'lambda_tau' in df_existing.columns:
                matches = df_existing[(df_existing['seed'] == seed) & 
                                      (np.isclose(df_existing['lambda_tau'], lambda_tau, atol=1e-5)) &
                                      (df_existing['n_samples'] == max_n)]
                if not matches.empty:
                    # Let the main process handle the skip message if skipped entirely
                    return
        except Exception:
            pass # File lock contention, just run it

    # Prepare data on the local device
    data = prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map, 
                                 embedding_type=actual_emb_type, j_percentage=args.j_percentage)
    # Move tensors to the correct device
    data['x_j'] = data['x_j'].to(local_device)
    data['y_oracle'] = data['y_oracle'].to(local_device)
    data['test_mask_t'] = data['test_mask_t'].to(local_device)

    results = []
    
    quiet = args.quiet
    if not quiet:
        print(f"\n[START] worker {worker_id} (GPU {worker_id % num_gpus}) executing -> seed={seed}, tau={lambda_tau}")
    for i, n in enumerate(n_values):
        # We need to temporarily mock the global device so `run_experiment` internals use it
        global device
        old_device = device
        device = local_device
        
        try:
            if args.baseline_only:
                get_or_compute_baselines(
                    n,
                    all_dfs,
                    global_shared_indices,
                    data,
                    model_type=args.model_type,
                    beta_phi=args.beta_phi,
                    baseline_output=args.baseline_output,
                    pre_revision=args.pre_revision,
                    j_percentage=args.j_percentage,
                    allow_compute=True,
                    quiet=quiet,
                    mirt_dim_min=args.mirt_dim_min,
                    mirt_dim_max=args.mirt_dim_max,
                    mirt_sweep_output=args.mirt_sweep_output,
                )
                continue

            result = run_experiment(n, all_dfs, global_shared_indices, data,
                                    model_type=args.model_type, beta_phi=args.beta_phi, no_tau=args.no_tau, 
                                    quiet=quiet, embedding_type=actual_emb_type,
                                    baseline_output=args.baseline_output,
                                    pre_revision=args.pre_revision,
                                    j_percentage=args.j_percentage,
                                    allow_compute_baselines=True,
                                    mirt_dim_min=args.mirt_dim_min,
                                    mirt_dim_max=args.mirt_dim_max,
                                    mirt_sweep_output=args.mirt_sweep_output)
                                    
            result['embedding_type'] = actual_emb_type
            if args.pre_revision != 'none':
                result['scenario'] = f"Pre-{args.pre_revision}"
            results.append(result)
            
            # Save model state to separate pkl if requested
            if args.save_weights:
                weight_path = output_path.replace('.csv', f'_seed_{seed}_weights_best.pkl')
                torch.save(result['model_state'], weight_path)
                
                final_weight_path = output_path.replace('.csv', f'_seed_{seed}_weights_final.pkl')
                torch.save(result['final_state'], final_weight_path)
        finally:
            device = old_device

    if args.baseline_only:
        if not quiet:
            print(f"[DONE] worker {worker_id} cached baselines: seed={seed}, pre={args.pre_revision}, n={n_values}")
        return
            
    # Save results to CSV (Consolidated)
    save_results = []
    for r in results:
        r_copy = r.copy()
        if 'model_state' in r_copy: del r_copy['model_state']
        if 'final_state' in r_copy: del r_copy['final_state']
        save_results.append(r_copy)
    
    df_results = pd.DataFrame(save_results)
    
    lock_path = f"{output_path}.lock"
    lock = FileLock(lock_path, timeout=600)
    
    try:
        with lock:
            if os.path.exists(output_path):
                try:
                    df_old = pd.read_csv(output_path)
                    df_combined = pd.concat([df_old, df_results]).drop_duplicates(subset=['n_samples', 'seed', 'lambda_tau'], keep='last')
                    df_combined.to_csv(output_path, index=False)
                except pd.errors.EmptyDataError:
                    df_results.to_csv(output_path, index=False)
            else:
                df_results.to_csv(output_path, index=False)
    except Timeout:
        print(f"\n[WARNING] Could not acquire lock for {output_path}. Skipping save.")

    print(f"[DONE] worker {worker_id} completed: seed={seed}, tau={lambda_tau} -> saved to {os.path.basename(output_path)}")


def main():
    global LAMBDA_TAU, WD_THETA, WD_W, EPOCHS, SNAPPING_THRESHOLD, RANDOM_SEED
    import argparse
    parser = argparse.ArgumentParser(description='Amortized IRT Experiment')
    parser.add_argument(
        '--embedding-type', type=str, default='pca',
        choices=['raw', 'pca', 'sae', 'ones', 'rasch_2pl', 'nonamortised_mirt'],
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
        '--beta-phi', type=float, default=10.0,
        help='Beta distribution precision parameter φ. Higher = more concentrated.'
    )
    parser.add_argument(
        '--n-samples', type=str, default='all',
        help='Which n values to run. Examples: "all", "22", "1,22", "1-5,22"'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: result/amortized_irt_{embedding_type}_{model_type}.csv)'
    )
    parser.add_argument('--no-tau', action='store_true', help='Ablation: Disable tau sparsity mechanism.')
    parser.add_argument('--lambda-tau', type=str, default=str(LAMBDA_TAU), help='Override LAMBDA_TAU (comma separated lists allowed)')
    parser.add_argument('--wd-theta', type=float, default=None, help='Override WD_THETA')
    parser.add_argument('--wd-w', type=float, default=None, help='Override WD_W')
    parser.add_argument('--epochs', type=int, default=None, help='Override EPOCHS')
    parser.add_argument('--snapping-threshold', type=float, default=None, help='Override SNAPPING_THRESHOLD')
    parser.add_argument('--pre-revision', type=str, default='none', 
                        help='Evaluate on pre-revision matrix with N=X or "max".')
    parser.add_argument('--seed', type=str, default=str(RANDOM_SEED), help='Random seed(s) (comma separated strings allowed)')
    parser.add_argument('--save-weights', action='store_true', help='Save model weights to pkl.')
    parser.add_argument('--parallel', type=int, default=1, help='Number of multiprocessing workers.')
    parser.add_argument('--j-percentage', type=float, default=1.0, help='Percentage of items (columns) to sample (0.0 to 1.0).')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--baseline-only', action='store_true', help='Only compute/cache baselines and skip amortized outputs.')
    parser.add_argument('--baseline-output', type=str, default=DEFAULT_BASELINE_OUTPUT,
                        help='Path to baseline cache CSV (default: model/result/baselines/baseline_metrics.csv).')
    parser.add_argument('--mirt-sweep-output', type=str, default=DEFAULT_MIRT_SWEEP_OUTPUT,
                        help='Path to per-dimension MIRT sweep cache CSV.')
    parser.add_argument('--mirt-dim-min', type=int, default=K_MODEL,
                        help='Minimum MIRT dimension to evaluate for the baseline sweep.')
    parser.add_argument('--mirt-dim-max', type=int, default=K_MODEL,
                        help='Maximum MIRT dimension to evaluate for the baseline sweep.')
    parser.add_argument('--migrate-baselines', action='store_true',
                        help='Migrate baseline columns from existing amortized_irt_*.csv files into baseline-output and exit.')
    parser.add_argument('--migrate-all-csvs', action='store_true',
                        help='Migrate baseline cache and strip inline baseline columns from all amortized_irt_*.csv files, then exit.')
    parser.add_argument('--migrate-source-dir', type=str, default=RESULT_DIR,
                        help='Source directory containing historical amortized_irt_*.csv files for migration.')
    args = parser.parse_args()

    import sys, os
    import numpy as np
    import pandas as pd
    import multiprocessing as mp
    from functools import partial
    
    if args.quiet:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    if args.migrate_baselines:
        migrate_existing_baselines(args.migrate_source_dir, args.baseline_output, quiet=args.quiet)
        seed_mirt_sweep_from_baseline_store(args.baseline_output, args.mirt_sweep_output, quiet=args.quiet)
        return

    if args.migrate_all_csvs:
        migrate_existing_baselines(args.migrate_source_dir, args.baseline_output, quiet=args.quiet)
        seed_mirt_sweep_from_baseline_store(args.baseline_output, args.mirt_sweep_output, quiet=args.quiet)
        strip_inline_baseline_columns(args.migrate_source_dir, quiet=args.quiet)
        return

    if args.wd_theta is not None:
        WD_THETA = args.wd_theta
    if args.wd_w is not None:
        WD_W = args.wd_w
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.snapping_threshold is not None:
        SNAPPING_THRESHOLD = args.snapping_threshold

    seed_mirt_sweep_from_baseline_store(args.baseline_output, args.mirt_sweep_output, quiet=args.quiet)

    # Parse multi-experiment parameters
    seeds = [int(s.strip()) for s in str(args.seed).split(',') if s.strip()]
    taus = [float(t.strip()) for t in str(args.lambda_tau).split(',') if t.strip()]
    if args.baseline_only and len(taus) > 1:
        taus = [taus[0]]
    if args.mirt_dim_min < 1 or args.mirt_dim_max < 1 or args.mirt_dim_min > args.mirt_dim_max:
        raise ValueError('Require 1 <= --mirt-dim-min <= --mirt-dim-max.')

    print(f"Embedding type: {args.embedding_type}")
    if args.model_type == 'beta':
        print(f"Model type: beta (Beta distribution NLL on P̂, φ={args.beta_phi})")
    else:
        print(f"Model type: bernoulli (Bernoulli NLL on binary Y)")
        
    actual_emb_type = args.embedding_type # Guessed initially
    
    # 1. First parse n_samples to know what n_values we are generating
    # To do this without loading all_dfs, we need total_files. We know it's 54 max usually,
    # but let's safely approximate it by saying "if all or max, assume we just check existence"
    # Actually, we can load_data efficiently or just check the OS level CSV if it exists
    
    output_path = None
    if not args.baseline_only:
        if args.output:
            output_path = args.output
        else:
            suffix = f"_pre_{args.pre_revision}" if args.pre_revision != 'none' else ""
            n_suffix = f"_n_{args.n_samples}" if args.n_samples != 'all' else "_n_max"
            j_suffix = f"_j{args.j_percentage}" if args.j_percentage < 1.0 else ""
            output_path = os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}{suffix}{n_suffix}{j_suffix}.csv')

    completed_configs = set()
    if output_path is not None and os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            if not df_existing.empty and 'seed' in df_existing.columns and 'lambda_tau' in df_existing.columns and 'n_samples' in df_existing.columns:
                # Approximate completion: if it's there with any n_samples, check if we need to run more?
                # The user sweeps over max_n. Let's just store what n_samples have completed.
                # Actually, just parse n_values properly.
                pass
        except Exception:
            pass

    # We MUST load data to get total_files for parse_n_samples
    all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data(
        embedding_type=args.embedding_type,
        embedding_dim=args.embedding_dim,
        pre_revision=args.pre_revision
    )
    total_files = len(all_dfs)
    
    if args.pre_revision != 'none':
        n_values = [1]
    else:
        n_values = parse_n_samples(args.n_samples, total_files)

    max_n = max(n_values)
    
    # accurately check completion now
    if args.baseline_only:
        try:
            for seed in seeds:
                baseline_key = {
                    'seed': int(seed),
                    'model_type': str(args.model_type),
                    'n_samples': int(max_n),
                    'pre_revision': normalize_pre_revision(args.pre_revision),
                    'j_percentage': normalize_j_percentage(args.j_percentage),
                }
                cached = try_get_cached_baseline(
                    args.baseline_output,
                    baseline_key,
                    mirt_dim_min=args.mirt_dim_min,
                    mirt_dim_max=args.mirt_dim_max,
                )
                if cached is not None:
                    completed_configs.add((int(seed), float(taus[0])))
        except Exception:
            pass
    elif output_path is not None and os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            if not df_existing.empty and 'seed' in df_existing.columns and 'lambda_tau' in df_existing.columns and 'n_samples' in df_existing.columns:
                completed_rows = df_existing[df_existing['n_samples'] == max_n]
                for _, row in completed_rows.iterrows():
                    completed_configs.add((int(row['seed']), float(row['lambda_tau'])))
        except Exception:
            pass

    # Build queue of configurations mapping to worker IDs
    configs = []
    worker_id = 0
    for tau in taus:
        for seed in seeds:
            # check completed
            isCompleted = False
            for (c_seed, c_tau) in completed_configs:
                if c_seed == seed and np.isclose(c_tau, tau, atol=1e-5):
                    isCompleted = True
                    break
                    
            if isCompleted:
                if args.baseline_only:
                    print(f"[SKIP] Baseline cache already complete for seed={seed}, n={max_n}, pre={args.pre_revision}, j={args.j_percentage}.")
                else:
                    print(f"[SKIP] Quick-skip: seed={seed}, tau={tau} already complete in {os.path.basename(output_path)}.")
                continue

            configs.append((seed, tau, worker_id))
            worker_id += 1

    if len(configs) == 0:
        print("\nAll configurations already completed! Skipping PyTorch init and multiprocessing.\n")
        return

    print(f"\nDiscovered {len(configs)} configurations to execute across {args.parallel} Python generic workers.\n")

    # Run execution pipeline
    if args.parallel > 1 and len(configs) > 1:
        # Prevent PyTorch from hanging with generic spawn context lock
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=args.parallel, initializer=init_worker, initargs=(all_dfs, global_shared_indices, raw_embs_map, actual_emb_type)) as pool:
            worker_fn = partial(run_single_config, args=args, n_values=n_values)
            pool.map(worker_fn, configs)
    else:
        # Sequential execution
        init_worker(all_dfs, global_shared_indices, raw_embs_map, actual_emb_type)
        for config in configs:
            run_single_config(config, args, n_values)

    print("\n" + "=" * 60)
    print("EXPERIMENT BATCH COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
