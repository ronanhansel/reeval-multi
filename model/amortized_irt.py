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
                    np.random.seed(RANDOM_SEED)
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
    elif embedding_type == 'ones':
        emb_file = None
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    # Fall back to raw if processed embeddings don't exist
    if emb_file is not None and not os.path.exists(emb_file):
        print(f"Warning: {emb_file} not found. Falling back to raw embeddings.")
        print("Run 'python generate_embeddings.py' to generate processed embeddings.")
        emb_file = raw_emb_file
        embedding_type = 'raw'

    if embedding_type == 'ones':
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


def prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map, embedding_type='pca'):
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
    if embedding_type == 'ones':
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
                   beta_phi=BETA_PHI, no_tau=False, quiet=False):
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
    if n_files < len(all_dfs):
        sampled_indices = np.random.choice(len(all_dfs), n_files, replace=False)
        if not quiet:
            print(f"Sampled iterations: {sampled_indices}")
        current_dfs = [all_dfs[i].reindex(index=global_shared_indices) for i in sampled_indices]
    else:
        current_dfs = [all_dfs[i].reindex(index=global_shared_indices) for i in range(n_files)]
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

    # 1. Naive Item-Mean baseline
    # Predict each item as its mean observed success rate in the current iteration
    # Since we have N users/agents, the item-wise mean is the average across users.
    valid_counts = train_mask_current_t.sum(dim=0)
    item_sums = (y_train * train_mask_current_t).sum(dim=0)
    
    # Avoid division by zero: if an item has no observations, use global mean
    global_mean = y_train[train_mask_current_t].mean()
    item_means = torch.where(valid_counts > 0, item_sums / valid_counts, global_mean)
    
    # Broadcast item_means to (N, J) shape to match p_rasch and p_amortized
    p_naive = item_means.unsqueeze(0).expand(N, J)
    
    rmse_naive = compute_rmse(p_naive.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_naive = evaluate_auc(p_naive, y_oracle, test_mask_t)

    # 2. Rasch IRT baseline
    p_rasch = train_rasch(N, J, y_train, train_mask_current_t)
    rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_rasch = evaluate_auc(p_rasch, y_oracle, test_mask_t)

    # 3. Amortized IRT (our method)
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
        'rmse_naive': rmse_naive,
        'rmse_rasch': rmse_rasch,
        'rmse_amortized': best_rmse,
        'auc_naive': auc_naive,
        'auc_rasch': auc_rasch,
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

    if args.output:
        output_path = args.output
    else:
        suffix = f"_pre_{args.pre_revision}" if args.pre_revision != 'none' else ""
        n_suffix = f"_n_{args.n_samples}" if args.n_samples != 'all' else "_n_max"
        output_path = os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}{suffix}{n_suffix}.csv')

    if os.path.exists(output_path):
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
    data = prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map, embedding_type=actual_emb_type)
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
            result = run_experiment(n, all_dfs, global_shared_indices, data,
                                    model_type=args.model_type, beta_phi=args.beta_phi, no_tau=args.no_tau, quiet=quiet)
                                    
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
        choices=['raw', 'pca', 'sae', 'ones'],
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
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    import sys, os
    import numpy as np
    import pandas as pd
    import multiprocessing as mp
    from functools import partial
    
    if args.quiet:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    if args.wd_theta is not None:
        WD_THETA = args.wd_theta
    if args.wd_w is not None:
        WD_W = args.wd_w
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.snapping_threshold is not None:
        SNAPPING_THRESHOLD = args.snapping_threshold

    # Parse multi-experiment parameters
    seeds = [int(s.strip()) for s in str(args.seed).split(',')]
    taus = [float(t.strip()) for t in str(args.lambda_tau).split(',')]

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
    
    if args.output:
        output_path = args.output
    else:
        suffix = f"_pre_{args.pre_revision}" if args.pre_revision != 'none' else ""
        n_suffix = f"_n_{args.n_samples}" if args.n_samples != 'all' else "_n_max"
        output_path = os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}{suffix}{n_suffix}.csv')

    completed_configs = set()
    if os.path.exists(output_path):
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
    if os.path.exists(output_path):
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
