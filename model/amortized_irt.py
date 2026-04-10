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
import hashlib
import json
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
from sklearn.decomposition import PCA

try:
    from hypothesaes.quickstart import train_sae
    HAS_SPLIT_SAE = True
except ImportError:
    HAS_SPLIT_SAE = False

from model.utility.utils import compute_rmse, evaluate_auc
import model.baseline_cache as bc
from model.utility.result_paths import configured_main_result_dir, ensure_main_result_dir

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = str(configured_main_result_dir())
ensure_main_result_dir()
os.makedirs(RESULT_DIR, exist_ok=True)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODEL_DIR)
PROCESSED_EMBEDDING_DIR = os.path.join(MODEL_DIR, 'processed_embeddings')
SPLIT_EMBEDDING_CACHE_DIR = os.path.join(PROCESSED_EMBEDDING_DIR, 'split_refit_v1')
os.makedirs(SPLIT_EMBEDDING_CACHE_DIR, exist_ok=True)

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
MIRT_SUMMARY_COLS = ['rmse_mirt', 'auc_mirt', 'selected_mirt_dim', 'mirt_sweep_min', 'mirt_sweep_max', 'mirt_selection_version']

BASELINE_PROFILE_FULL = 'full'
BASELINE_PROFILE_KNN_MIRT = 'knn_mirt'
BASELINE_PROFILE_KNN_ONLY = 'knn_only'
PRE_POPULATION_MATCH_NONE = 'none'
PRE_POPULATION_MATCH_TRANSPORT_BINARY_STRENGTH = 'transport_binary_strength'
PRE_POPULATION_MATCH_VERSION = 1

BASELINE_KEY_COLS = ['seed', 'model_type', 'n_samples', 'pre_revision', 'j_percentage', 'train_retention', 'baseline_embedding_type']
INLINE_BASELINE_COLS = BASELINE_METRIC_COLS.copy()
BASELINE_AUX_COLS = [
    'agent_batch_size',
    'selected_knn_k',
    'selected_mirt_dim',
    'mirt_sweep_min',
    'mirt_sweep_max',
    'mirt_selection_version',
    'pre_population_match',
    'pre_population_match_version',
    'effective_pre_weight_sum',
]

MIRT_SWEEP_METRIC_COLS = ['rmse_mirt', 'auc_mirt', 'val_rmse_mirt', 'val_auc_mirt']
MIRT_SWEEP_KEY_COLS = BASELINE_KEY_COLS + ['mirt_dim']

# Data paths
HF_REPO_ID = "ronanhansel/data-reeval-multi"

# Data split
TEST_SIZE = 0.1
RANDOM_SEED = 42
MIRT_SELECTION_VERSION = 2
MIRT_VALIDATION_FRACTION = 0.15
MIRT_VALIDATION_MIN_PAIRS = 12

BENCHMARKS = [
    'colbench_backend_programming',
    'corebench_hard',
    'scicode',
    'scienceagentbench',
]

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
KNN_K_GRID = (5, 10, 20, 50)

EMBEDDING_PROTOCOL_TRAIN_REFIT = 'train_items_only_refit_v1'
EMBEDDING_PROTOCOL_FROZEN_RAW = 'external_raw_frozen_v1'
EMBEDDING_PROTOCOL_CONSTANT = 'constant_features_v1'
EMBEDDING_FIT_SCOPE_TRAIN_ITEMS = 'train_items_only'
EMBEDDING_FIT_SCOPE_FROZEN = 'external_frozen'
EMBEDDING_FIT_SCOPE_CONSTANT = 'constant'
SPLIT_SAE_K_SPARSITY = 4
SPLIT_SAE_EPOCHS = 100
SPLIT_SAE_LR = 5e-4
SPLIT_SAE_BATCH_SIZE = 512

# CSV append operations for study sweeps can queue behind many workers.
CSV_LOCK_TIMEOUT = 7200

warnings.filterwarnings('ignore')

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def masked_weighted_mean(values, weights):
    weights = weights.to(values.dtype)
    denom = weights.sum().clamp_min(1e-12)
    return (values * weights).sum() / denom


def normalize_embedding_matrix(embeddings):
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


def _load_raw_embedding_map(raw_emb_file, processed_emb_dir):
    candidate_files = [raw_emb_file, os.path.join(processed_emb_dir, 'embeddings_raw.pkl')]
    emb_file = next((path for path in candidate_files if os.path.exists(path)), None)
    if emb_file is None:
        raise FileNotFoundError(
            f"No raw embedding source found. Checked: {candidate_files}"
        )

    print(f"Loading raw embedding source from {emb_file}...")
    emb_df = pd.read_pickle(emb_file)
    id_col = 'task_id' if 'task_id' in emb_df.columns else 'benchmark.task_id'
    raw_embs_map = {}
    for _, row in emb_df.iterrows():
        task_id = str(row[id_col])
        raw_embs_map[task_id] = row['embedding']
        if task_id.startswith('colbench_backend_programming'):
            suffix = task_id.split('.')[-1]
            raw_embs_map[f'colbench.{suffix}'] = row['embedding']
    return raw_embs_map


def align_raw_item_embeddings(task_ids, raw_embs_map):
    embeddings = []
    sample_emb = next(iter(raw_embs_map.values())) if raw_embs_map else [0.0]
    fallback_dim = len(sample_emb) if hasattr(sample_emb, '__len__') else 4096
    for task_id in task_ids:
        emb = raw_embs_map.get(str(task_id))
        if emb is None and str(task_id).startswith('colbench.'):
            number = str(task_id).split('.')[-1]
            emb = raw_embs_map.get(f'colbench_backend_programming.{number}')
        if emb is None:
            emb = np.zeros(fallback_dim, dtype=np.float32)
        elif isinstance(emb, str):
            emb = ast.literal_eval(emb)
        embeddings.append(np.asarray(emb, dtype=np.float32))
    return np.stack(embeddings)


def embedding_metadata_for_type(embedding_type):
    if embedding_type in {'pca', 'sae'}:
        return EMBEDDING_PROTOCOL_TRAIN_REFIT, EMBEDDING_FIT_SCOPE_TRAIN_ITEMS
    if embedding_type in {'ones', 'rasch_2pl', 'nonamortised_mirt'}:
        return EMBEDDING_PROTOCOL_CONSTANT, EMBEDDING_FIT_SCOPE_CONSTANT
    return EMBEDDING_PROTOCOL_FROZEN_RAW, EMBEDDING_FIT_SCOPE_FROZEN


def _split_embedding_cache_path(embedding_type, embedding_dim, task_ids, train_item_ids):
    payload = {
        'protocol': EMBEDDING_PROTOCOL_TRAIN_REFIT,
        'embedding_type': str(embedding_type),
        'embedding_dim': int(embedding_dim),
        'task_ids': [str(task_id) for task_id in task_ids],
        'train_item_ids': [str(task_id) for task_id in train_item_ids],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()[:24]
    return os.path.join(
        SPLIT_EMBEDDING_CACHE_DIR,
        f'{embedding_type}_{int(embedding_dim)}_{digest}.pkl'
    )


def fit_split_embeddings(raw_embeddings, task_ids, train_idx, embedding_type, embedding_dim):
    train_idx = np.asarray(train_idx, dtype=int)
    train_item_ids = [str(task_ids[idx]) for idx in train_idx.tolist()]
    cache_path = _split_embedding_cache_path(embedding_type, embedding_dim, task_ids, train_item_ids)
    lock = FileLock(f"{cache_path}.lock", timeout=CSV_LOCK_TIMEOUT)
    with lock:
        if os.path.exists(cache_path):
            payload = pd.read_pickle(cache_path)
            return payload['embeddings'], {
                'embedding_cache_path': cache_path,
                'embedding_fit_item_count': int(payload['fit_item_count']),
            }

        raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        raw_embeddings_norm = normalize_embedding_matrix(raw_embeddings)
        fit_source = raw_embeddings_norm[train_idx]
        fit_item_count = int(fit_source.shape[0])
        if fit_item_count <= 0:
            raise ValueError('Cannot fit split embeddings without train items.')

        if embedding_type == 'pca':
            n_components = min(int(embedding_dim), fit_source.shape[0], fit_source.shape[1])
            if n_components < 1:
                raise ValueError('PCA requires at least one train item and one feature.')
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            transformed = pca.fit(fit_source).transform(raw_embeddings_norm).astype(np.float32)
            if transformed.shape[1] < int(embedding_dim):
                padded = np.zeros((transformed.shape[0], int(embedding_dim)), dtype=np.float32)
                padded[:, :transformed.shape[1]] = transformed
                transformed = padded
            embeddings = normalize_embedding_matrix(transformed)
        elif embedding_type == 'sae':
            if not HAS_SPLIT_SAE:
                raise RuntimeError(
                    'SAE split refits require the hypothesaes package in the active environment.'
                )
            checkpoint_dir = os.path.join('/tmp', f'amortized_irt_split_sae_{os.path.basename(cache_path).replace(".pkl", "")}')
            sae = train_sae(
                embeddings=fit_source,
                M=int(embedding_dim),
                K=SPLIT_SAE_K_SPARSITY,
                batch_size=SPLIT_SAE_BATCH_SIZE,
                n_epochs=SPLIT_SAE_EPOCHS,
                learning_rate=SPLIT_SAE_LR,
                checkpoint_dir=checkpoint_dir,
            )
            embeddings = normalize_embedding_matrix(sae.get_activations(raw_embeddings_norm).astype(np.float32))
        else:
            raise ValueError(f'Unsupported split-refit embedding type: {embedding_type}')

        pd.to_pickle(
            {
                'embeddings': embeddings,
                'fit_item_count': fit_item_count,
                'task_ids': [str(task_id) for task_id in task_ids],
                'train_item_ids': train_item_ids,
            },
            cache_path,
        )
        return embeddings, {
            'embedding_cache_path': cache_path,
            'embedding_fit_item_count': fit_item_count,
        }


def _load_pre_revision_response_matrix(pre_revision):
    """Load the aggregated pre-revision matrix."""
    pre_rev_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'item-editor',
        'eval_response_matrix',
        'pre-revision',
    )
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

        if df is None:
            continue

        if b == 'scicode':
            scicode_post_ids = ['12', '14', '15', '16', '2', '23', '28', '32', '35', '41', '43', '46', '48', '52', '56', '58', '59', '61', '62', '63', '64', '66', '67', '71', '72', '77', '79', '80', '9']
            post_cols = [f"scicode.{it}" for it in scicode_post_ids]
            valid_df_cols = [c for c in df.columns if c in post_cols]
            df = df[valid_df_cols]

        df.index = [f"{b}.{a}" for a in df.index]
        df.columns = [f"{b}.{c}" if not str(c).startswith(b) and not str(c).startswith(b.replace('_hard', '')) else c for c in df.columns]
        combined_dfs.append(df)

    if not combined_dfs:
        raise FileNotFoundError(f"No pre-revision data found in {pre_rev_dir}")

    final_df = pd.concat(combined_dfs, axis=1, join='outer')

    final_df = final_df.dropna(axis=1, how='all')
    return [final_df], sorted(list(final_df.index))


def _load_post_revision_response_matrices():
    """Load the post-revision response-matrix ensemble."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    post_rev_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'post-revision')

    colbench_dir = os.path.join(post_rev_dir, 'colbench_backend_programming', 'resmat')
    colbench_files = sorted([f for f in os.listdir(colbench_dir) if f.startswith('resmat')])

    colbench_dfs = []
    for f in colbench_files:
        df = pd.read_csv(os.path.join(colbench_dir, f), index_col=0)
        valid_cols = [c for c in df.columns if str(c).startswith("colbench")]
        if valid_cols:
            df = df[valid_cols]
            df.columns = [f"colbench_backend_programming.{c}" if not str(c).startswith("colbench") else c for c in df.columns]
            colbench_dfs.append(df)

    other_benchmarks = [b for b in os.listdir(post_rev_dir) if b != 'colbench_backend_programming' and os.path.isdir(os.path.join(post_rev_dir, b))]

    from model.utility.utils import get_benchmark_iterations
    bench_iterations = {}
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
            idx = np.random.randint(0, len(iters))
            current_bench_parts.append(iters[idx])
        all_dfs.append(pd.concat(current_bench_parts, axis=1, join='outer'))

    global_shared_indices = sorted(list(set().union(*[df.index for df in all_dfs])))
    return all_dfs, global_shared_indices


def row_group_from_index(row_id):
    row_str = str(row_id)
    for benchmark in BENCHMARKS:
        aliases = {benchmark, benchmark.replace('_', '.')}
        if benchmark == 'colbench_backend_programming':
            aliases.add('colbench')
        if benchmark == 'corebench_hard':
            aliases.add('corebench')
        for alias in aliases:
            if row_str.startswith(f"{alias}.") or row_str.startswith(f"{alias}_"):
                return benchmark
    if '.' in row_str:
        return row_str.split('.', 1)[0]
    return row_str


def row_group_ids(frame, group):
    return [idx for idx in frame.index if row_group_from_index(idx) == group]


def columns_for_benchmark(columns, benchmark):
    return [col for col in columns if str(col).startswith(f"{benchmark}.")]


def row_strength_within_benchmark(frame, benchmark):
    cols = columns_for_benchmark(frame.columns, benchmark)
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame.loc[:, cols].to_numpy(dtype=float)
    return pd.Series(np.nanmean(values, axis=1), index=frame.index, dtype=float)


def effective_pre_population_match(pre_revision):
    pre = normalize_pre_revision(pre_revision)
    if pre in {'none', 'max'}:
        return PRE_POPULATION_MATCH_NONE
    try:
        int(pre)
    except Exception:
        return PRE_POPULATION_MATCH_NONE
    return PRE_POPULATION_MATCH_TRANSPORT_BINARY_STRENGTH


def target_benchmark_counts(n_total):
    n_total = int(n_total)
    n_per_benchmark = n_total // len(BENCHMARKS)
    remainder = n_total % len(BENCHMARKS)
    return {
        benchmark: n_per_benchmark + (1 if idx < remainder else 0)
        for idx, benchmark in enumerate(BENCHMARKS)
    }


def compute_post_binary_oracle():
    post_dfs, post_indices = _load_post_revision_response_matrices()
    all_columns = sorted(list(set().union(*[df.columns for df in post_dfs])))
    post_filtered = [df.reindex(index=post_indices, columns=all_columns) for df in post_dfs]
    post_stack = np.array([df.values for df in post_filtered], dtype=float)
    post_mean = np.nanmean(post_stack, axis=0)
    return pd.DataFrame((post_mean > 0.5).astype(np.float32), index=post_indices, columns=all_columns)


def deterministic_quantile_support(values, target_count):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if target_count <= 0 or arr.size == 0:
        return np.array([], dtype=float)
    arr.sort()
    if arr.size == target_count:
        return arr.copy()
    if target_count == 1:
        return np.array([float(np.nanmedian(arr))], dtype=float)
    qs = np.linspace(0.0, 1.0, target_count)
    return np.quantile(arr, qs, method='linear')


def compute_pre_revision_row_weights(pre_df, pre_revision, post_binary_df=None):
    pre = normalize_pre_revision(pre_revision)
    weights = pd.Series(1.0, index=pre_df.index, dtype=float)
    benchmark_weight_sums = {benchmark: 0.0 for benchmark in BENCHMARKS}

    if pre == 'none':
        return {
            'row_weights': weights,
            'pre_population_match': PRE_POPULATION_MATCH_NONE,
            'pre_population_match_version': np.nan,
            'effective_pre_weight_sum': float(weights.sum()),
            'benchmark_weight_sums': benchmark_weight_sums,
        }
    if pre == 'max':
        for benchmark in BENCHMARKS:
            benchmark_rows = row_group_ids(pre_df, benchmark)
            benchmark_weight_sums[benchmark] = float(weights.loc[benchmark_rows].sum())
        return {
            'row_weights': weights,
            'pre_population_match': PRE_POPULATION_MATCH_NONE,
            'pre_population_match_version': np.nan,
            'effective_pre_weight_sum': float(weights.sum()),
            'benchmark_weight_sums': benchmark_weight_sums,
        }

    target_counts = target_benchmark_counts(int(pre))
    weights = pd.Series(0.0, index=pre_df.index, dtype=float)
    if post_binary_df is None:
        post_binary_df = compute_post_binary_oracle()

    for benchmark, target_count in target_counts.items():
        if target_count <= 0:
            continue
        pre_rows = row_group_ids(pre_df, benchmark)
        post_rows = row_group_ids(post_binary_df, benchmark)
        if not pre_rows:
            raise ValueError(f"No pre rows found for benchmark {benchmark}.")
        if not post_rows:
            raise ValueError(f"No post rows found for benchmark {benchmark}.")
        pre_strength = row_strength_within_benchmark(pre_df.loc[pre_rows], benchmark).to_numpy(dtype=float)
        post_strength = row_strength_within_benchmark(post_binary_df.loc[post_rows], benchmark).to_numpy(dtype=float)
        target_support = deterministic_quantile_support(post_strength, target_count)
        if target_support.size == 0:
            continue
        for target_value in target_support:
            distances = np.abs(pre_strength - target_value)
            distances[~np.isfinite(distances)] = np.inf
            if np.isfinite(distances).any():
                chosen_pos = int(np.argmin(distances))
            else:
                chosen_pos = 0
            weights.loc[pre_rows[chosen_pos]] += 1.0
        benchmark_weight_sums[benchmark] = float(weights.loc[pre_rows].sum())

    return {
        'row_weights': weights,
        'pre_population_match': PRE_POPULATION_MATCH_TRANSPORT_BINARY_STRENGTH,
        'pre_population_match_version': PRE_POPULATION_MATCH_VERSION,
        'effective_pre_weight_sum': float(weights.sum()),
        'benchmark_weight_sums': benchmark_weight_sums,
    }


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

def train_rasch(N, J, y_train, train_mask_t, row_weight_t=None, n_outer_iter=100):
    """Train a basic Rasch IRT model (baseline)."""
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)

    optimizer = torch.optim.LBFGS(
        [theta, beta], lr=0.1, max_iter=20,
        history_size=10, line_search_fn="strong_wolfe"
    )
    if row_weight_t is None:
        row_weight_t = torch.ones(N, device=device)

    def closure():
        optimizer.zero_grad()
        probs = torch.sigmoid(theta.unsqueeze(1) - beta.unsqueeze(0))
        loss = F.binary_cross_entropy(probs, y_train, reduction='none')
        loss_weight = train_mask_t.float() * row_weight_t.unsqueeze(1)
        total_loss = masked_weighted_mean(loss, loss_weight)
        total_loss.backward()
        return total_loss

    for _ in range(n_outer_iter):
        optimizer.step(closure)

    with torch.no_grad():
        probs = torch.sigmoid(theta.unsqueeze(1) - beta.unsqueeze(0))

    return probs


def train_2pl(N, J, y_train, train_mask_t, row_weight_t=None, n_outer_iter=100):
    """Train a 2-parameter logistic (2PL) IRT model (baseline)."""
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
    alpha = nn.Parameter(torch.ones(J, device=device)) # Discrimination

    optimizer = torch.optim.LBFGS(
        [theta, beta, alpha], lr=0.1, max_iter=20,
        history_size=10, line_search_fn="strong_wolfe"
    )
    if row_weight_t is None:
        row_weight_t = torch.ones(N, device=device)

    def closure():
        optimizer.zero_grad()
        probs = torch.sigmoid(alpha.unsqueeze(0) * (theta.unsqueeze(1) - beta.unsqueeze(0)))
        loss = F.binary_cross_entropy(probs, y_train, reduction='none')
        loss_weight = train_mask_t.float() * row_weight_t.unsqueeze(1)
        total_loss = masked_weighted_mean(loss, loss_weight)
        total_loss.backward()
        return total_loss

    for _ in range(n_outer_iter):
        optimizer.step(closure)

    with torch.no_grad():
        probs = torch.sigmoid(alpha.unsqueeze(0) * (theta.unsqueeze(1) - beta.unsqueeze(0)))

    return probs


def train_amortized_irt(model, y_train, train_mask_t, y_oracle, test_mask_oracle,
                        model_type='beta', beta_phi=BETA_PHI, epochs=EPOCHS, lambda_tau=LAMBDA_TAU,
                        quiet=False, row_weight_t=None):
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
    if row_weight_t is None:
        row_weight_t = torch.ones(y_train.shape[0], device=y_train.device)

    eps = 1e-6

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        optimizer_tau.zero_grad()
        probs = model()
        observed_row_weights = row_weight_t.unsqueeze(1).expand_as(y_train)[train_mask_t]

        # Reconstruction loss using torch.distributions
        p = probs[train_mask_t].clamp(eps, 1 - eps)
        if model_type == 'beta':
            # Beta NLL: α = μφ, β = (1-μ)φ
            y = y_train[train_mask_t].clamp(eps, 1 - eps)
            dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
            loss_fit = masked_weighted_mean(-dist.log_prob(y), observed_row_weights)
        else:
            # Bernoulli NLL
            y = y_train[train_mask_t]
            dist = torch.distributions.Bernoulli(probs=p)
            loss_fit = masked_weighted_mean(-dist.log_prob(y), observed_row_weights)

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
                
                # Test metrics
                curr_rmse = compute_rmse(p_eval.cpu().numpy(), y_oracle.cpu().numpy(), test_mask_oracle)
                
                tau_vals = model.get_tau()
                if not quiet:
                    print(f"Epoch {epoch:4d} | Loss: {loss_fit.item():.4f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {curr_rmse:.4f}")
                    
                    active_indices = torch.where(tau_vals > SNAPPING_THRESHOLD)[0].cpu().numpy()
                    print(f"  Active Dims ({len(active_indices)}): Tau Mean={tau_vals.mean().item():.4f}, Max={tau_vals.max().item():.4f}")
                    print("-" * 40)

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        p_final = model()
        final_rmse = compute_rmse(p_final.cpu().numpy(), y_oracle.cpu().numpy(), test_mask_oracle)
    return final_rmse, final_state


def adapt_amortized_irt_users(model, y_support, support_mask_t, model_type='beta',
                              beta_phi=BETA_PHI, epochs=None, quiet=False, row_weight_t=None):
    """Infer user latents for a new response matrix while keeping item parameters fixed."""
    for name, param in model.named_parameters():
        param.requires_grad = name in {'theta', 'theta_bias'}

    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
    ])

    fit_epochs = max(200, EPOCHS // 2) if epochs is None else int(epochs)
    eps = 1e-6
    best_state = None
    best_loss = float('inf')
    if row_weight_t is None:
        row_weight_t = torch.ones(y_support.shape[0], device=y_support.device)

    for epoch in range(fit_epochs):
        model.train()
        optimizer.zero_grad()
        probs = model()
        p = probs[support_mask_t].clamp(eps, 1 - eps)
        observed_row_weights = row_weight_t.unsqueeze(1).expand_as(y_support)[support_mask_t]
        if model_type == 'beta':
            y = y_support[support_mask_t].clamp(eps, 1 - eps)
            dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
            loss = masked_weighted_mean(-dist.log_prob(y), observed_row_weights)
        else:
            y = y_support[support_mask_t]
            dist = torch.distributions.Bernoulli(probs=p)
            loss = masked_weighted_mean(-dist.log_prob(y), observed_row_weights)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not quiet and (epoch % EVAL_EVERY == 0 or epoch == fit_epochs - 1):
            print(f"Adapt Epoch {epoch:4d} | Support Loss: {loss_val:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        probs = model().detach().clone()
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return probs, final_state


    # Combined parser moved to main()


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(embedding_type='pca', embedding_dim=48, pre_revision='none'):
    """
    Load response matrices and the raw embedding source used by all protocols.

    Args:
        embedding_type: 'raw', 'pca', or 'sae'
        embedding_dim: dimension for pca/sae embeddings (ignored at load time)
        pre_revision: 'none', '8', or 'max' to override post-revision loading.
    """
    repo_root = REPO_ROOT
    resmat_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix')

    processed_emb_dir = PROCESSED_EMBEDDING_DIR
    raw_emb_file = os.path.join(resmat_dir, 'all_benchmarks_embeddings_4096_8B.pkl')

    if pre_revision != 'none':
        print(f"Bypassing Post-Revision. Loading Pre-Revision Data ({pre_revision} agents)...")
        all_dfs, global_shared_indices = _load_pre_revision_response_matrix(pre_revision)
    else:
        all_dfs, global_shared_indices = _load_post_revision_response_matrices()

    if embedding_type not in ['raw', 'pca', 'sae', 'ones', 'rasch_2pl', 'nonamortised_mirt']:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    if embedding_type in ['ones', 'rasch_2pl', 'nonamortised_mirt']:
        raw_embs_map = {}
    else:
        raw_embs_map = _load_raw_embedding_map(raw_emb_file, processed_emb_dir)

    return all_dfs, global_shared_indices, raw_embs_map, embedding_type


def prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map, embedding_type='pca', j_percentage=1.0,
                            pre_revision='none', cross_revision_post_binary=False, user_count=None,
                            binarize_oracle=False):
    """Prepare oracle ground truth, train/test splits, and embedding tensors."""
    print("=" * 60)
    print("PREPARING EXPERIMENT DATA")
    print("=" * 60)

    pre_revision_norm = normalize_pre_revision(pre_revision)

    if cross_revision_post_binary and pre_revision_norm != 'none':
        post_dfs, post_shared_indices = _load_post_revision_response_matrices()
        all_columns = sorted(list(set().union(*[df.columns for df in post_dfs])))
        oracle_dfs_filtered = [df.reindex(index=post_shared_indices, columns=all_columns) for df in post_dfs]
        oracle_stacked = np.array([df.values for df in oracle_dfs_filtered], dtype=float)
        oracle_matrix = np.nanmean(oracle_stacked, axis=0)
        oracle_df = pd.DataFrame(oracle_matrix, index=post_shared_indices, columns=all_columns)

        print(f"Pre-train matrices: {len(all_dfs)}")
        print(f"Post-eval matrices: {len(post_dfs)}")
        print(f"Post oracle shape: {oracle_df.shape}")
        train_all_dfs = all_dfs
        train_global_shared_indices = global_shared_indices
    else:
        all_columns = sorted(list(set().union(*[df.columns for df in all_dfs])))
        oracle_dfs_filtered = [df.reindex(index=global_shared_indices, columns=all_columns) for df in all_dfs]
        oracle_stacked = np.array([df.values for df in oracle_dfs_filtered], dtype=float)
        oracle_matrix = np.nanmean(oracle_stacked, axis=0)
        oracle_df = pd.DataFrame(oracle_matrix, index=global_shared_indices, columns=all_columns)

        print(f"Total matrices: {len(all_dfs)}")
        print(f"Global user intersection: {len(global_shared_indices)} users")
        print(f"Oracle matrix shape: {oracle_df.shape}")
        train_all_dfs = all_dfs
        train_global_shared_indices = global_shared_indices

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N, J_full = oracle_df.shape

    weight_meta = compute_pre_revision_row_weights(
        all_dfs[0],
        pre_revision_norm,
        post_binary_df=(compute_post_binary_oracle() if pre_revision_norm not in {'none', 'max'} else None),
    )
    full_row_weights = weight_meta['row_weights'].reindex(train_global_shared_indices).fillna(0.0)

    if user_count is not None:
        user_count = int(user_count)
        if 0 < user_count < N:
            all_user_indices = np.arange(N)
            np.random.seed(RANDOM_SEED + 4242)
            sampled_user_indices = np.random.choice(all_user_indices, size=user_count, replace=False)
            sampled_user_indices.sort()
            sampled_users = oracle_df.index[sampled_user_indices].tolist()
            oracle_df = oracle_df.iloc[sampled_user_indices]
            train_global_shared_indices = sampled_users
            train_all_dfs = [df.reindex(index=sampled_users) for df in train_all_dfs]
            full_row_weights = full_row_weights.reindex(sampled_users).fillna(0.0)
            N, J_full = oracle_df.shape
            print(f"Sub-sampling Users: {len(global_shared_indices)} -> {user_count}")

    if j_percentage < 1.0:
        n_j_keep = max(10, int(j_percentage * J_full))
        all_j_indices = np.arange(J_full)
        np.random.seed(RANDOM_SEED + 999)
        sampled_j_indices = np.random.choice(all_j_indices, size=n_j_keep, replace=False)
        sampled_j_indices.sort()

        oracle_df = oracle_df.iloc[:, sampled_j_indices]
        print(f"Sub-sampling Items: {J_full} -> {n_j_keep} ({j_percentage*100:.1f}%)")

    N, J = oracle_df.shape
    sampled_columns = oracle_df.columns.tolist()

    train_all_dfs_filtered = [df.reindex(index=train_global_shared_indices, columns=sampled_columns) for df in train_all_dfs]

    J_indices = np.arange(J)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    print(f"Train items: {len(train_idx)}, Test items: {len(test_idx)}")

    oracle_values = oracle_df.values.copy()
    oracle_mask = ~np.isnan(oracle_values)
    if (cross_revision_post_binary and pre_revision_norm != 'none') or binarize_oracle:
        oracle_values = (oracle_values > 0.5).astype(np.float32)
    oracle_values_clean = np.nan_to_num(oracle_values, nan=0.5)
    y_oracle = torch.from_numpy(oracle_values_clean.astype(np.float32)).to(device)

    train_mask = np.zeros_like(oracle_df.values, dtype=bool)
    train_mask[:, train_idx] = oracle_mask[:, train_idx]

    test_mask = np.zeros_like(oracle_df.values, dtype=bool)
    test_mask[:, test_idx] = oracle_mask[:, test_idx]

    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    task_ids = oracle_df.columns.tolist()
    embedding_protocol, embedding_fit_scope = embedding_metadata_for_type(embedding_type)
    if embedding_type in ['ones', 'rasch_2pl', 'nonamortised_mirt']:
        embeddings = np.ones((len(task_ids), 1), dtype=np.float32)
        embedding_cache_path = ''
        embedding_fit_item_count = 0
    else:
        raw_embeddings = align_raw_item_embeddings(task_ids, raw_embs_map)
        if embedding_type in {'pca', 'sae'}:
            embeddings, fit_meta = fit_split_embeddings(
                raw_embeddings,
                task_ids,
                train_idx,
                embedding_type,
                embedding_dim,
            )
            embedding_cache_path = fit_meta['embedding_cache_path']
            embedding_fit_item_count = int(fit_meta['embedding_fit_item_count'])
        else:
            embeddings = normalize_embedding_matrix(raw_embeddings)
            embedding_cache_path = ''
            embedding_fit_item_count = 0

    x_j = torch.tensor(embeddings, dtype=torch.float32).to(device)
    train_item_ids = [str(task_ids[idx]) for idx in train_idx.tolist()]
    test_item_ids = [str(task_ids[idx]) for idx in test_idx.tolist()]

    print(f"Embeddings shape: {x_j.shape}")

    return {
        'all_dfs': train_all_dfs_filtered,
        'y_oracle': y_oracle,
        'y_auc_oracle': y_oracle,
        'train_mask_t': train_mask_t,
        'test_mask_t': test_mask_t,
        'auc_test_mask_t': test_mask_t,
        'test_mask': test_mask,
        'x_j': x_j,
        'test_idx': test_idx,
        'train_idx': train_idx,
        'pre_train_idx': (
            np.arange(J, dtype=int)
            if cross_revision_post_binary and pre_revision_norm != 'none'
            else train_idx
        ),
        'N': N,
        'J': J,
        'embedding_dim': x_j.shape[1],
        'cross_revision_post_binary': bool(cross_revision_post_binary and pre_revision_norm != 'none'),
        'train_global_shared_indices': train_global_shared_indices,
        'support_values': y_oracle.detach().cpu().numpy(),
        'support_mask': train_mask,
        'user_count': N,
        'row_weights': full_row_weights.to_numpy(dtype=np.float32),
        'pre_population_match': weight_meta['pre_population_match'],
        'pre_population_match_version': weight_meta['pre_population_match_version'],
        'effective_pre_weight_sum': float(full_row_weights.sum()),
        'benchmark_weight_sums': weight_meta['benchmark_weight_sums'],
        'task_ids': [str(task_id) for task_id in task_ids],
        'train_item_ids': train_item_ids,
        'test_item_ids': test_item_ids,
        'embedding_protocol': embedding_protocol,
        'embedding_fit_scope': embedding_fit_scope,
        'embedding_cache_path': embedding_cache_path,
        'embedding_fit_item_count': int(embedding_fit_item_count),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════════

def build_training_targets(n_files, all_dfs, global_shared_indices, data, model_type='beta',
                           quiet=False, train_retention=1.0):
    """Build training matrix/mask for a specific n_files configuration."""
    train_idx = np.asarray(data.get('pre_train_idx', data['train_idx']), dtype=int)

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
    N, J = train_target_df.shape

    train_values = train_target_df.values.copy()
    if model_type == 'bernoulli':
        train_values = (train_values > 0.5).astype(np.float32)

    train_values = np.nan_to_num(train_values, nan=0.5)
    y_train = torch.from_numpy(train_values.astype(np.float32)).to(device)

    train_mask_current = np.zeros_like(train_target_df.values, dtype=bool)
    train_mask_current[:, train_idx] = ~np.isnan(train_target_df.values)[:, train_idx]

    train_retention = float(train_retention)
    if train_retention < 1.0:
        observed_coords = np.argwhere(train_mask_current)
        if observed_coords.size > 0:
            rng = np.random.default_rng(RANDOM_SEED + 1701)
            keep_mask = rng.random(observed_coords.shape[0]) < train_retention
            train_mask_current[:, :] = False
            kept_coords = observed_coords[keep_mask]
            if kept_coords.size == 0:
                kept_coords = observed_coords[rng.choice(observed_coords.shape[0], size=1, replace=False)]
            train_mask_current[kept_coords[:, 0], kept_coords[:, 1]] = True

    train_mask_current_t = torch.from_numpy(train_mask_current).to(device)
    row_weights = np.asarray(data.get('row_weights', np.ones(N, dtype=np.float32)), dtype=np.float32)
    if row_weights.shape[0] != N:
        row_weights = np.ones(N, dtype=np.float32)
    row_weight_t = torch.from_numpy(row_weights.astype(np.float32)).to(device)

    return N, J, y_train, train_mask_current_t, row_weight_t


def build_cross_revision_support_targets(data, train_retention=1.0):
    """Build the observed post-revision support matrix used for adaptation/evaluation."""
    support_values = np.asarray(data['support_values'], dtype=np.float32)
    support_mask_current = np.asarray(data['support_mask'], dtype=bool).copy()

    train_retention = float(train_retention)
    if train_retention < 1.0:
        observed_coords = np.argwhere(support_mask_current)
        if observed_coords.size > 0:
            rng = np.random.default_rng(RANDOM_SEED + 1701)
            keep_mask = rng.random(observed_coords.shape[0]) < train_retention
            support_mask_current[:, :] = False
            kept_coords = observed_coords[keep_mask]
            if kept_coords.size == 0:
                kept_coords = observed_coords[rng.choice(observed_coords.shape[0], size=1, replace=False)]
            support_mask_current[kept_coords[:, 0], kept_coords[:, 1]] = True

    y_support = torch.from_numpy(support_values.astype(np.float32)).to(device)
    support_mask_current_t = torch.from_numpy(support_mask_current).to(device)
    return y_support, support_mask_current_t


def prune_embedding_only_train_columns(y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                                       x_j=None, y_auc_oracle=None, auc_mask_t=None, task_ids=None):
    """Drop train columns with zero retained support while preserving evaluated test columns."""
    test_mask_t = test_mask_t if test_mask_t is not None else torch.from_numpy(np.asarray(test_mask, dtype=bool)).to(y_train.device)
    y_auc_oracle = y_oracle if y_auc_oracle is None else y_auc_oracle
    auc_mask_t = test_mask_t if auc_mask_t is None else auc_mask_t

    active_mask_t = train_mask_current_t.any(dim=0) | test_mask_t.any(dim=0)
    active_mask = active_mask_t.detach().cpu().numpy().astype(bool)
    if active_mask.all():
        return {
            'N': int(y_train.shape[0]),
            'J': int(y_train.shape[1]),
            'y_train': y_train,
            'train_mask_t': train_mask_current_t,
            'y_oracle': y_oracle,
            'y_auc_oracle': y_auc_oracle,
            'test_mask': np.asarray(test_mask, dtype=bool),
            'test_mask_t': test_mask_t,
            'auc_mask_t': auc_mask_t,
            'x_j': x_j,
            'task_ids': list(task_ids) if task_ids is not None else None,
            'train_item_ids': [
                str(task_ids[idx]) for idx in np.where(train_mask_current_t.any(dim=0).detach().cpu().numpy().astype(bool))[0].tolist()
            ] if task_ids is not None else None,
            'test_item_ids': [
                str(task_ids[idx]) for idx in np.where(test_mask_t.any(dim=0).detach().cpu().numpy().astype(bool))[0].tolist()
            ] if task_ids is not None else None,
            'test_idx': np.where(test_mask_t.any(dim=0).detach().cpu().numpy().astype(bool))[0],
        }

    y_train_pruned = y_train[:, active_mask_t]
    train_mask_pruned_t = train_mask_current_t[:, active_mask_t]
    y_oracle_pruned = y_oracle[:, active_mask_t]
    y_auc_oracle_pruned = y_auc_oracle[:, active_mask_t]
    test_mask_pruned = np.asarray(test_mask, dtype=bool)[:, active_mask]
    test_mask_pruned_t = test_mask_t[:, active_mask_t]
    auc_mask_pruned_t = auc_mask_t[:, active_mask_t]
    x_j_pruned = x_j[active_mask_t] if x_j is not None else None
    task_ids_pruned = [str(task_ids[idx]) for idx in np.where(active_mask)[0].tolist()] if task_ids is not None else None
    train_item_ids = [
        str(task_ids[idx]) for idx in np.where(train_mask_current_t.any(dim=0).detach().cpu().numpy().astype(bool))[0].tolist() if active_mask[idx]
    ] if task_ids is not None else None
    test_item_ids = [
        str(task_ids[idx]) for idx in np.where(test_mask_t.any(dim=0).detach().cpu().numpy().astype(bool))[0].tolist() if active_mask[idx]
    ] if task_ids is not None else None

    return {
        'N': int(y_train_pruned.shape[0]),
        'J': int(y_train_pruned.shape[1]),
        'y_train': y_train_pruned,
        'train_mask_t': train_mask_pruned_t,
        'y_oracle': y_oracle_pruned,
        'y_auc_oracle': y_auc_oracle_pruned,
        'test_mask': test_mask_pruned,
        'test_mask_t': test_mask_pruned_t,
        'auc_mask_t': auc_mask_pruned_t,
        'x_j': x_j_pruned,
        'task_ids': task_ids_pruned,
        'train_item_ids': train_item_ids,
        'test_item_ids': test_item_ids,
        'test_idx': np.where(test_mask_pruned.any(axis=0))[0],
    }


def compute_knn_predictions(y_train, train_mask_current_t, x_j, test_mask, knn_k=KNN_K, row_weight_t=None):
    """Return kNN predictions plus per-pair neighborhood support diagnostics."""
    N, J = y_train.shape
    if row_weight_t is None:
        row_weight_t = torch.ones(N, device=y_train.device, dtype=torch.float32)

    weighted_mask = train_mask_current_t.float() * row_weight_t.unsqueeze(1)
    valid_counts = weighted_mask.sum(dim=0)
    item_sums = (y_train * weighted_mask).sum(dim=0)
    global_mean = masked_weighted_mean(y_train, weighted_mask)
    item_means = torch.where(valid_counts > 0, item_sums / valid_counts, global_mean)
    p_naive = item_means.unsqueeze(0).expand(N, J)

    train_obs = train_mask_current_t.float()
    user_counts = train_obs.sum(dim=1)
    user_means = torch.where(
        user_counts > 0,
        (y_train * train_obs).sum(dim=1) / user_counts.clamp_min(1.0),
        global_mean
    )

    coverage_count = torch.zeros((N, J), dtype=torch.float32, device=y_train.device)
    coverage_rate = torch.zeros((N, J), dtype=torch.float32, device=y_train.device)
    weighted_coverage = torch.zeros((N, J), dtype=torch.float32, device=y_train.device)
    fallback_mask = torch.zeros((N, J), dtype=torch.bool, device=y_train.device)
    top_similarity = torch.zeros((J,), dtype=torch.float32, device=y_train.device)

    if x_j is None:
        return p_naive.clone(), {
            'coverage_count': coverage_count,
            'coverage_rate': coverage_rate,
            'weighted_coverage': weighted_coverage,
            'fallback_mask': fallback_mask,
            'top_similarity': top_similarity,
            'k_eff': 0,
        }

    x = x_j
    if x.dim() != 2 or x.shape[0] != J:
        return p_naive.clone(), {
            'coverage_count': coverage_count,
            'coverage_rate': coverage_rate,
            'weighted_coverage': weighted_coverage,
            'fallback_mask': fallback_mask,
            'top_similarity': top_similarity,
            'k_eff': 0,
        }

    train_item_mask = train_mask_current_t.any(dim=0)
    test_item_mask = torch.from_numpy(test_mask).to(y_train.device).any(dim=0)

    train_item_idx = torch.where(train_item_mask)[0]
    test_item_idx = torch.where(test_item_mask)[0]

    if train_item_idx.numel() == 0 or test_item_idx.numel() == 0:
        return user_means.unsqueeze(1).expand(N, J).clone(), {
            'coverage_count': coverage_count,
            'coverage_rate': coverage_rate,
            'weighted_coverage': weighted_coverage,
            'fallback_mask': fallback_mask,
            'top_similarity': top_similarity,
            'k_eff': 0,
        }

    p_knn = user_means.unsqueeze(1).expand(N, J).clone()

    x_norm = F.normalize(x, dim=1)
    sims = x_norm[test_item_idx] @ x_norm[train_item_idx].T
    k_eff = max(1, min(int(knn_k), train_item_idx.numel()))
    sim_vals, sim_pos = torch.topk(sims, k=k_eff, dim=1)
    nn_item_idx = train_item_idx[sim_pos]
    top_similarity[test_item_idx] = sim_vals[:, 0]

    nn_weights = torch.clamp(sim_vals, min=0.0)
    zero_rows = nn_weights.sum(dim=1, keepdim=True) <= 1e-12
    if zero_rows.any():
        nn_weights[zero_rows.squeeze(1)] = 1.0

    for t in range(test_item_idx.numel()):
        item_idx = test_item_idx[t]
        nbrs = nn_item_idx[t]
        w = nn_weights[t].unsqueeze(0)
        obs = train_mask_current_t[:, nbrs].float()
        yy = y_train[:, nbrs]

        num = (yy * obs * w).sum(dim=1)
        den = (obs * w).sum(dim=1)
        pred = torch.where(den > 0, num / den.clamp_min(1e-12), user_means)
        p_knn[:, item_idx] = pred

        coverage_count[:, item_idx] = obs.sum(dim=1)
        coverage_rate[:, item_idx] = obs.sum(dim=1) / float(k_eff)
        weighted_coverage[:, item_idx] = den / w.sum()
        fallback_mask[:, item_idx] = den <= 1e-12

    return p_knn, {
        'coverage_count': coverage_count,
        'coverage_rate': coverage_rate,
        'weighted_coverage': weighted_coverage,
        'fallback_mask': fallback_mask,
        'top_similarity': top_similarity,
        'k_eff': k_eff,
    }


def build_pair_efficiency_row(n_files, model_type, pre_revision, j_percentage, embedding_type,
                              baseline_embedding_type, observed_train_pairs, baselines,
                              rmse_amortized, auc_amortized):
    """Summarize one run for observed-pair efficiency analysis."""
    return {
        'seed': int(RANDOM_SEED),
        'lambda_tau': float(LAMBDA_TAU),
        'n_samples': int(n_files),
        'model_type': str(model_type),
        'pre_revision': normalize_pre_revision(pre_revision),
        'j_percentage': normalize_j_percentage(j_percentage),
        'embedding_type': str(embedding_type),
        'baseline_embedding_type': normalize_baseline_embedding_type(baseline_embedding_type),
        'observed_train_pairs': int(observed_train_pairs),
        'auc_knn': float(baselines['auc_knn']),
        'rmse_knn': float(baselines['rmse_knn']),
        'auc_araf': float(auc_amortized),
        'rmse_araf': float(rmse_amortized),
    }


def build_support_thinning_row(n_files, model_type, pre_revision, j_percentage, embedding_type,
                               baseline_embedding_type, knn_k, train_retention, observed_train_pairs,
                               baselines, rmse_amortized, auc_amortized):
    return {
        'seed': int(RANDOM_SEED),
        'lambda_tau': float(LAMBDA_TAU),
        'n_samples': int(n_files),
        'model_type': str(model_type),
        'pre_revision': normalize_pre_revision(pre_revision),
        'j_percentage': normalize_j_percentage(j_percentage),
        'embedding_type': str(embedding_type),
        'baseline_embedding_type': normalize_baseline_embedding_type(baseline_embedding_type),
        'knn_k': int(knn_k),
        'train_retention': float(train_retention),
        'observed_train_pairs': int(observed_train_pairs),
        'auc_knn': float(baselines['auc_knn']),
        'rmse_knn': float(baselines['rmse_knn']),
        'auc_araf': float(auc_amortized),
        'rmse_araf': float(rmse_amortized),
    }


def append_support_thinning_rows(path, rows):
    if not path or not rows:
        return

    lock = FileLock(f"{path}.lock", timeout=CSV_LOCK_TIMEOUT)
    with lock:
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path)
            except Exception:
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        df_new = pd.DataFrame(rows)
        df = pd.concat([df_old, df_new], ignore_index=True)
        dedupe_cols = [
            'seed', 'lambda_tau', 'n_samples', 'model_type', 'pre_revision',
            'j_percentage', 'embedding_type', 'baseline_embedding_type', 'knn_k', 'train_retention'
        ]
        present_cols = [c for c in dedupe_cols if c in df.columns]
        if present_cols:
            df = df.drop_duplicates(subset=present_cols, keep='last')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def _masked_prob_arrays(pred_tensor, target_tensor, mask_tensor):
    valid_mask = mask_tensor.detach()
    if not valid_mask.any():
        return np.array([]), np.array([])
    y_true = target_tensor[valid_mask].detach().cpu().numpy().astype(float)
    y_pred = pred_tensor[valid_mask].detach().cpu().numpy().astype(float)
    return y_true, y_pred


def compute_prob_metrics(pred_tensor, target_tensor, mask_tensor, n_bins=10):
    y_true, y_pred = _masked_prob_arrays(pred_tensor, target_tensor, mask_tensor)
    if y_true.size == 0:
        return {
            'brier': np.nan,
            'log_loss': np.nan,
            'ece': np.nan,
            'mean_ae': np.nan,
            'p90ae': np.nan,
            'p95ae': np.nan,
        }

    y_true_bin = (y_true > 0.5).astype(float)
    y_pred_clip = np.clip(y_pred, 1e-6, 1 - 1e-6)
    abs_err = np.abs(y_pred - y_true)
    brier = float(np.mean((y_pred - y_true) ** 2))
    log_loss = float(-np.mean(y_true_bin * np.log(y_pred_clip) + (1 - y_true_bin) * np.log(1 - y_pred_clip)))

    order = np.argsort(y_pred_clip)
    y_pred_sorted = y_pred_clip[order]
    y_true_sorted = y_true_bin[order]
    bins = np.array_split(np.arange(y_pred_sorted.size), min(n_bins, y_pred_sorted.size))
    ece = 0.0
    total = float(y_pred_sorted.size)
    for idx in bins:
        if idx.size == 0:
            continue
        conf = float(np.mean(y_pred_sorted[idx]))
        acc = float(np.mean(y_true_sorted[idx]))
        ece += (idx.size / total) * abs(acc - conf)

    return {
        'brier': brier,
        'log_loss': log_loss,
        'ece': float(ece),
        'mean_ae': float(np.mean(abs_err)),
        'p90ae': float(np.quantile(abs_err, 0.90)),
        'p95ae': float(np.quantile(abs_err, 0.95)),
    }


def build_outlier_robustness_rows(n_files, model_type, pre_revision, j_percentage, embedding_type,
                                  baseline_embedding_type, x_j, test_mask_t, test_idx,
                                  p_knn, p_amortized, y_oracle):
    train_idx = sorted(set(range(x_j.shape[0])) - set(test_idx.tolist()))
    if len(train_idx) == 0 or len(test_idx) == 0:
        return []

    x_norm = F.normalize(x_j, dim=1)
    sims = x_norm[test_idx] @ x_norm[torch.as_tensor(train_idx, device=x_j.device)].T
    d_min = 1.0 - sims.max(dim=1).values.detach().cpu().numpy()

    q50 = float(np.quantile(d_min, 0.50))
    q80 = float(np.quantile(d_min, 0.80))
    item_bins = [
        ('inlier', 0, lambda d: d <= q50),
        ('moderate', 1, lambda d: (d > q50) & (d <= q80)),
        ('outlier', 2, lambda d: d > q80),
    ]

    rows = []
    for bin_name, bin_order, predicate in item_bins:
        select = predicate(d_min)
        selected_items = test_idx[select]
        item_mask = torch.zeros((x_j.shape[0],), dtype=torch.bool, device=test_mask_t.device)
        if selected_items.size > 0:
            item_mask[selected_items] = True
        bin_mask_t = test_mask_t.clone()
        bin_mask_t[:, ~item_mask] = False

        num_items = int(select.sum())
        num_pairs = int(bin_mask_t.sum().item())
        if num_pairs == 0 or num_items == 0:
            auc_knn = np.nan
            rmse_knn = np.nan
            auc_araf = np.nan
            rmse_araf = np.nan
            knn_prob = compute_prob_metrics(p_knn, y_oracle, bin_mask_t)
            araf_prob = compute_prob_metrics(p_amortized, y_oracle, bin_mask_t)
            mean_d = np.nan
        else:
            bin_mask_np = bin_mask_t.detach().cpu().numpy().astype(bool)
            auc_knn = evaluate_auc(p_knn, y_oracle, bin_mask_t)
            rmse_knn = compute_rmse(p_knn.detach().cpu().numpy(), y_oracle.detach().cpu().numpy(), bin_mask_np)
            auc_araf = evaluate_auc(p_amortized, y_oracle, bin_mask_t)
            rmse_araf = compute_rmse(p_amortized.detach().cpu().numpy(), y_oracle.detach().cpu().numpy(), bin_mask_np)
            knn_prob = compute_prob_metrics(p_knn, y_oracle, bin_mask_t)
            araf_prob = compute_prob_metrics(p_amortized, y_oracle, bin_mask_t)
            mean_d = float(np.mean(d_min[select]))

        rows.append({
            'seed': int(RANDOM_SEED),
            'lambda_tau': float(LAMBDA_TAU),
            'n_samples': int(n_files),
            'model_type': str(model_type),
            'pre_revision': normalize_pre_revision(pre_revision),
            'j_percentage': normalize_j_percentage(j_percentage),
            'embedding_type': str(embedding_type),
            'baseline_embedding_type': normalize_baseline_embedding_type(baseline_embedding_type),
            'novelty_bin': bin_name,
            'novelty_bin_order': int(bin_order),
            'num_items': num_items,
            'num_pairs': num_pairs,
            'mean_dmin': mean_d,
            'q50_dmin': q50,
            'q80_dmin': q80,
            'auc_knn': auc_knn,
            'rmse_knn': rmse_knn,
            'brier_knn': knn_prob['brier'],
            'logloss_knn': knn_prob['log_loss'],
            'ece_knn': knn_prob['ece'],
            'mean_ae_knn': knn_prob['mean_ae'],
            'p90ae_knn': knn_prob['p90ae'],
            'p95ae_knn': knn_prob['p95ae'],
            'auc_araf': auc_araf,
            'rmse_araf': rmse_araf,
            'brier_araf': araf_prob['brier'],
            'logloss_araf': araf_prob['log_loss'],
            'ece_araf': araf_prob['ece'],
            'mean_ae_araf': araf_prob['mean_ae'],
            'p90ae_araf': araf_prob['p90ae'],
            'p95ae_araf': araf_prob['p95ae'],
        })

    return rows


def append_outlier_robustness_rows(path, rows):
    if not path or not rows:
        return

    lock = FileLock(f"{path}.lock", timeout=CSV_LOCK_TIMEOUT)
    with lock:
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path)
            except Exception:
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        df_new = pd.DataFrame(rows)
        df = pd.concat([df_old, df_new], ignore_index=True)
        dedupe_cols = [
            'seed', 'lambda_tau', 'n_samples', 'model_type', 'pre_revision',
            'j_percentage', 'embedding_type', 'baseline_embedding_type', 'novelty_bin'
        ]
        present_cols = [c for c in dedupe_cols if c in df.columns]
        if present_cols:
            df = df.drop_duplicates(subset=present_cols, keep='last')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def append_pair_efficiency_rows(path, rows):
    """Atomically append observed-pair efficiency rows."""
    if not path or not rows:
        return

    lock = FileLock(f"{path}.lock", timeout=CSV_LOCK_TIMEOUT)
    with lock:
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path)
            except Exception:
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        df_new = pd.DataFrame(rows)
        df = pd.concat([df_old, df_new], ignore_index=True)
        dedupe_cols = [
            'seed', 'lambda_tau', 'n_samples', 'model_type', 'pre_revision',
            'j_percentage', 'embedding_type', 'baseline_embedding_type'
        ]
        present_cols = [c for c in dedupe_cols if c in df.columns]
        if present_cols:
            df = df.drop_duplicates(subset=present_cols, keep='last')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


NEIGHBOR_SUPPORT_BINS = [
    ('zero', 0, 0.0, 0.0, '0%'),
    ('very_low', 1, 0.0, 0.2, '(0,20%]'),
    ('low_mid', 2, 0.2, 0.4, '(20,40%]'),
    ('mid', 3, 0.4, 0.7, '(40,70%]'),
    ('dense', 4, 0.7, 1.0, '(70,100%]'),
]


def build_neighbor_support_rows(n_files, model_type, pre_revision, j_percentage, embedding_type,
                                baseline_embedding_type, support_diag, p_knn, p_amortized,
                                y_oracle, test_mask, test_mask_t, baselines,
                                rmse_amortized, auc_amortized):
    """Summarize kNN vs ARAF inside local-neighborhood support bins."""
    coverage_rate = support_diag['coverage_rate']
    coverage_count = support_diag['coverage_count']
    fallback_mask = support_diag['fallback_mask']
    total_test_pairs = int(test_mask_t.sum().item())

    rows = []
    test_mask_np = np.asarray(test_mask, dtype=bool)
    for bin_name, bin_order, low, high, label in NEIGHBOR_SUPPORT_BINS:
        if bin_name == 'zero':
            bin_mask_t = test_mask_t & (coverage_rate <= 1e-12)
        elif high >= 1.0:
            bin_mask_t = test_mask_t & (coverage_rate > low) & (coverage_rate <= high + 1e-12)
        else:
            bin_mask_t = test_mask_t & (coverage_rate > low) & (coverage_rate <= high)

        num_pairs = int(bin_mask_t.sum().item())
        if num_pairs > 0:
            bin_mask_np = bin_mask_t.detach().cpu().numpy().astype(bool)
            auc_knn = evaluate_auc(p_knn, y_oracle, bin_mask_t)
            rmse_knn = compute_rmse(p_knn.detach().cpu().numpy(), y_oracle.detach().cpu().numpy(), bin_mask_np)
            auc_araf_bin = evaluate_auc(p_amortized, y_oracle, bin_mask_t)
            rmse_araf_bin = compute_rmse(
                p_amortized.detach().cpu().numpy(),
                y_oracle.detach().cpu().numpy(),
                bin_mask_np,
            )
            mean_count = float(coverage_count[bin_mask_t].float().mean().item())
            mean_rate = float(coverage_rate[bin_mask_t].float().mean().item())
            fallback_rate = float(fallback_mask[bin_mask_t].float().mean().item())
        else:
            auc_knn = np.nan
            rmse_knn = np.nan
            auc_araf_bin = np.nan
            rmse_araf_bin = np.nan
            mean_count = np.nan
            mean_rate = np.nan
            fallback_rate = np.nan

        rows.append({
            'seed': int(RANDOM_SEED),
            'lambda_tau': float(LAMBDA_TAU),
            'n_samples': int(n_files),
            'model_type': str(model_type),
            'pre_revision': normalize_pre_revision(pre_revision),
            'j_percentage': normalize_j_percentage(j_percentage),
            'embedding_type': str(embedding_type),
            'baseline_embedding_type': normalize_baseline_embedding_type(baseline_embedding_type),
            'support_metric': 'coverage_rate',
            'support_bin': bin_name,
            'support_bin_order': int(bin_order),
            'support_bin_label': label,
            'num_pairs': int(num_pairs),
            'pair_fraction': float(num_pairs / max(total_test_pairs, 1)),
            'mean_coverage_count': mean_count,
            'mean_coverage_rate': mean_rate,
            'fallback_rate': fallback_rate,
            'auc_knn': float(auc_knn) if not np.isnan(auc_knn) else np.nan,
            'rmse_knn': float(rmse_knn) if not np.isnan(rmse_knn) else np.nan,
            'auc_araf': float(auc_araf_bin) if not np.isnan(auc_araf_bin) else np.nan,
            'rmse_araf': float(rmse_araf_bin) if not np.isnan(rmse_araf_bin) else np.nan,
            'overall_auc_knn': float(baselines['auc_knn']),
            'overall_rmse_knn': float(baselines['rmse_knn']),
            'overall_auc_araf': float(auc_amortized),
            'overall_rmse_araf': float(rmse_amortized),
        })

    return rows


def append_neighbor_support_rows(path, rows):
    """Atomically append neighbor-support study rows."""
    if not path or not rows:
        return

    lock = FileLock(f"{path}.lock", timeout=CSV_LOCK_TIMEOUT)
    with lock:
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path)
            except Exception:
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        df_new = pd.DataFrame(rows)
        df = pd.concat([df_old, df_new], ignore_index=True)
        dedupe_cols = [
            'seed', 'lambda_tau', 'n_samples', 'model_type', 'pre_revision',
            'j_percentage', 'embedding_type', 'baseline_embedding_type', 'support_bin'
        ]
        present_cols = [c for c in dedupe_cols if c in df.columns]
        if present_cols:
            df = df.drop_duplicates(subset=present_cols, keep='last')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def compute_best_knn_metrics(y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t, x_j,
                             y_auc_oracle=None, auc_mask_t=None, knn_k=KNN_K, knn_k_values=None,
                             row_weight_t=None):
    """Evaluate one or more kNN neighbor counts and keep the best by AUC, then RMSE."""
    y_auc_oracle = y_oracle if y_auc_oracle is None else y_auc_oracle
    auc_mask_t = test_mask_t if auc_mask_t is None else auc_mask_t

    candidates = parse_knn_k_grid(knn_k_values if knn_k_values is not None else [knn_k])
    best = None
    for curr_k in candidates:
        p_knn, support_diag = compute_knn_predictions(
            y_train, train_mask_current_t, x_j, test_mask, knn_k=curr_k, row_weight_t=row_weight_t
        )
        curr = {
            'selected_knn_k': int(curr_k),
            'p_knn': p_knn,
            'support_diag': support_diag,
            'rmse_knn': compute_rmse(p_knn.cpu().numpy(), y_oracle.cpu().numpy(), test_mask),
            'auc_knn': evaluate_auc(p_knn, y_auc_oracle, auc_mask_t),
        }
        if best is None:
            best = curr
            continue

        if (
            curr['auc_knn'] > best['auc_knn'] or
            (np.isclose(curr['auc_knn'], best['auc_knn']) and curr['rmse_knn'] < best['rmse_knn'])
        ):
            best = curr

    return best


def compute_non_mirt_baseline_metrics(N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                                      model_type='beta', beta_phi=BETA_PHI, x_j=None, knn_k=KNN_K,
                                      y_auc_oracle=None, auc_mask_t=None, knn_k_values=None,
                                      row_weight_t=None):
    """Compute non-MIRT baselines for one configuration."""
    if row_weight_t is None:
        row_weight_t = torch.ones(N, device=y_train.device, dtype=torch.float32)
    # 1. Naive item-mean baseline
    weighted_mask = train_mask_current_t.float() * row_weight_t.unsqueeze(1)
    valid_counts = weighted_mask.sum(dim=0)
    item_sums = (y_train * weighted_mask).sum(dim=0)
    global_mean = masked_weighted_mean(y_train, weighted_mask)
    item_means = torch.where(valid_counts > 0, item_sums / valid_counts, global_mean)
    p_naive = item_means.unsqueeze(0).expand(N, J)

    rmse_naive = compute_rmse(p_naive.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    y_auc_oracle = y_oracle if y_auc_oracle is None else y_auc_oracle
    auc_mask_t = test_mask_t if auc_mask_t is None else auc_mask_t

    auc_naive = evaluate_auc(p_naive, y_auc_oracle, auc_mask_t)

    # 2. Rasch (1PL)
    p_rasch = train_rasch(N, J, y_train, train_mask_current_t, row_weight_t=row_weight_t)
    rmse_rasch = compute_rmse(p_rasch.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_rasch = evaluate_auc(p_rasch, y_auc_oracle, auc_mask_t)

    # 3. Rasch (2PL)
    p_2pl = train_2pl(N, J, y_train, train_mask_current_t, row_weight_t=row_weight_t)
    rmse_2pl = compute_rmse(p_2pl.cpu().numpy(), y_oracle.cpu().numpy(), test_mask)
    auc_2pl = evaluate_auc(p_2pl, y_auc_oracle, auc_mask_t)

    # 4. Embedding kNN cold-start baseline (predict held-out items from nearest train items)
    # Uses item embeddings only and observed user responses on train items.
    knn_result = compute_best_knn_metrics(
        y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t, x_j,
        y_auc_oracle=y_auc_oracle, auc_mask_t=auc_mask_t, knn_k=knn_k, knn_k_values=knn_k_values,
        row_weight_t=row_weight_t,
    )
    rmse_knn = knn_result['rmse_knn']
    auc_knn = knn_result['auc_knn']

    return {
        'rmse_naive': rmse_naive,
        'rmse_rasch': rmse_rasch,
        'rmse_2pl': rmse_2pl,
        'auc_naive': auc_naive,
        'auc_rasch': auc_rasch,
        'auc_2pl': auc_2pl,
        'rmse_knn': rmse_knn,
        'auc_knn': auc_knn,
        'selected_knn_k': int(knn_result['selected_knn_k']),
    }


def build_mirt_validation_masks(y_train, train_mask_current_t, validation_fraction=MIRT_VALIDATION_FRACTION,
                                min_pairs=MIRT_VALIDATION_MIN_PAIRS):
    """Split observed training entries into fit/validation masks for MIRT selection."""
    observed_idx = torch.nonzero(train_mask_current_t, as_tuple=False)
    n_observed = int(observed_idx.shape[0])
    if n_observed < 8:
        return train_mask_current_t.clone(), None

    n_val = max(int(round(n_observed * validation_fraction)), int(min_pairs))
    n_val = min(n_val, n_observed - 4)
    if n_val < 2:
        return train_mask_current_t.clone(), None

    observed_targets = y_train[train_mask_current_t]
    pos_pool = torch.nonzero(observed_targets > 0.5, as_tuple=False).squeeze(1).cpu().numpy()
    neg_pool = torch.nonzero(observed_targets <= 0.5, as_tuple=False).squeeze(1).cpu().numpy()
    rng = np.random.default_rng(RANDOM_SEED + 2027)

    chosen = []
    if len(pos_pool) > 1 and len(neg_pool) > 1:
        n_pos = max(1, int(round(n_val * len(pos_pool) / n_observed)))
        n_pos = min(n_pos, len(pos_pool) - 1)
        n_neg = max(1, n_val - n_pos)
        n_neg = min(n_neg, len(neg_pool) - 1)

        while n_pos + n_neg > n_val:
            if n_pos > n_neg and n_pos > 1:
                n_pos -= 1
            elif n_neg > 1:
                n_neg -= 1
            else:
                break

        remaining = n_val - (n_pos + n_neg)
        pos_extra = max(0, (len(pos_pool) - 1) - n_pos)
        neg_extra = max(0, (len(neg_pool) - 1) - n_neg)
        add_pos = min(remaining, pos_extra)
        n_pos += add_pos
        remaining -= add_pos
        n_neg += min(remaining, neg_extra)

        chosen.append(rng.choice(pos_pool, size=n_pos, replace=False))
        chosen.append(rng.choice(neg_pool, size=n_neg, replace=False))
    else:
        population = np.arange(n_observed)
        keep = max(4, n_observed - n_val)
        chosen.append(rng.choice(population, size=n_observed - keep, replace=False))

    val_flat_idx = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if val_flat_idx.size == 0:
        return train_mask_current_t.clone(), None

    val_mask_t = torch.zeros_like(train_mask_current_t, dtype=torch.bool)
    val_coords = observed_idx[torch.as_tensor(val_flat_idx, device=observed_idx.device, dtype=torch.long)]
    val_mask_t[val_coords[:, 0], val_coords[:, 1]] = True

    fit_mask_t = train_mask_current_t & (~val_mask_t)
    if not fit_mask_t.any():
        return train_mask_current_t.clone(), None
    return fit_mask_t, val_mask_t


def compute_single_mirt_metrics(N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                                model_type='beta', beta_phi=BETA_PHI, mirt_dim=K_MODEL,
                                y_auc_oracle=None, auc_mask_t=None, row_weight_t=None):
    """Train one standalone MIRT baseline at a specific latent dimension."""
    if row_weight_t is None:
        row_weight_t = torch.ones(N, device=y_train.device, dtype=torch.float32)
    fit_mask_t, val_mask_t = build_mirt_validation_masks(y_train, train_mask_current_t)
    mirt_model = MIRTModel(N, J, mirt_dim).to(device)
    mirt_optimizer = optim.AdamW(mirt_model.parameters(), lr=0.01, weight_decay=0.1)
    mirt_model.train()

    best_val_auc = float('-inf')
    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    best_p_mirt = None
    best_state = None
    y_train_np = y_train.detach().cpu().numpy()
    y_oracle_np = y_oracle.detach().cpu().numpy()
    val_mask_np = val_mask_t.detach().cpu().numpy() if val_mask_t is not None else None

    for _ in range(EPOCHS // 2):
        mirt_optimizer.zero_grad()
        p_mirt = mirt_model()
        p_m_clamp = p_mirt[fit_mask_t].clamp(1e-6, 1 - 1e-6)
        y_m_clamp = y_train[fit_mask_t]
        fit_weights = row_weight_t.unsqueeze(1).expand_as(y_train)[fit_mask_t]

        if model_type == 'beta':
            y_m_clamp = y_m_clamp.clamp(1e-6, 1 - 1e-6)
            dist = torch.distributions.Beta(p_m_clamp * beta_phi, (1 - p_m_clamp) * beta_phi)
            loss = masked_weighted_mean(-dist.log_prob(y_m_clamp), fit_weights)
        else:
            dist = torch.distributions.Bernoulli(probs=p_m_clamp)
            loss = masked_weighted_mean(-dist.log_prob(y_m_clamp), fit_weights)

        loss.backward()
        mirt_optimizer.step()

        with torch.no_grad():
            p_mirt_eval = mirt_model().detach()
            if val_mask_t is not None:
                curr_val_rmse = compute_rmse(p_mirt_eval.cpu().numpy(), y_train_np, val_mask_np)
                curr_val_auc = evaluate_auc(p_mirt_eval, y_train, val_mask_t)
            else:
                curr_val_rmse = compute_rmse(p_mirt_eval.cpu().numpy(), y_train_np, fit_mask_t.detach().cpu().numpy())
                curr_val_auc = evaluate_auc(p_mirt_eval, y_train, fit_mask_t)

            is_better = (
                curr_val_auc > best_val_auc or
                (np.isclose(curr_val_auc, best_val_auc) and curr_val_rmse < best_val_rmse)
            )
            if is_better:
                best_val_auc = float(curr_val_auc)
                best_val_rmse = float(curr_val_rmse)
                best_test_rmse = compute_rmse(p_mirt_eval.cpu().numpy(), y_oracle_np, test_mask)
                best_p_mirt = p_mirt.clone()
                best_state = {k: v.detach().cpu().clone() for k, v in mirt_model.state_dict().items()}

    if best_p_mirt is None:
        with torch.no_grad():
            best_p_mirt = mirt_model().detach().clone()
            best_test_rmse = compute_rmse(best_p_mirt.cpu().numpy(), y_oracle_np, test_mask)
            if val_mask_t is not None:
                best_val_rmse = compute_rmse(best_p_mirt.cpu().numpy(), y_train_np, val_mask_np)
                best_val_auc = evaluate_auc(best_p_mirt, y_train, val_mask_t)
            else:
                fit_mask_np = fit_mask_t.detach().cpu().numpy()
                best_val_rmse = compute_rmse(best_p_mirt.cpu().numpy(), y_train_np, fit_mask_np)
                best_val_auc = evaluate_auc(best_p_mirt, y_train, fit_mask_t)
            best_state = {k: v.detach().cpu().clone() for k, v in mirt_model.state_dict().items()}

    y_auc_oracle = y_oracle if y_auc_oracle is None else y_auc_oracle
    auc_mask_t = test_mask_t if auc_mask_t is None else auc_mask_t
    auc_mirt = evaluate_auc(best_p_mirt, y_auc_oracle, auc_mask_t)

    return {
        'rmse_mirt': float(best_test_rmse),
        'auc_mirt': float(auc_mirt),
        'val_rmse_mirt': float(best_val_rmse),
        'val_auc_mirt': float(best_val_auc),
        'mirt_dim': int(mirt_dim),
        'mirt_state': best_state,
    }


def compute_baseline_metrics(N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
                             model_type='beta', beta_phi=BETA_PHI, x_j=None, knn_k=KNN_K,
                             mirt_dim=K_MODEL, row_weight_t=None):
    """Compute naive + Rasch + 2PL + standalone MIRT baselines for one configuration."""
    results = compute_non_mirt_baseline_metrics(
        N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
        model_type=model_type, beta_phi=beta_phi, x_j=x_j, knn_k=knn_k, row_weight_t=row_weight_t
    )
    results.update(
        compute_single_mirt_metrics(
            N, J, y_train, train_mask_current_t, y_oracle, test_mask, test_mask_t,
            model_type=model_type, beta_phi=beta_phi, mirt_dim=mirt_dim, row_weight_t=row_weight_t
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


def required_non_mirt_metric_cols(baseline_profile=BASELINE_PROFILE_FULL):
    if baseline_profile in {BASELINE_PROFILE_KNN_MIRT, BASELINE_PROFILE_KNN_ONLY}:
        return ['rmse_knn', 'auc_knn']
    return NON_MIRT_METRIC_COLS


def required_baseline_metric_cols(baseline_profile=BASELINE_PROFILE_FULL):
    cols = list(required_non_mirt_metric_cols(baseline_profile))
    if baseline_profile != BASELINE_PROFILE_KNN_ONLY:
        cols.extend(['rmse_mirt', 'auc_mirt'])
    return cols


def _baseline_row_matches_mirt_request(row, mirt_dim_min, mirt_dim_max, baseline_profile=BASELINE_PROFILE_FULL):
    if not _row_has_complete_metrics(row, required_baseline_metric_cols(baseline_profile)):
        return False
    if baseline_profile == BASELINE_PROFILE_KNN_ONLY:
        return True
    if _optional_int(row.get('mirt_selection_version')) != MIRT_SELECTION_VERSION:
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
    payload = {}
    for k in BASELINE_METRIC_COLS:
        v = row.get(k, np.nan)
        payload[k] = float(v) if not pd.isna(v) else np.nan
    selected_knn_k = _optional_int(row.get('selected_knn_k'))
    if selected_knn_k is not None:
        payload['selected_knn_k'] = selected_knn_k
    selected_dim = _optional_int(row.get('selected_mirt_dim'))
    if selected_dim is not None:
        payload['selected_mirt_dim'] = selected_dim
    payload['mirt_selection_version'] = _optional_int(row.get('mirt_selection_version'))
    return payload


def select_best_mirt_result(results):
    """Pick the best MIRT sweep candidate by validation AUC, then validation RMSE, then smaller dimension."""
    return max(
        results,
        key=lambda r: (
            float(r['val_auc_mirt']),
            -float(r['val_rmse_mirt']),
            -int(r['mirt_dim']),
        )
    )


def get_or_compute_baselines(n_files, all_dfs, global_shared_indices, data, model_type='beta', beta_phi=BETA_PHI,
                             baseline_output=DEFAULT_BASELINE_OUTPUT, pre_revision='none', j_percentage=1.0,
                             allow_compute=True, quiet=False, mirt_dim_min=K_MODEL, mirt_dim_max=K_MODEL,
                             mirt_sweep_output=DEFAULT_MIRT_SWEEP_OUTPUT, embedding_type=None,
                             baseline_embedding_type=None, train_retention=1.0, knn_k=KNN_K,
                             baseline_profile=BASELINE_PROFILE_FULL, knn_k_grid=None):
    """Fetch baselines from cache, or compute and persist once per unique configuration."""
    actual_embedding_type = normalize_baseline_embedding_type(embedding_type)
    baseline_embedding_type = normalize_baseline_embedding_type(
        baseline_embedding_type if baseline_embedding_type is not None else actual_embedding_type
    )
    baseline_key = {
        'seed': int(RANDOM_SEED),
        'model_type': str(model_type),
        'n_samples': int(n_files),
        'pre_revision': normalize_pre_revision(pre_revision),
        'j_percentage': normalize_j_percentage(j_percentage),
        'train_retention': normalize_train_retention(train_retention),
        'baseline_embedding_type': baseline_embedding_type,
        'pre_population_match': data.get('pre_population_match', PRE_POPULATION_MATCH_NONE),
        'pre_population_match_version': data.get('pre_population_match_version', np.nan),
    }
    knn_k_grid = parse_knn_k_grid(knn_k_grid)

    cached = try_get_cached_baseline(
        baseline_output,
        baseline_key,
        mirt_dim_min=mirt_dim_min,
        mirt_dim_max=mirt_dim_max,
        baseline_profile=baseline_profile,
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
    expected_non_mirt_cols = required_non_mirt_metric_cols(baseline_profile)
    if existing_row is not None and _row_has_complete_metrics(existing_row, expected_non_mirt_cols):
        non_mirt_metrics = {k: float(existing_row[k]) for k in expected_non_mirt_cols}

    if data.get('cross_revision_post_binary', False):
        y_support, support_mask_current_t = build_cross_revision_support_targets(
            data,
            train_retention=1.0,
        )
        run_view = prune_embedding_only_train_columns(
            y_support,
            support_mask_current_t,
            data['y_oracle'],
            data['test_mask'],
            data['test_mask_t'],
            x_j=data.get('x_j'),
            y_auc_oracle=data['y_auc_oracle'],
            auc_mask_t=data['auc_test_mask_t'],
            task_ids=data.get('task_ids'),
        )
        N = run_view['N']
        J = run_view['J']
        y_train = run_view['y_train']
        train_mask_current_t = run_view['train_mask_t']
        row_weight_t = torch.ones(N, device=device)
    else:
        train_indices = data.get('train_global_shared_indices', global_shared_indices)
        N, J, y_train, train_mask_current_t, row_weight_t = build_training_targets(
            n_files, all_dfs, train_indices, data, model_type=model_type, quiet=quiet,
            train_retention=train_retention
        )
        run_view = prune_embedding_only_train_columns(
            y_train,
            train_mask_current_t,
            data['y_oracle'],
            data['test_mask'],
            data['test_mask_t'],
            x_j=data.get('x_j'),
            y_auc_oracle=data['y_auc_oracle'],
            auc_mask_t=data['auc_test_mask_t'],
            task_ids=data.get('task_ids'),
        )
        N = run_view['N']
        J = run_view['J']
        y_train = run_view['y_train']
        train_mask_current_t = run_view['train_mask_t']

    if non_mirt_metrics is None:
        if baseline_embedding_type != actual_embedding_type:
            raise RuntimeError(
                f"Missing baseline cache row for {baseline_key}. "
                f"Baseline computation requires embeddings '{baseline_embedding_type}', "
                f"but this run loaded '{actual_embedding_type}'. "
                f"Prime the cache first with --baseline-only --embedding-type {baseline_embedding_type} "
                f"--baseline-embedding-type {baseline_embedding_type}."
            )
        if baseline_profile in {BASELINE_PROFILE_KNN_MIRT, BASELINE_PROFILE_KNN_ONLY}:
            knn_result = compute_best_knn_metrics(
                y_train, train_mask_current_t,
                run_view['y_oracle'], run_view['test_mask'], run_view['test_mask_t'], run_view.get('x_j'),
                y_auc_oracle=run_view['y_auc_oracle'], auc_mask_t=run_view['auc_mask_t'],
                knn_k=knn_k,
                knn_k_values=[knn_k] if baseline_profile == BASELINE_PROFILE_KNN_ONLY else knn_k_grid,
                row_weight_t=row_weight_t,
            )
            non_mirt_metrics = {
                'rmse_naive': np.nan,
                'rmse_rasch': np.nan,
                'rmse_2pl': np.nan,
                'auc_naive': np.nan,
                'auc_rasch': np.nan,
                'auc_2pl': np.nan,
                'rmse_knn': knn_result['rmse_knn'],
                'auc_knn': knn_result['auc_knn'],
                'selected_knn_k': int(knn_result['selected_knn_k']),
            }
        else:
            non_mirt_metrics = compute_non_mirt_baseline_metrics(
                N, J, y_train, train_mask_current_t,
                run_view['y_oracle'], run_view['test_mask'], run_view['test_mask_t'],
                model_type=model_type, beta_phi=beta_phi, x_j=run_view.get('x_j'), knn_k=knn_k,
                y_auc_oracle=run_view['y_auc_oracle'], auc_mask_t=run_view['auc_mask_t'],
                knn_k_values=knn_k_grid, row_weight_t=row_weight_t,
            )

    best_mirt = None
    if baseline_profile != BASELINE_PROFILE_KNN_ONLY:
        mirt_results = []
        for mirt_dim in range(int(mirt_dim_min), int(mirt_dim_max) + 1):
            sweep_key = baseline_key.copy()
            sweep_key['mirt_dim'] = int(mirt_dim)
            cached_mirt = try_get_cached_mirt_sweep_row(mirt_sweep_output, sweep_key)
            if cached_mirt is not None:
                mirt_results.append(cached_mirt)
                continue

            computed_mirt = compute_single_mirt_metrics(
                N, J, y_train, train_mask_current_t,
                run_view['y_oracle'], run_view['test_mask'], run_view['test_mask_t'],
                model_type=model_type, beta_phi=beta_phi, mirt_dim=mirt_dim,
                y_auc_oracle=run_view['y_auc_oracle'], auc_mask_t=run_view['auc_mask_t'],
                row_weight_t=row_weight_t,
            )
            append_mirt_sweep_row(
                mirt_sweep_output,
                {
                    **sweep_key,
                    'rmse_mirt': float(computed_mirt['rmse_mirt']),
                    'auc_mirt': float(computed_mirt['auc_mirt']),
                    'val_rmse_mirt': float(computed_mirt['val_rmse_mirt']),
                    'val_auc_mirt': float(computed_mirt['val_auc_mirt']),
                }
            )
            mirt_results.append(computed_mirt)

        best_mirt = select_best_mirt_result(mirt_results)

    baseline_row = baseline_key.copy()
    baseline_row['agent_batch_size'] = compute_agent_batch_size(
        baseline_key['pre_revision'], baseline_key['n_samples']
    )
    baseline_row['pre_population_match'] = data.get('pre_population_match', PRE_POPULATION_MATCH_NONE)
    baseline_row['pre_population_match_version'] = data.get('pre_population_match_version', np.nan)
    baseline_row['effective_pre_weight_sum'] = float(data.get('effective_pre_weight_sum', np.nan))
    for col, value in non_mirt_metrics.items():
        baseline_row[col] = float(value)
    if best_mirt is not None:
        baseline_row['rmse_mirt'] = float(best_mirt['rmse_mirt'])
        baseline_row['auc_mirt'] = float(best_mirt['auc_mirt'])
        baseline_row['selected_mirt_dim'] = int(best_mirt['mirt_dim'])
        baseline_row['mirt_sweep_min'] = int(mirt_dim_min)
        baseline_row['mirt_sweep_max'] = int(mirt_dim_max)
        baseline_row['mirt_selection_version'] = int(MIRT_SELECTION_VERSION)
    else:
        baseline_row['rmse_mirt'] = np.nan
        baseline_row['auc_mirt'] = np.nan
        baseline_row['selected_mirt_dim'] = np.nan
        baseline_row['mirt_sweep_min'] = np.nan
        baseline_row['mirt_sweep_max'] = np.nan
        baseline_row['mirt_selection_version'] = np.nan
    append_baseline_row(baseline_output, baseline_row)

    cached_now = _baseline_payload_from_row(baseline_row)
    return cached_now, (best_mirt.get('mirt_state') if best_mirt is not None else None)


def run_experiment(n_files, all_dfs, global_shared_indices, data, model_type='beta',
                   beta_phi=BETA_PHI, no_tau=False, quiet=False, embedding_type=None,
                   baseline_output=DEFAULT_BASELINE_OUTPUT, pre_revision='none', j_percentage=1.0,
                   baseline_embedding_type=None, pair_efficiency_output=None,
                   neighbor_support_output=None, support_thinning_output=None,
                   outlier_robustness_output=None,
                   train_retention=1.0, knn_k=KNN_K,
                   knn_k_grid=KNN_K_GRID,
                   allow_compute_baselines=True, mirt_dim_min=K_MODEL, mirt_dim_max=K_MODEL,
                   mirt_sweep_output=DEFAULT_MIRT_SWEEP_OUTPUT,
                   cross_revision_araf_mode='transfer',
                   baseline_profile=BASELINE_PROFILE_FULL):
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
    test_idx = data['test_idx']
    run_task_ids = list(data.get('task_ids', []))
    run_train_item_ids = list(data.get('train_item_ids', []))
    run_test_item_ids = list(data.get('test_item_ids', []))
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
        embedding_type=embedding_type,
        baseline_embedding_type=baseline_embedding_type,
        train_retention=train_retention,
        knn_k=knn_k,
        knn_k_grid=knn_k_grid,
        baseline_profile=baseline_profile,
    )

    rmse_2pl = float(baselines.get('rmse_2pl', np.nan))
    best_mirt_rmse = float(baselines.get('rmse_mirt', np.nan))
    auc_2pl = float(baselines.get('auc_2pl', np.nan))
    auc_mirt = float(baselines.get('auc_mirt', np.nan))
    selected_mirt_dim = int(baselines.get('selected_mirt_dim', K_MODEL)) if not pd.isna(baselines.get('selected_mirt_dim', K_MODEL)) else K_MODEL

    if embedding_type == 'rasch_2pl':
        return {
            'n_samples': n_files,
            'model_type': model_type,
            'seed': RANDOM_SEED,
            'lambda_tau': LAMBDA_TAU,
            'pre_population_match': data.get('pre_population_match', PRE_POPULATION_MATCH_NONE),
            'pre_population_match_version': data.get('pre_population_match_version', np.nan),
            'effective_pre_weight_sum': float(data.get('effective_pre_weight_sum', np.nan)),
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
            'pre_population_match': data.get('pre_population_match', PRE_POPULATION_MATCH_NONE),
            'pre_population_match_version': data.get('pre_population_match_version', np.nan),
            'effective_pre_weight_sum': float(data.get('effective_pre_weight_sum', np.nan)),
            'rmse_amortized': best_mirt_rmse,
            'auc_amortized': auc_mirt,
            'active_dims': selected_mirt_dim,
            'active_indices': str(list(range(selected_mirt_dim))),
            'tau_values': str([1.0] * selected_mirt_dim),
            'model_state': model_state,
            'final_state': model_state
        }

    if data.get('cross_revision_post_binary', False):
        mode = str(cross_revision_araf_mode).strip().lower()
        if mode == 'post':
            y_support, support_mask_current_t = build_cross_revision_support_targets(
                data,
                train_retention=train_retention,
            )
            run_view = prune_embedding_only_train_columns(
                y_support,
                support_mask_current_t,
                y_oracle,
                test_mask,
                test_mask_t,
                x_j=x_j,
                y_auc_oracle=data['y_auc_oracle'],
                auc_mask_t=data['auc_test_mask_t'],
                task_ids=data.get('task_ids'),
            )
            N = run_view['N']
            J = run_view['J']
            y_support = run_view['y_train']
            support_mask_current_t = run_view['train_mask_t']
            y_oracle = run_view['y_oracle']
            test_mask = run_view['test_mask']
            test_mask_t = run_view['test_mask_t']
            x_j = run_view['x_j']
            test_idx = run_view['test_idx']
            run_task_ids = run_view['task_ids'] or run_task_ids
            run_train_item_ids = run_view['train_item_ids'] or run_train_item_ids
            run_test_item_ids = run_view['test_item_ids'] or run_test_item_ids
            observed_train_pairs = int(support_mask_current_t.sum().item())

            model = AmortizedIRTModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.5, no_tau=no_tau).to(device)
            final_rmse, final_state = train_amortized_irt(
                model,
                y_support,
                support_mask_current_t,
                y_oracle,
                test_mask,
                model_type=model_type,
                beta_phi=beta_phi,
                epochs=EPOCHS,
                lambda_tau=LAMBDA_TAU,
                quiet=quiet,
                row_weight_t=torch.ones(N, device=device),
            )

            model.eval()
            with torch.no_grad():
                model.load_state_dict({k: v.to(device) for k, v in final_state.items()})
                p_amortized = model()
                auc_amortized = evaluate_auc(p_amortized, run_view['y_auc_oracle'], run_view['auc_mask_t'])

                tau_val = model.get_tau()
                active_mask = tau_val > TAU_THRESHOLD
                active_dims = active_mask.sum().item()
                active_dim_indices = torch.nonzero(active_mask).squeeze().cpu().tolist()
                if isinstance(active_dim_indices, int):
                    active_dim_indices = [active_dim_indices]
        else:
            train_indices = data.get('train_global_shared_indices', global_shared_indices)
            support_retention = train_retention if mode == 'post_support_transfer' else 1.0
            pre_retention = 1.0 if mode == 'post_support_transfer' else train_retention
            N_pre, _, y_train_pre, train_mask_pre_t, row_weight_pre_t = build_training_targets(
                n_files,
                all_dfs,
                train_indices,
                data,
                model_type=model_type,
                quiet=quiet,
                train_retention=pre_retention,
            )
            pre_run_view = prune_embedding_only_train_columns(
                y_train_pre,
                train_mask_pre_t,
                y_oracle,
                test_mask,
                test_mask_t,
                x_j=x_j,
                y_auc_oracle=data['y_auc_oracle'],
                auc_mask_t=data['auc_test_mask_t'],
                task_ids=data.get('task_ids'),
            )
            y_support, support_mask_current_t = build_cross_revision_support_targets(
                data,
                train_retention=support_retention,
            )
            support_run_view = prune_embedding_only_train_columns(
                y_support,
                support_mask_current_t,
                y_oracle,
                test_mask,
                test_mask_t,
                x_j=x_j,
                y_auc_oracle=data['y_auc_oracle'],
                auc_mask_t=data['auc_test_mask_t'],
                task_ids=data.get('task_ids'),
            )
            N_pre = pre_run_view['N']
            y_train_pre = pre_run_view['y_train']
            train_mask_pre_t = pre_run_view['train_mask_t']
            N = support_run_view['N']
            J = support_run_view['J']
            y_support = support_run_view['y_train']
            support_mask_current_t = support_run_view['train_mask_t']
            y_oracle = support_run_view['y_oracle']
            test_mask = support_run_view['test_mask']
            test_mask_t = support_run_view['test_mask_t']
            x_j = support_run_view['x_j']
            test_idx = support_run_view['test_idx']
            run_task_ids = support_run_view['task_ids'] or run_task_ids
            run_train_item_ids = support_run_view['train_item_ids'] or run_train_item_ids
            run_test_item_ids = support_run_view['test_item_ids'] or run_test_item_ids
            observed_train_pairs = int(
                support_mask_current_t.sum().item()
                if mode == 'post_support_transfer'
                else train_mask_pre_t.sum().item()
            )

            pre_model = AmortizedIRTModel(N_pre, J, K_MODEL, embedding_dim, x_j, dropout=0.5, no_tau=no_tau).to(device)
            _, pre_state = train_amortized_irt(
                pre_model,
                y_train_pre,
                train_mask_pre_t,
                y_train_pre,
                train_mask_pre_t.detach().cpu().numpy().astype(bool),
                model_type=model_type,
                beta_phi=beta_phi,
                epochs=EPOCHS,
                lambda_tau=LAMBDA_TAU,
                quiet=quiet,
                row_weight_t=row_weight_pre_t,
            )

            model = AmortizedIRTModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.0, no_tau=no_tau).to(device)
            state_dict = model.state_dict()
            for key, value in pre_state.items():
                if key in {'theta', 'theta_bias'}:
                    continue
                if key in state_dict and tuple(state_dict[key].shape) == tuple(value.shape):
                    state_dict[key] = value.to(device)
            model.load_state_dict(state_dict)
            p_amortized, final_state = adapt_amortized_irt_users(
                model,
                y_support,
                support_mask_current_t,
                model_type=model_type,
                beta_phi=beta_phi,
                quiet=quiet,
                row_weight_t=torch.ones(N, device=device),
            )
            final_rmse = compute_rmse(
                p_amortized.detach().cpu().numpy(),
                y_oracle.detach().cpu().numpy(),
                test_mask,
            )
            auc_amortized = evaluate_auc(p_amortized, support_run_view['y_auc_oracle'], support_run_view['auc_mask_t'])
            tau_val = model.get_tau()
            active_mask = tau_val > TAU_THRESHOLD
            active_dims = active_mask.sum().item()
            active_dim_indices = torch.nonzero(active_mask).squeeze().cpu().tolist()
            if isinstance(active_dim_indices, int):
                active_dim_indices = [active_dim_indices]
    else:
        train_indices = data.get('train_global_shared_indices', global_shared_indices)
        _, _, y_train, train_mask_current_t, train_row_weight_t = build_training_targets(
            n_files, all_dfs, train_indices, data, model_type=model_type, quiet=quiet,
            train_retention=train_retention
        )
        run_view = prune_embedding_only_train_columns(
            y_train,
            train_mask_current_t,
            y_oracle,
            test_mask,
            test_mask_t,
            x_j=x_j,
            y_auc_oracle=data['y_auc_oracle'],
            auc_mask_t=data['auc_test_mask_t'],
            task_ids=data.get('task_ids'),
        )
        N = run_view['N']
        J = run_view['J']
        y_train = run_view['y_train']
        train_mask_current_t = run_view['train_mask_t']
        y_oracle = run_view['y_oracle']
        test_mask = run_view['test_mask']
        test_mask_t = run_view['test_mask_t']
        x_j = run_view['x_j']
        test_idx = run_view['test_idx']
        run_task_ids = run_view['task_ids'] or run_task_ids
        run_train_item_ids = run_view['train_item_ids'] or run_train_item_ids
        run_test_item_ids = run_view['test_item_ids'] or run_test_item_ids
        observed_train_pairs = int(train_mask_current_t.sum().item())

        model = AmortizedIRTModel(N, J, K_MODEL, embedding_dim, x_j, dropout=0.5, no_tau=no_tau).to(device)
        final_rmse, final_state = train_amortized_irt(model, y_train, train_mask_current_t, y_oracle, test_mask,
                                         model_type=model_type, beta_phi=beta_phi,
                                         epochs=EPOCHS, lambda_tau=LAMBDA_TAU, quiet=quiet,
                                         row_weight_t=train_row_weight_t)

        model.eval()
        with torch.no_grad():
            model.load_state_dict({k: v.to(device) for k, v in final_state.items()})
            p_amortized = model()
            auc_amortized = evaluate_auc(p_amortized, run_view['y_auc_oracle'], run_view['auc_mask_t'])

            tau_val = model.get_tau()
            active_mask = tau_val > TAU_THRESHOLD
            active_dims = active_mask.sum().item()

            active_dim_indices = torch.nonzero(active_mask).squeeze().cpu().tolist()
            if isinstance(active_dim_indices, int):
                active_dim_indices = [active_dim_indices]

    neighbor_support_rows = []
    p_knn = None
    if neighbor_support_output:
        if data.get('cross_revision_post_binary', False):
            y_knn, knn_mask_t = build_cross_revision_support_targets(data, train_retention=1.0)
            knn_view = prune_embedding_only_train_columns(
                y_knn,
                knn_mask_t,
                y_oracle,
                test_mask,
                test_mask_t,
                x_j=x_j,
                y_auc_oracle=run_view['y_auc_oracle'] if 'run_view' in locals() else data['y_auc_oracle'],
                auc_mask_t=run_view['auc_mask_t'] if 'run_view' in locals() else data['auc_test_mask_t'],
                task_ids=run_task_ids,
            )
            y_knn = knn_view['y_train']
            knn_mask_t = knn_view['train_mask_t']
            knn_row_weight_t = torch.ones(int(y_knn.shape[0]), device=device)
        else:
            y_knn, knn_mask_t = y_train, train_mask_current_t
            knn_row_weight_t = train_row_weight_t
        p_knn, support_diag = compute_knn_predictions(
            y_knn, knn_mask_t, x_j, test_mask, knn_k=knn_k, row_weight_t=knn_row_weight_t
        )
        neighbor_support_rows = build_neighbor_support_rows(
            n_files,
            model_type,
            pre_revision,
            j_percentage,
            embedding_type,
            baseline_embedding_type,
            support_diag,
            p_knn,
            p_amortized,
            y_oracle,
            test_mask,
            data['auc_test_mask_t'],
            baselines,
            final_rmse,
            auc_amortized,
        )

    outlier_robustness_rows = []
    if outlier_robustness_output:
        if p_knn is None:
            if data.get('cross_revision_post_binary', False):
                y_knn, knn_mask_t = build_cross_revision_support_targets(data, train_retention=1.0)
                knn_view = prune_embedding_only_train_columns(
                    y_knn,
                    knn_mask_t,
                    y_oracle,
                    test_mask,
                    test_mask_t,
                    x_j=x_j,
                    y_auc_oracle=run_view['y_auc_oracle'] if 'run_view' in locals() else data['y_auc_oracle'],
                    auc_mask_t=run_view['auc_mask_t'] if 'run_view' in locals() else data['auc_test_mask_t'],
                    task_ids=run_task_ids,
                )
                y_knn = knn_view['y_train']
                knn_mask_t = knn_view['train_mask_t']
                knn_row_weight_t = torch.ones(int(y_knn.shape[0]), device=device)
            else:
                y_knn, knn_mask_t = y_train, train_mask_current_t
                knn_row_weight_t = train_row_weight_t
            p_knn, _ = compute_knn_predictions(y_knn, knn_mask_t, x_j, test_mask, knn_k=knn_k, row_weight_t=knn_row_weight_t)
        outlier_robustness_rows = build_outlier_robustness_rows(
            n_files,
            model_type,
            pre_revision,
            j_percentage,
            embedding_type,
            baseline_embedding_type,
            x_j,
            data['auc_test_mask_t'],
            test_idx,
            p_knn,
            p_amortized,
            data['y_auc_oracle'],
        )

    return {
        'n_samples': n_files,
        'model_type': model_type,
        'seed': RANDOM_SEED,
        'lambda_tau': LAMBDA_TAU,
        'embedding_protocol': data.get('embedding_protocol', embedding_metadata_for_type(embedding_type)[0]),
        'embedding_fit_scope': data.get('embedding_fit_scope', embedding_metadata_for_type(embedding_type)[1]),
        'embedding_fit_item_count': int(data.get('embedding_fit_item_count', 0)),
        'task_ids': run_task_ids,
        'train_item_ids': run_train_item_ids,
        'test_item_ids': run_test_item_ids,
        'pre_population_match': data.get('pre_population_match', PRE_POPULATION_MATCH_NONE),
        'pre_population_match_version': data.get('pre_population_match_version', np.nan),
        'effective_pre_weight_sum': float(data.get('effective_pre_weight_sum', np.nan)),
        'observed_train_pairs': observed_train_pairs,
        'rmse_amortized': final_rmse,
        'auc_amortized': auc_amortized,
        'active_dims': active_dims,
        'active_indices': str(active_dim_indices),
        'tau_values': str(tau_val.cpu().tolist()),
        'pair_efficiency_rows': (
            [build_pair_efficiency_row(
                n_files,
                model_type,
                pre_revision,
                j_percentage,
                embedding_type,
                baseline_embedding_type,
                observed_train_pairs,
                baselines,
                final_rmse,
                auc_amortized,
            )] if pair_efficiency_output else []
        ),
        'neighbor_support_rows': neighbor_support_rows,
        'support_thinning_rows': (
            [build_support_thinning_row(
                n_files,
                model_type,
                pre_revision,
                j_percentage,
                embedding_type,
                baseline_embedding_type,
                knn_k,
                train_retention,
                observed_train_pairs,
                baselines,
                final_rmse,
                auc_amortized,
            )] if support_thinning_output else []
        ),
        'outlier_robustness_rows': outlier_robustness_rows,
        'model_state': final_state,
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


def parse_explicit_n_samples(arg):
    """Parse --n-samples without loading data when no 'all'/'max' expansion is needed."""
    arg = str(arg).strip()
    if not arg or arg in {'all', 'max', '1,all'}:
        return None

    result = []
    for part in arg.split(','):
        part = part.strip()
        if not part:
            continue
        if part in {'all', 'max'}:
            return None
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))

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


def normalize_train_retention(value):
    """Normalize train_retention for reliable float comparisons in CSV cache."""
    return float(f"{float(value):.6f}")


def normalize_baseline_embedding_type(value):
    """Normalize baseline embedding type and preserve PCA for legacy rows."""
    if value is None or pd.isna(value):
        return 'pca'
    v = str(value).strip().lower()
    return v if v and v != 'nan' else 'pca'


def normalize_pre_population_match(value):
    if value is None or pd.isna(value):
        return PRE_POPULATION_MATCH_NONE
    v = str(value).strip().lower()
    return v if v else PRE_POPULATION_MATCH_NONE


def normalize_pre_population_match_version(value, pre_population_match=None):
    match = normalize_pre_population_match(pre_population_match)
    if match == PRE_POPULATION_MATCH_NONE:
        return np.nan
    if value is None or pd.isna(value):
        return int(PRE_POPULATION_MATCH_VERSION)
    return int(value)


def parse_knn_k_grid(values):
    """Normalize a kNN sweep specification into a sorted tuple of unique positive ints."""
    if values is None:
        return tuple(KNN_K_GRID)
    if isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [tok.strip() for tok in str(values).split(',') if tok.strip()]

    parsed = sorted({int(v) for v in raw_values if int(v) > 0})
    return tuple(parsed) if parsed else tuple(KNN_K_GRID)


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


def baseline_store_root(path):
    """Return the file-backed baseline cache root for a legacy CSV path."""
    return f"{os.path.splitext(path)[0]}.d"


def mirt_sweep_store_root(path):
    """Return the file-backed MIRT sweep cache root for a legacy CSV path."""
    return f"{os.path.splitext(path)[0]}.d"


def _baseline_key_rel_dir(key):
    pre_revision = normalize_pre_revision(key['pre_revision'])
    j_percentage = normalize_j_percentage(key['j_percentage'])
    baseline_embedding_type = normalize_baseline_embedding_type(key['baseline_embedding_type'])
    pre_population_match = normalize_pre_population_match(key.get('pre_population_match'))
    pre_population_match_version = normalize_pre_population_match_version(
        key.get('pre_population_match_version'),
        pre_population_match=pre_population_match,
    )
    return os.path.join(
        f"model_{str(key['model_type'])}",
        f"embed_{baseline_embedding_type}",
        f"pre_{pre_revision}",
        f"match_{pre_population_match}",
        f"matchv_{pre_population_match_version if not pd.isna(pre_population_match_version) else 'na'}",
        f"j_{j_percentage:.6f}",
        f"n_{int(key['n_samples'])}",
    )


def non_mirt_cache_file(path, key):
    root = baseline_store_root(path)
    return os.path.join(root, 'non_mirt', _baseline_key_rel_dir(key), f"seed_{int(key['seed'])}.json")


def mirt_selected_cache_file(path, key):
    root = baseline_store_root(path)
    return os.path.join(root, 'mirt_selected', _baseline_key_rel_dir(key), f"seed_{int(key['seed'])}.json")


def mirt_sweep_cache_file(path, key):
    root = mirt_sweep_store_root(path)
    return os.path.join(
        root,
        'rows',
        _baseline_key_rel_dir(key),
        f"seed_{int(key['seed'])}__dim_{int(key['mirt_dim'])}.json",
    )


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _iter_json_rows(root):
    if not os.path.isdir(root):
        return
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if not filename.endswith('.json'):
                continue
            path = os.path.join(dirpath, filename)
            try:
                yield _read_json(path)
            except Exception:
                continue


def _normalize_key_payload(payload):
    out = dict(payload)
    out['seed'] = int(out['seed'])
    out['model_type'] = str(out['model_type'])
    out['n_samples'] = int(out['n_samples'])
    out['pre_revision'] = normalize_pre_revision(out['pre_revision'])
    out['j_percentage'] = normalize_j_percentage(out['j_percentage'])
    out['train_retention'] = normalize_train_retention(out.get('train_retention', 1.0))
    out['baseline_embedding_type'] = normalize_baseline_embedding_type(out['baseline_embedding_type'])
    out['pre_population_match'] = normalize_pre_population_match(out.get('pre_population_match'))
    out['pre_population_match_version'] = normalize_pre_population_match_version(
        out.get('pre_population_match_version'),
        pre_population_match=out['pre_population_match'],
    )
    return out


def write_non_mirt_cache(path, row):
    key = _normalize_key_payload({
        **{k: row[k] for k in BASELINE_KEY_COLS},
        'pre_population_match': row.get('pre_population_match'),
        'pre_population_match_version': row.get('pre_population_match_version'),
    })
    payload = {
        **key,
        'agent_batch_size': row.get('agent_batch_size', compute_agent_batch_size(key['pre_revision'], key['n_samples'])),
    }
    for col in NON_MIRT_METRIC_COLS:
        if col in row and not pd.isna(row[col]):
            payload[col] = float(row[col])
    _write_json_atomic(non_mirt_cache_file(path, key), payload)


def write_mirt_selected_cache(path, row):
    key = _normalize_key_payload({
        **{k: row[k] for k in BASELINE_KEY_COLS},
        'pre_population_match': row.get('pre_population_match'),
        'pre_population_match_version': row.get('pre_population_match_version'),
    })
    payload = {**key}
    for col in MIRT_SUMMARY_COLS:
        if col not in row or pd.isna(row[col]):
            continue
        if col in {'selected_mirt_dim', 'mirt_sweep_min', 'mirt_sweep_max', 'mirt_selection_version'}:
            payload[col] = int(row[col])
        else:
            payload[col] = float(row[col])
    _write_json_atomic(mirt_selected_cache_file(path, key), payload)


def read_non_mirt_cache(path, key):
    cache_path = non_mirt_cache_file(path, key)
    if not os.path.exists(cache_path):
        return None
    try:
        return _read_json(cache_path)
    except Exception:
        return None


def read_mirt_selected_cache(path, key):
    cache_path = mirt_selected_cache_file(path, key)
    if not os.path.exists(cache_path):
        return None
    try:
        return _read_json(cache_path)
    except Exception:
        return None


def baseline_row_matches(df, key):
    """Return rows matching baseline key with tolerance on j_percentage."""
    if df.empty:
        return df

    seed_col = pd.to_numeric(df['seed'], errors='coerce')
    n_samples_col = pd.to_numeric(df['n_samples'], errors='coerce')
    j_percentage_col = pd.to_numeric(df['j_percentage'], errors='coerce')
    train_retention_col = pd.to_numeric(df['train_retention'], errors='coerce')
    if 'pre_population_match' in df.columns:
        match_col = df['pre_population_match'].astype(str).map(normalize_pre_population_match)
    else:
        match_col = pd.Series(PRE_POPULATION_MATCH_NONE, index=df.index)
    if 'pre_population_match_version' in df.columns:
        match_version_col = pd.to_numeric(df['pre_population_match_version'], errors='coerce')
    else:
        match_version_col = pd.Series(np.nan, index=df.index)
    key_match = normalize_pre_population_match(key.get('pre_population_match'))
    key_match_version = normalize_pre_population_match_version(
        key.get('pre_population_match_version'),
        pre_population_match=key_match,
    )
    j_match = pd.Series(
        np.isclose(j_percentage_col.to_numpy(dtype=float), float(key['j_percentage']), atol=1e-6, equal_nan=False),
        index=df.index,
    )
    retention_match = pd.Series(
        np.isclose(train_retention_col.to_numpy(dtype=float), float(key['train_retention']), atol=1e-6, equal_nan=False),
        index=df.index,
    )
    if pd.isna(key_match_version):
        match_version_ok = pd.Series(match_version_col.isna().to_numpy(), index=df.index)
    else:
        match_version_ok = pd.Series(
            np.isclose(match_version_col.to_numpy(dtype=float), float(key_match_version), atol=1e-6, equal_nan=False),
            index=df.index,
        )
    mask = (
        (seed_col == int(key['seed'])) &
        (df['model_type'].astype(str) == str(key['model_type'])) &
        (n_samples_col == int(key['n_samples'])) &
        (df['pre_revision'].astype(str) == str(key['pre_revision'])) &
        j_match &
        retention_match &
        (match_col == key_match) &
        match_version_ok &
        (
            df['baseline_embedding_type'].astype(str).map(normalize_baseline_embedding_type) ==
            str(key['baseline_embedding_type'])
        )
    )
    return df[mask.fillna(False)]


def mirt_sweep_row_matches(df, key):
    """Return rows matching a per-dimension MIRT sweep key."""
    if df.empty:
        return df
    match = baseline_row_matches(df, key)
    if match.empty:
        return match
    mirt_dim_col = pd.to_numeric(match['mirt_dim'], errors='coerce')
    return match[mirt_dim_col == int(key['mirt_dim'])]


def load_baseline_store(path):
    """Load baseline store or return an empty frame with required schema."""
    return bc.load_baseline_store(path)


def load_mirt_sweep_store(path):
    """Load MIRT sweep store or return an empty frame with required schema."""
    return bc.load_mirt_sweep_store(path)


def append_baseline_row(path, row):
    """Atomically append or upsert one baseline row in the baseline cache file."""
    lock = FileLock(f"{path}.lock", timeout=600)
    with lock:
        if 'agent_batch_size' not in row:
            row['agent_batch_size'] = compute_agent_batch_size(row.get('pre_revision', 'none'), row.get('n_samples', 0))
        bc.write_grouped_baseline_files(path, row)


def append_mirt_sweep_row(path, row):
    """Atomically append or upsert one MIRT sweep row."""
    lock = FileLock(f"{path}.lock", timeout=600)
    with lock:
        bc.write_grouped_mirt_sweep_file(path, row)


def load_existing_baseline_row(path, key):
    """Lookup baseline row by key without enforcing MIRT sweep coverage."""
    df = load_baseline_store(path)
    if df.empty:
        return None
    match = baseline_row_matches(df, key)
    if match.empty:
        return None
    return match.iloc[-1].to_dict()


def try_get_cached_baseline(path, key, mirt_dim_min=K_MODEL, mirt_dim_max=K_MODEL,
                            baseline_profile=BASELINE_PROFILE_FULL):
    """Lookup cached baseline row by key and requested MIRT sweep range."""
    row = load_existing_baseline_row(path, key)
    if row is None or not _baseline_row_matches_mirt_request(row, mirt_dim_min, mirt_dim_max, baseline_profile=baseline_profile):
        return None
    return _baseline_payload_from_row(row)


def try_get_cached_mirt_sweep_row(path, key):
    """Lookup one cached MIRT sweep row by key."""
    df = load_mirt_sweep_store(path)
    if df.empty:
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
        'val_rmse_mirt': float(row['val_rmse_mirt']),
        'val_auc_mirt': float(row['val_auc_mirt']),
        'mirt_dim': int(row['mirt_dim']),
    }


def infer_completed_max_n_from_output(path):
    """Infer the max n_samples already written to an output CSV, if any."""
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or 'n_samples' not in df.columns:
        return None
    n_col = pd.to_numeric(df['n_samples'], errors='coerce').dropna()
    if n_col.empty:
        return None
    return int(n_col.max())


def infer_completed_max_n_from_baseline_cache(path, seed, model_type, pre_revision, j_percentage, baseline_embedding_type,
                                             train_retention=1.0):
    """Infer cached max n_samples from baseline cache for resume-only checks."""
    df = load_baseline_store(path)
    if df.empty:
        return None

    key = {
        'seed': int(seed),
        'model_type': str(model_type),
        'n_samples': 0,  # ignored below
        'pre_revision': normalize_pre_revision(pre_revision),
        'j_percentage': normalize_j_percentage(j_percentage),
        'train_retention': normalize_train_retention(train_retention),
        'baseline_embedding_type': normalize_baseline_embedding_type(baseline_embedding_type),
        'pre_population_match': effective_pre_population_match(pre_revision),
        'pre_population_match_version': normalize_pre_population_match_version(
            None,
            pre_population_match=effective_pre_population_match(pre_revision),
        ),
    }

    match = baseline_row_matches(df, key)
    if match.empty:
        return None

    n_values = pd.to_numeric(match['n_samples'], errors='coerce').dropna()
    if n_values.empty:
        return None
    return int(n_values.max())


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

        baseline_embedding_type = 'pca'
        if 'embedding_type' in df.columns and df['embedding_type'].notna().any():
            baseline_embedding_type = normalize_baseline_embedding_type(df['embedding_type'].dropna().iloc[0])
        else:
            m_emb = re.match(r"amortized_irt_([^_]+)_", fname)
            if m_emb:
                baseline_embedding_type = normalize_baseline_embedding_type(m_emb.group(1))

        sub = df[required].copy()
        sub['pre_revision'] = pre_revision
        sub['j_percentage'] = j_percentage
        sub['baseline_embedding_type'] = baseline_embedding_type
        sub['pre_population_match'] = PRE_POPULATION_MATCH_NONE
        sub['pre_population_match_version'] = np.nan
        sub['effective_pre_weight_sum'] = np.nan
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
    migrated_df['baseline_embedding_type'] = migrated_df['baseline_embedding_type'].map(normalize_baseline_embedding_type)

    lock = FileLock(f"{baseline_output}.lock", timeout=600)
    with lock:
        existing = load_baseline_store(baseline_output)
        combined = pd.concat([existing, migrated_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=BASELINE_KEY_COLS, keep='last')
        os.makedirs(os.path.dirname(baseline_output), exist_ok=True)
        combined.to_csv(baseline_output, index=False)

    if not quiet:
        print(f"Migrated {len(migrated_df)} baseline rows into {baseline_output}")


def migrate_pair_efficiency_from_results(source_dir, pair_efficiency_output, baseline_output,
                                         quiet=False, model_type_filter=None):
    """Backfill observed-pair study rows from existing amortized result CSVs."""
    if not os.path.isdir(source_dir):
        if not quiet:
            print(f"Pair-efficiency source dir not found: {source_dir}")
        return

    file_pattern = re.compile(
        r"amortized_irt_(sae|pca|raw)_(beta|bernoulli)_pre_([^_]+)_n_(max|1)"
        r"(?:_j([0-9]+(?:\.[0-9]+)?))?\.csv$"
    )
    baseline_df = load_baseline_store(baseline_output)
    rows = []
    observed_cache = {}

    for fname in sorted(os.listdir(source_dir)):
        match = file_pattern.match(fname)
        if not match:
            continue

        embedding_type, model_type, pre_revision, n_token, j_token = match.groups()
        if model_type_filter is not None and model_type != model_type_filter:
            continue
        j_percentage = normalize_j_percentage(float(j_token) if j_token is not None else 1.0)
        pre_revision = normalize_pre_revision(pre_revision)
        path = os.path.join(source_dir, fname)

        try:
            df = pd.read_csv(path, on_bad_lines='skip')
        except Exception:
            continue
        if df.empty:
            continue

        for col in ['seed', 'lambda_tau', 'n_samples', 'auc_amortized', 'rmse_amortized']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['seed', 'lambda_tau', 'auc_amortized', 'rmse_amortized'])
        if df.empty:
            continue

        observed_key = (model_type, pre_revision, j_percentage)
        if observed_key not in observed_cache:
            all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data(
                embedding_type='ones',
                pre_revision=pre_revision,
            )
            data = prepare_experiment_data(
                all_dfs,
                global_shared_indices,
                raw_embs_map,
                embedding_type=actual_emb_type,
                j_percentage=j_percentage,
            )
            n_files = len(all_dfs)
            _, _, _, train_mask_current_t, _ = build_training_targets(
                n_files,
                all_dfs,
                global_shared_indices,
                data,
                model_type=model_type,
                quiet=True,
            )
            observed_cache[observed_key] = {
                'n_files': n_files,
                'observed_train_pairs': int(train_mask_current_t.sum().item()),
            }

        n_files = observed_cache[observed_key]['n_files']
        observed_train_pairs = observed_cache[observed_key]['observed_train_pairs']

        for _, row in df.iterrows():
            baseline_key = {
                'seed': int(row['seed']),
                'model_type': model_type,
                'n_samples': int(n_files),
                'pre_revision': pre_revision,
                'j_percentage': j_percentage,
                'baseline_embedding_type': 'raw',
                'pre_population_match': effective_pre_population_match(pre_revision),
                'pre_population_match_version': normalize_pre_population_match_version(
                    None,
                    pre_population_match=effective_pre_population_match(pre_revision),
                ),
            }
            baseline_row = load_existing_baseline_row(baseline_output, baseline_key)
            if baseline_row is None:
                if not quiet:
                    print(f"Skipping pair-efficiency migration for missing baseline: {baseline_key}")
                continue

            rows.append({
                'seed': int(row['seed']),
                'lambda_tau': float(row['lambda_tau']),
                'n_samples': int(n_files),
                'model_type': model_type,
                'pre_revision': pre_revision,
                'j_percentage': j_percentage,
                'embedding_type': embedding_type,
                'baseline_embedding_type': 'raw',
                'observed_train_pairs': int(observed_train_pairs),
                'auc_knn': float(baseline_row['auc_knn']),
                'rmse_knn': float(baseline_row['rmse_knn']),
                'auc_araf': float(row['auc_amortized']),
                'rmse_araf': float(row['rmse_amortized']),
            })

    if not rows:
        if not quiet:
            print("No pair-efficiency rows discovered to migrate.")
        return

    append_pair_efficiency_rows(pair_efficiency_output, rows)
    if not quiet:
        print(f"Migrated {len(rows)} pair-efficiency rows into {pair_efficiency_output}")


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


# Global worker-local cache. Under spawn, sending large pandas objects through
# initargs adds noticeable startup latency, so each worker loads its data once.
_WORKER_DFS = None
_WORKER_INDICES = None
_WORKER_EMBS_MAP = None
_WORKER_EMB_TYPE = None

def init_worker():
    """Initialize worker process state lazily on first task."""
    global _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE
    _WORKER_DFS = None
    _WORKER_INDICES = None
    _WORKER_EMBS_MAP = None
    _WORKER_EMB_TYPE = None


def ensure_worker_data(args):
    """Load experiment data once per worker instead of pickling it from the parent."""
    global _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE
    if _WORKER_DFS is None or _WORKER_INDICES is None or _WORKER_EMBS_MAP is None or _WORKER_EMB_TYPE is None:
        _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE = load_data(
            embedding_type=args.embedding_type,
            embedding_dim=args.embedding_dim,
            pre_revision=args.pre_revision
        )
    return _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE

def run_single_config(config, args, n_values):
    """Worker function for running a single (seed, lambda_tau) configuration."""
    seed, lambda_tau, worker_id = config

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
        
    global LAMBDA_TAU, RANDOM_SEED, WD_THETA, WD_W, EPOCHS, SNAPPING_THRESHOLD
    LAMBDA_TAU = lambda_tau
    RANDOM_SEED = seed
    if args.wd_theta is not None:
        WD_THETA = args.wd_theta
    if args.wd_w is not None:
        WD_W = args.wd_w
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.snapping_threshold is not None:
        SNAPPING_THRESHOLD = args.snapping_threshold

    if not args.quiet:
        print(f"\n[BOOT] worker {worker_id} initializing -> seed={seed}, tau={lambda_tau}")

    all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = ensure_worker_data(args)

    j_suffix = f"_j{args.j_percentage}" if args.j_percentage < 1.0 else ""
    u_suffix = f"_u{int(args.user_count)}" if args.user_count is not None else ""
    output_path = None
    if not args.baseline_only:
        if args.output:
            output_path = args.output
        else:
            suffix = f"_pre_{args.pre_revision}" if args.pre_revision != 'none' else ""
            n_suffix = f"_n_{args.n_samples}" if args.n_samples != 'all' else "_n_max"
            output_path = os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}{suffix}{u_suffix}{n_suffix}{j_suffix}.csv')

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
                                 embedding_type=actual_emb_type, j_percentage=args.j_percentage,
                                 pre_revision=args.pre_revision,
                                 cross_revision_post_binary=args.cross_revision_post_binary,
                                 user_count=args.user_count,
                                 binarize_oracle=(args.model_type == 'bernoulli' and args.pre_revision == 'none'))
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
                    embedding_type=actual_emb_type,
                    baseline_embedding_type=args.baseline_embedding_type,
                    train_retention=args.train_retention,
                    knn_k=args.knn_k,
                    knn_k_grid=args.knn_k_grid,
                    baseline_profile=args.baseline_profile,
                )
                continue

            result = run_experiment(n, all_dfs, global_shared_indices, data,
                                    model_type=args.model_type, beta_phi=args.beta_phi, no_tau=args.no_tau, 
                                    quiet=quiet, embedding_type=actual_emb_type,
                                    baseline_output=args.baseline_output,
                                    pre_revision=args.pre_revision,
                                    j_percentage=args.j_percentage,
                                    baseline_embedding_type=args.baseline_embedding_type,
                                    pair_efficiency_output=args.pair_efficiency_output,
                                    neighbor_support_output=args.neighbor_support_output,
                                    support_thinning_output=args.support_thinning_output,
                                    outlier_robustness_output=args.outlier_robustness_output,
                                    train_retention=args.train_retention,
                                    knn_k=args.knn_k,
                                    knn_k_grid=args.knn_k_grid,
                                    allow_compute_baselines=True,
                                    mirt_dim_min=args.mirt_dim_min,
                                    mirt_dim_max=args.mirt_dim_max,
                                    mirt_sweep_output=args.mirt_sweep_output,
                                    cross_revision_araf_mode=args.cross_revision_araf_mode,
                                    baseline_profile=args.baseline_profile)

            pair_efficiency_rows = result.pop('pair_efficiency_rows', [])
            neighbor_support_rows = result.pop('neighbor_support_rows', [])
            support_thinning_rows = result.pop('support_thinning_rows', [])
            outlier_robustness_rows = result.pop('outlier_robustness_rows', [])
            if args.pair_efficiency_output:
                append_pair_efficiency_rows(args.pair_efficiency_output, pair_efficiency_rows)
            if args.neighbor_support_output:
                append_neighbor_support_rows(args.neighbor_support_output, neighbor_support_rows)
            if args.support_thinning_output:
                append_support_thinning_rows(args.support_thinning_output, support_thinning_rows)
            if args.outlier_robustness_output:
                append_outlier_robustness_rows(args.outlier_robustness_output, outlier_robustness_rows)

            result['embedding_type'] = actual_emb_type
            result['user_count'] = int(data.get('user_count', data['N']))
            if args.pre_revision != 'none':
                result['scenario'] = f"Pre-{args.pre_revision}"
            results.append(result)
            
            # Save model state to separate pkl if requested
            if args.save_weights:
                weight_meta = {
                    'embedding_type': actual_emb_type,
                    'embedding_protocol': result.get('embedding_protocol', data.get('embedding_protocol')),
                    'embedding_fit_scope': result.get('embedding_fit_scope', data.get('embedding_fit_scope')),
                    'embedding_fit_item_count': int(result.get('embedding_fit_item_count', data.get('embedding_fit_item_count', 0))),
                    'task_ids': result.get('task_ids', []),
                    'train_item_ids': result.get('train_item_ids', []),
                    'test_item_ids': result.get('test_item_ids', []),
                }
                weight_path = output_path.replace('.csv', f'_seed_{seed}_weights_best.pkl')
                torch.save({'state_dict': result['model_state'], **weight_meta}, weight_path)
                
                final_weight_path = output_path.replace('.csv', f'_seed_{seed}_weights_final.pkl')
                torch.save({'state_dict': result['final_state'], **weight_meta}, final_weight_path)
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
        if 'task_ids' in r_copy: del r_copy['task_ids']
        if 'train_item_ids' in r_copy: del r_copy['train_item_ids']
        if 'test_item_ids' in r_copy: del r_copy['test_item_ids']
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
    global _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE
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
    parser.add_argument('--user-count', type=int, default=None, help='Optional number of users (rows) to sample from the oracle matrix.')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--baseline-only', action='store_true', help='Only compute/cache baselines and skip amortized outputs.')
    parser.add_argument('--baseline-embedding-type', type=str, default=None,
                        choices=['raw', 'pca', 'sae'],
                        help='Embedding type used for the cached kNN baseline (defaults to --embedding-type).')
    parser.add_argument('--baseline-output', type=str, default=DEFAULT_BASELINE_OUTPUT,
                        help='Path to baseline cache CSV (default: model/result/baselines/baseline_metrics.csv).')
    parser.add_argument('--mirt-sweep-output', type=str, default=DEFAULT_MIRT_SWEEP_OUTPUT,
                        help='Path to per-dimension MIRT sweep cache CSV.')
    parser.add_argument('--pair-efficiency-output', type=str, default=None,
                        help='Optional CSV path for observed-pair efficiency rows.')
    parser.add_argument('--neighbor-support-output', type=str, default=None,
                        help='Optional CSV path for local neighbor-support study rows.')
    parser.add_argument('--support-thinning-output', type=str, default=None,
                        help='Optional CSV path for support-thinning study rows.')
    parser.add_argument('--outlier-robustness-output', type=str, default=None,
                        help='Optional CSV path for outlier-item and robustness rows.')
    parser.add_argument('--train-retention', type=float, default=1.0,
                        help='Retention rate for observed training entries (0,1].')
    parser.add_argument('--knn-k', type=int, default=KNN_K,
                        help='Number of nearest neighbors used by the kNN baseline.')
    parser.add_argument('--knn-k-grid', type=str, default="5,10,20,50",
                        help='Comma-separated k values used to pick the best headline kNN baseline.')
    parser.add_argument('--baseline-profile', type=str, default=BASELINE_PROFILE_FULL,
                        choices=[BASELINE_PROFILE_FULL, BASELINE_PROFILE_KNN_MIRT, BASELINE_PROFILE_KNN_ONLY],
                        help='Baseline computation profile: full computes all classic baselines; knn_mirt computes only kNN + MIRT; knn_only computes only kNN.')
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
    parser.add_argument('--migrate-pair-efficiency', action='store_true',
                        help='Migrate observed-pair efficiency rows from existing amortized result CSVs, then exit.')
    parser.add_argument('--migrate-model-type', type=str, default=None,
                        choices=['beta', 'bernoulli'],
                        help='Optional model type filter for pair-efficiency migration.')
    parser.add_argument('--cross-revision-post-binary', action='store_true',
                        help='For pre-revision runs, train on pre data but evaluate on a binary post-revision oracle with a held-out 10%% item split.')
    parser.add_argument('--cross-revision-araf-mode', type=str, default='transfer',
                        choices=['transfer', 'post', 'post_support_transfer'],
                        help='Cross-revision ARAF mode: transfer uses the current pre-train/post-eval path; post trains directly on post support; post_support_transfer matches the original post-support-thinning transfer setup.')
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

    if args.migrate_pair_efficiency:
        migrate_pair_efficiency_from_results(
            args.migrate_source_dir,
            args.pair_efficiency_output,
            args.baseline_output,
            quiet=args.quiet,
            model_type_filter=args.migrate_model_type,
        )
        return

    if args.wd_theta is not None:
        WD_THETA = args.wd_theta
    if args.wd_w is not None:
        WD_W = args.wd_w
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.snapping_threshold is not None:
        SNAPPING_THRESHOLD = args.snapping_threshold
    args.knn_k_grid = parse_knn_k_grid(args.knn_k_grid)

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
            u_suffix = f"_u{int(args.user_count)}" if args.user_count is not None else ""
            output_path = os.path.join(RESULT_DIR, f'amortized_irt_{actual_emb_type}_{args.model_type}{suffix}{u_suffix}{n_suffix}{j_suffix}.csv')

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

    n_values = None
    if args.pre_revision != 'none':
        n_values = [1]
    else:
        n_values = parse_explicit_n_samples(args.n_samples)
        if n_values is None and str(args.n_samples).strip() == 'max':
            if args.baseline_only:
                baseline_emb_type = normalize_baseline_embedding_type(
                    args.baseline_embedding_type if args.baseline_embedding_type is not None else args.embedding_type
                )
                n_inferred = infer_completed_max_n_from_baseline_cache(
                    args.baseline_output,
                    seed=seeds[0],
                    model_type=args.model_type,
                    pre_revision=args.pre_revision,
                    j_percentage=args.j_percentage,
                    baseline_embedding_type=baseline_emb_type,
                    train_retention=args.train_retention,
                )
            else:
                n_inferred = infer_completed_max_n_from_output(output_path)
            if n_inferred is not None:
                n_values = [n_inferred]

    all_dfs = global_shared_indices = raw_embs_map = None
    if n_values is None:
        all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data(
            embedding_type=args.embedding_type,
            embedding_dim=args.embedding_dim,
            pre_revision=args.pre_revision
        )
        total_files = len(all_dfs)
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
                    'train_retention': normalize_train_retention(args.train_retention),
                    'baseline_embedding_type': normalize_baseline_embedding_type(
                        args.baseline_embedding_type if args.baseline_embedding_type is not None else args.embedding_type
                    ),
                    'pre_population_match': effective_pre_population_match(args.pre_revision),
                    'pre_population_match_version': normalize_pre_population_match_version(
                        None,
                        pre_population_match=effective_pre_population_match(args.pre_revision),
                    ),
                }
                cached = try_get_cached_baseline(
                    args.baseline_output,
                    baseline_key,
                    mirt_dim_min=args.mirt_dim_min,
                    mirt_dim_max=args.mirt_dim_max,
                    baseline_profile=args.baseline_profile,
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

    if all_dfs is None or global_shared_indices is None or raw_embs_map is None:
        all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data(
            embedding_type=args.embedding_type,
            embedding_dim=args.embedding_dim,
            pre_revision=args.pre_revision
        )

    print(f"\nDiscovered {len(configs)} configurations to execute across {args.parallel} Python generic workers.\n")

    # Run execution pipeline
    if args.parallel > 1 and len(configs) > 1:
        # Prevent PyTorch from hanging with generic spawn context lock
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=args.parallel, initializer=init_worker) as pool:
            worker_fn = partial(run_single_config, args=args, n_values=n_values)
            pool.map(worker_fn, configs)
    else:
        # Sequential execution
        init_worker()
        _WORKER_DFS, _WORKER_INDICES, _WORKER_EMBS_MAP, _WORKER_EMB_TYPE = all_dfs, global_shared_indices, raw_embs_map, actual_emb_type
        for config in configs:
            run_single_config(config, args, n_values)

    print("\n" + "=" * 60)
    print("EXPERIMENT BATCH COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
