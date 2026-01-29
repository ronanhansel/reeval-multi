"""
Aggregate Survey - N-Holdout Response Matrix Experiment

Evaluates model performance across varying numbers of response matrix samples.
Shows how prediction quality improves with more training data (n=1, n=2, ..., n=max).

Runs both PCA and SAE Amortised IRT models for comparison.

Reuses:
- embeddings.py for embedding loading
- models.py for IRT models and baselines

Usage:
    python aggregate.py                           # Run all n values
    python aggregate.py --n-samples 1,22          # Run only n=1 and n=22
    python aggregate.py --n-samples 1-5,22        # Run n=1 through 5, plus 22
    python aggregate.py --model beta              # Use Beta IRT (default)
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

from embeddings import get_embeddings, align_embeddings_to_tasks, ensure_data_downloaded
from models import (
    BernoulliIRT, BetaIRT,
    train_model, train_rasch_baseline, compute_global_mean_baseline,
    device, RANDOM_SEED, TEST_SIZE, K_MODEL, EPOCHS
)

warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data-reeval-multi')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')


def load_colbench_matrices():
    """
    Load all ColBench response matrices.

    Returns:
        all_dfs: list of DataFrames (one per response matrix)
        shared_indices: list of model indices present in all matrices
        task_ids: list of task/item identifiers
    """
    ensure_data_downloaded()

    resmat_dir = os.path.join(DATA_DIR, 'colbench')
    all_files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])

    print(f"Loading {len(all_files)} ColBench response matrices...")

    all_dfs = [pd.read_csv(os.path.join(resmat_dir, f), index_col=0) for f in all_files]

    # Find shared indices (models present in all matrices)
    shared_indices = set(all_dfs[0].index)
    for df in all_dfs[1:]:
        shared_indices = shared_indices.intersection(set(df.index))
    shared_indices = sorted(list(shared_indices))

    # Get task IDs from columns
    task_ids = all_dfs[0].columns.tolist()

    print(f"Total matrices: {len(all_dfs)}")
    print(f"Shared models: {len(shared_indices)}")
    print(f"Items per matrix: {len(task_ids)}")

    return all_dfs, shared_indices, task_ids


def create_oracle_matrix(all_dfs, shared_indices):
    """
    Create oracle (ground truth) matrix by averaging all response matrices.

    Returns:
        oracle_df: DataFrame with averaged responses
    """
    filtered_dfs = [df.loc[shared_indices] for df in all_dfs]
    stacked = np.array([df.values for df in filtered_dfs])
    oracle_matrix = np.nanmean(stacked, axis=0)
    oracle_df = pd.DataFrame(oracle_matrix, index=shared_indices, columns=filtered_dfs[0].columns)
    return oracle_df


def prepare_train_test_split(oracle_df):
    """
    Create train/test split by items (cold-start evaluation).

    Returns:
        dict with y_oracle, train_idx, test_idx, masks, etc.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N, J = oracle_df.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    # Oracle tensors
    oracle_values = np.nan_to_num(oracle_df.values, nan=0.0).astype(np.float32)
    y_oracle = torch.from_numpy(oracle_values).to(device)

    # Masks for oracle
    oracle_train_mask = np.zeros_like(oracle_df.values, dtype=bool)
    oracle_train_mask[:, train_idx] = ~np.isnan(oracle_df.values)[:, train_idx]

    oracle_test_mask = np.zeros_like(oracle_df.values, dtype=bool)
    oracle_test_mask[:, test_idx] = ~np.isnan(oracle_df.values)[:, test_idx]

    return {
        'N': N,
        'J': J,
        'y_oracle': y_oracle,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'oracle_train_mask': oracle_train_mask,
        'oracle_test_mask': oracle_test_mask,
        'oracle_test_mask_t': torch.from_numpy(oracle_test_mask).to(device),
    }


def run_single_n_experiment(n_files, all_dfs, shared_indices, data, x_j_pca, x_j_sae, model_type='beta'):
    """
    Run experiment for a specific number of response matrix samples.

    Args:
        n_files: number of response matrices to use for training
        all_dfs: list of all response matrices
        shared_indices: shared model indices
        data: dict from prepare_train_test_split
        x_j_pca: PCA embedding tensor
        x_j_sae: SAE embedding tensor
        model_type: 'bernoulli' or 'beta'

    Returns:
        dict with metrics for each model
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N = data['N']
    J = data['J']
    y_oracle = data['y_oracle']
    train_idx = data['train_idx']
    oracle_test_mask = data['oracle_test_mask']
    oracle_test_mask_t = data['oracle_test_mask_t']

    # Create training target from n_files matrices
    current_dfs = [all_dfs[i].loc[shared_indices] for i in range(n_files)]
    current_stacked = np.array([df.values for df in current_dfs])
    train_target = np.nanmean(current_stacked, axis=0)
    train_target_clean = np.nan_to_num(train_target, nan=0.0).astype(np.float32)
    y_train = torch.from_numpy(train_target_clean).to(device)

    # Training mask for current n
    train_mask = np.zeros_like(train_target, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(train_target)[:, train_idx]
    train_mask_t = torch.from_numpy(train_mask).to(device)

    # Helper to compute metrics
    def compute_metrics(probs):
        p_flat = probs[oracle_test_mask_t].cpu().numpy()
        y_flat = y_oracle[oracle_test_mask_t].cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_flat, p_flat))
        y_bin = (y_flat > 0.5).astype(int)
        auc = roc_auc_score(y_bin, p_flat) if len(np.unique(y_bin)) > 1 else 0.5
        return rmse, auc

    # 1. Global Mean baseline
    pred_mean = compute_global_mean_baseline(y_train, train_mask_t, y_train.shape)
    rmse_mean, auc_mean = compute_metrics(pred_mean)

    # 2. Rasch IRT baseline
    p_rasch = train_rasch_baseline(N, J, y_train, train_mask_t)
    rmse_rasch, auc_rasch = compute_metrics(p_rasch)

    # 3. Amortised Difficulty (theta + amortized difficulty, no latent factors)
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
        loss = F.mse_loss(probs[train_mask_t], y_train[train_mask_t])
        loss.backward()
        opt_ad.step()

    model_ad.eval()
    with torch.no_grad():
        p_ad = model_ad()
        rmse_ad, auc_ad = compute_metrics(p_ad)

    # 4. Amortized IRT (PCA)
    torch.manual_seed(RANDOM_SEED)
    if model_type == 'bernoulli':
        model_pca = BernoulliIRT(N, J, K_MODEL, x_j_pca.shape[1], x_j_pca, dropout=0.5).to(device)
    else:
        model_pca = BetaIRT(N, J, K_MODEL, x_j_pca.shape[1], x_j_pca, dropout=0.5).to(device)

    _, _ = train_model(model_pca, y_train, train_mask_t, y_oracle, oracle_test_mask,
                       epochs=EPOCHS, model_type=model_type)

    model_pca.eval()
    with torch.no_grad():
        p_pca = model_pca()
        rmse_pca, auc_pca = compute_metrics(p_pca)
        active_dims_pca = model_pca.get_active_dims()

    # 5. Amortized IRT (SAE)
    torch.manual_seed(RANDOM_SEED)
    if model_type == 'bernoulli':
        model_sae = BernoulliIRT(N, J, K_MODEL, x_j_sae.shape[1], x_j_sae, dropout=0.5).to(device)
    else:
        model_sae = BetaIRT(N, J, K_MODEL, x_j_sae.shape[1], x_j_sae, dropout=0.5).to(device)

    _, _ = train_model(model_sae, y_train, train_mask_t, y_oracle, oracle_test_mask,
                       epochs=EPOCHS, model_type=model_type)

    model_sae.eval()
    with torch.no_grad():
        p_sae = model_sae()
        rmse_sae, auc_sae = compute_metrics(p_sae)
        active_dims_sae = model_sae.get_active_dims()

    return {
        'n_samples': n_files,
        'rmse_mean': rmse_mean,
        'rmse_rasch': rmse_rasch,
        'rmse_ad': rmse_ad,
        'rmse_pca': rmse_pca,
        'rmse_sae': rmse_sae,
        'auc_mean': auc_mean,
        'auc_rasch': auc_rasch,
        'auc_ad': auc_ad,
        'auc_pca': auc_pca,
        'auc_sae': auc_sae,
        'active_dims_pca': active_dims_pca,
        'active_dims_sae': active_dims_sae,
    }


def parse_n_samples(arg, total_files):
    """
    Parse --n-samples argument into list of integers.

    Examples:
        'all' -> [1, 2, ..., total_files]
        '22' -> [22]
        '1,22' -> [1, 22]
        '1-5,22' -> [1, 2, 3, 4, 5, 22]
    """
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


def run_aggregate_survey(embedding_dim=48, k_sparsity=4,
                         model_type='beta', n_samples_arg='all'):
    """
    Run the full aggregate survey across varying n values.

    Args:
        embedding_dim: dimension for pca/sae
        k_sparsity: sparsity for sae
        model_type: 'bernoulli' or 'beta'
        n_samples_arg: which n values to run (e.g., 'all', '1,22', '1-5')

    Returns:
        DataFrame with results for all n values
    """
    print("=" * 60)
    print("AGGREGATE SURVEY - N-HOLDOUT EXPERIMENT")
    print("=" * 60)
    print(f"Model type: {model_type}")

    # Load data
    all_dfs, shared_indices, task_ids = load_colbench_matrices()
    total_files = len(all_dfs)

    # Parse n_samples
    n_values = parse_n_samples(n_samples_arg, total_files)
    print(f"\nWill run experiments for n = {n_values}")

    # Create oracle and splits
    oracle_df = create_oracle_matrix(all_dfs, shared_indices)
    data = prepare_train_test_split(oracle_df)

    print(f"Oracle matrix shape: {oracle_df.shape}")
    print(f"Train items: {len(data['train_idx'])}, Test items: {len(data['test_idx'])}")

    # Load and align PCA embeddings
    print("\nLoading PCA embeddings...")
    pca_emb, pca_task_ids, _ = get_embeddings(
        embedding_type='pca',
        dim=embedding_dim,
        benchmark='colbench'
    )
    pca_aligned = align_embeddings_to_tasks(pca_emb, pca_task_ids, task_ids, 'colbench')
    x_j_pca = torch.tensor(pca_aligned, dtype=torch.float32).to(device)
    print(f"PCA embeddings shape: {x_j_pca.shape}")

    # Load and align SAE embeddings
    print("Loading SAE embeddings...")
    sae_emb, sae_task_ids, _ = get_embeddings(
        embedding_type='sae',
        dim=embedding_dim,
        k_sparsity=k_sparsity,
        benchmark='colbench'
    )
    sae_aligned = align_embeddings_to_tasks(sae_emb, sae_task_ids, task_ids, 'colbench')
    x_j_sae = torch.tensor(sae_aligned, dtype=torch.float32).to(device)
    print(f"SAE embeddings shape: {x_j_sae.shape}")

    # Run experiments
    print("\n" + "=" * 60)
    print("STARTING EXPERIMENTS")
    print("=" * 60)

    results = []
    for i, n in enumerate(n_values):
        print(f"\n[{i+1}/{len(n_values)}] Processing with n={n} sample(s)...")

        result = run_single_n_experiment(n, all_dfs, shared_indices, data, x_j_pca, x_j_sae, model_type)
        results.append(result)

        print(f"   RMSE | Mean: {result['rmse_mean']:.4f} | Rasch: {result['rmse_rasch']:.4f} | "
              f"AD: {result['rmse_ad']:.4f} | PCA: {result['rmse_pca']:.4f} | SAE: {result['rmse_sae']:.4f}")
        print(f"   AUC  | Mean: {result['auc_mean']:.4f} | Rasch: {result['auc_rasch']:.4f} | "
              f"AD: {result['auc_ad']:.4f} | PCA: {result['auc_pca']:.4f} | SAE: {result['auc_sae']:.4f}")
        print(f"   Active dims | PCA: {result['active_dims_pca']} | SAE: {result['active_dims_sae']}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Aggregate Survey - N-Holdout Experiment')
    parser.add_argument('--embedding-dim', type=int, default=48,
                        help='Embedding dimension (default: 48)')
    parser.add_argument('--k-sparsity', type=int, default=4,
                        help='SAE sparsity (default: 4)')
    parser.add_argument('--model', type=str, default='beta',
                        choices=['bernoulli', 'beta'],
                        help='Model type (default: beta)')
    parser.add_argument('--n-samples', type=str, default='all',
                        help='Which n values to run. Examples: "all", "22", "1,22", "1-5,22"')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Using device: {device}")

    # Run aggregate survey (runs both PCA and SAE)
    results_df = run_aggregate_survey(
        embedding_dim=args.embedding_dim,
        k_sparsity=args.k_sparsity,
        model_type=args.model,
        n_samples_arg=args.n_samples
    )

    # Save results
    output_file = os.path.join(RESULT_DIR, 'convergence_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n[OUTPUT] Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATE SURVEY COMPLETE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Show improvement from n=1 to n=max
    if len(results_df) > 1:
        first = results_df.iloc[0]
        last = results_df.iloc[-1]
        print(f"\nImprovement from n={int(first['n_samples'])} to n={int(last['n_samples'])}:")
        print(f"  PCA RMSE: {first['rmse_pca']:.4f} -> {last['rmse_pca']:.4f} "
              f"({(first['rmse_pca'] - last['rmse_pca']) / first['rmse_pca'] * 100:.1f}% reduction)")
        print(f"  SAE RMSE: {first['rmse_sae']:.4f} -> {last['rmse_sae']:.4f} "
              f"({(first['rmse_sae'] - last['rmse_sae']) / first['rmse_sae'] * 100:.1f}% reduction)")
        print(f"  PCA AUC:  {first['auc_pca']:.4f} -> {last['auc_pca']:.4f} "
              f"({(last['auc_pca'] - first['auc_pca']) / first['auc_pca'] * 100:.1f}% improvement)")
        print(f"  SAE AUC:  {first['auc_sae']:.4f} -> {last['auc_sae']:.4f} "
              f"({(last['auc_sae'] - first['auc_sae']) / first['auc_sae'] * 100:.1f}% improvement)")


if __name__ == '__main__':
    main()
