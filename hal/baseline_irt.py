"""Baseline-only evaluation: global mean and Rasch-IRT metrics."""

# ==========================================================================
# CONFIGURATION
# ==========================================================================
USE_EMPIRICAL_BASELINE = True  # True: use full empirical, False: use 1 random binary
TEST_SIZE = 0.1  # 10% holdout
RANDOM_SEED = 42
HOLDOUT_CELLS = False  # True: hold out individual cells, False: hold out whole items

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def run_baseline_irt(
    use_empirical_baseline=USE_EMPIRICAL_BASELINE,
    test_size=TEST_SIZE,
    random_seed=RANDOM_SEED,
    holdout_cells=HOLDOUT_CELLS,
    resmat_dir='resmats',
    rasch_epochs=500,
    return_all=False,
):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print(f"Baseline mode: {'EMPIRICAL' if use_empirical_baseline else 'SINGLE BINARY'}")
    split_type = "Cell-wise" if holdout_cells else "Item-wise"
    print(f"Train/Test split: {int((1-test_size)*100)}%/{int(test_size*100)}% ({split_type})")
    print("=" * 60)

    def compute_rmse(predictions, targets, mask):
        valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
        return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

    def compute_corr(predictions, targets, mask):
        valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
        flat_pred = predictions[valid]
        flat_true = targets[valid]
        if flat_pred.size < 2:
            return float('nan')
        return float(np.corrcoef(flat_pred, flat_true)[0, 1])

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # ----------------------------------------------------------------------
    # 1. Load Response Matrices
    # ----------------------------------------------------------------------
    files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
    n_samples = len(files)

    all_dfs = [pd.read_csv(os.path.join(resmat_dir, f), index_col=0) for f in files]

    # Find shared models across all matrices
    shared_indices = set(all_dfs[0].index)
    for df in all_dfs[1:]:
        shared_indices &= set(df.index)
    shared_indices = sorted(list(shared_indices))

    # Filter to shared rows
    filtered_dfs = [df.loc[shared_indices] for df in all_dfs]

    # Calculate empirical probability matrix (element-wise mean across samples)
    prob_matrix = np.nanmean([df.values for df in filtered_dfs], axis=0)
    prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

    print(f"Loaded {n_samples} response matrices")
    print(f"Shared models: {len(shared_indices)}, Tasks: {prob_df.shape[1]}")
    print(f"Empirical prob range: [{np.nanmin(prob_df.values):.3f}, {np.nanmax(prob_df.values):.3f}]")

    # ----------------------------------------------------------------------
    # 2. Train/Test Split
    # ----------------------------------------------------------------------
    N, J = prob_df.shape
    y_empirical = torch.from_numpy(prob_df.values.astype(np.float32)).to(device)

    if holdout_cells:
        valid_idx = np.argwhere(~np.isnan(prob_df.values))
        np.random.shuffle(valid_idx)
        n_test = int(len(valid_idx) * test_size)
        test_pairs = valid_idx[:n_test]
        train_pairs = valid_idx[n_test:]

        train_mask = np.zeros_like(prob_df.values, dtype=bool)
        test_mask = np.zeros_like(prob_df.values, dtype=bool)
        train_mask[train_pairs[:, 0], train_pairs[:, 1]] = True
        test_mask[test_pairs[:, 0], test_pairs[:, 1]] = True

        print(f"Split: {len(train_pairs)} train cells, {len(test_pairs)} test cells")
    else:
        J_indices = np.arange(J)
        np.random.shuffle(J_indices)

        n_test = int(test_size * J)
        test_idx = J_indices[:n_test]
        train_idx = J_indices[n_test:]

        train_mask = np.zeros_like(prob_df.values, dtype=bool)
        train_mask[:, train_idx] = ~np.isnan(prob_df.values)[:, train_idx]
        test_mask = np.zeros_like(prob_df.values, dtype=bool)
        test_mask[:, test_idx] = ~np.isnan(prob_df.values)[:, test_idx]

        print(f"Split: {len(train_idx)} train items, {len(test_idx)} test items")

    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    print(f"Entries: {train_mask.sum()} train, {test_mask.sum()} test")

    # ----------------------------------------------------------------------
    # 3. Baseline Metrics & Evaluation
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    if use_empirical_baseline:
        y_baseline = y_empirical
        baseline_type = f"Empirical ({n_samples} samples)"
    else:
        random_idx = np.random.randint(0, len(filtered_dfs))
        single_df = filtered_dfs[random_idx]
        y_baseline = torch.from_numpy(single_df.values.astype(np.float32)).to(device)
        baseline_type = f"Single Binary (matrix {random_idx + 1}/{n_samples})"

    print(f"Baseline data: {baseline_type}")

    # Baseline 1: Global Mean
    mean_val = torch.nanmean(y_baseline[train_mask_t])
    pred_mean = mean_val.expand_as(y_baseline)

    train_rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
    test_rmse_mean = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
    train_corr_mean = compute_corr(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
    test_corr_mean = compute_corr(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

    print(f"\n1. Global Mean: {mean_val:.4f}")
    print(f"   Train RMSE: {train_rmse_mean:.4f} | Test RMSE: {test_rmse_mean:.4f}")

    # Baseline 2: Rasch-IRT
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
    optimizer = torch.optim.Adam([theta, beta], lr=0.01, weight_decay=1e-5)

    print(f"\n2. Rasch-IRT (training {rasch_epochs} epochs...)")
    for _ in range(rasch_epochs):
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
        train_corr_rasch = compute_corr(probs_rasch.cpu().numpy(), y_empirical.cpu().numpy(), train_mask)
        test_corr_rasch = compute_corr(probs_rasch.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

    print(f"   Train RMSE: {train_rmse_rasch:.4f} | Test RMSE: {test_rmse_rasch:.4f}")

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"{baseline_type} Baselines:")
    print(f"  Global Mean     Train: {train_rmse_mean:.4f} | Test: {test_rmse_mean:.4f}")
    print(f"  Rasch-IRT       Train: {train_rmse_rasch:.4f} | Test: {test_rmse_rasch:.4f}")

    mean_result = {
        'model': 'global_mean',
        'train_rmse': train_rmse_mean,
        'test_rmse': test_rmse_mean,
        'train_corr': train_corr_mean,
        'test_corr': test_corr_mean,
    }
    rasch_result = {
        'model': 'rasch_irt',
        'train_rmse': train_rmse_rasch,
        'test_rmse': test_rmse_rasch,
        'train_corr': train_corr_rasch,
        'test_corr': test_corr_rasch,
    }

    if return_all:
        return {'global_mean': mean_result, 'rasch_irt': rasch_result}
    return rasch_result


if __name__ == "__main__":
    run_baseline_irt()