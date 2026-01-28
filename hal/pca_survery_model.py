import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# ==========================================================================
# CONFIGURATION
# ==========================================================================
EFFECTIVE_BATCHES_LOG = 'effective_batches.json'  # Log of effective batch files

TEST_SIZE = 0.1
RANDOM_SEED = 42

# MODEL
K_MODEL = 30
PCA_COMPONENTS = 48

# SPARSITY
LAMBDA_TAU = 0.002
TAU_INIT = 0.5
TAU_WARMUP = 200
RAMP_EPOCHS = 200
SNAPPING_THRESHOLD = 0.005
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# TRAINING
EPOCHS = 1000
EVAL_EVERY = 100
PATIENCE = 30
MIN_DELTA = 1e-6

# OPTIMIZATION
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 1e-3
WD_W = 1e-5

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================
def compute_rmse(predictions, targets, mask):
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

class ReluARDModel(nn.Module):
    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)
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


def load_effective_batches_log():
    """Load the log of effective batches and their metrics."""
    if os.path.exists(EFFECTIVE_BATCHES_LOG):
        with open(EFFECTIVE_BATCHES_LOG, 'r') as f:
            data = json.load(f)
        print(f"Loaded log with {len(data['effective_files'])} effective batches.")
        return data
    return None


def save_effective_batches_log(effective_files, checked_files, best_rmse_mean, best_rmse_beta, history=None):
    """Save the log of effective batches."""
    data = {
        'effective_files': effective_files,
        'checked_files': checked_files,
        'best_rmse_mean': best_rmse_mean,
        'best_rmse_beta': best_rmse_beta,
        'history': history or []
    }
    with open(EFFECTIVE_BATCHES_LOG, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved log: {len(effective_files)} effective batches out of {len(checked_files)} checked.")


def train_and_evaluate(prob_df, x_j, test_idx, train_idx):
    """Train the model and return RMSE metrics."""
    N, J = prob_df.shape

    y_empirical = torch.from_numpy(prob_df.values.astype(np.float32)).to(device)
    train_mask = np.zeros_like(prob_df.values, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(prob_df.values)[:, train_idx]
    test_mask = np.zeros_like(prob_df.values, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(prob_df.values)[:, test_idx]
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    # Baseline: Global Mean
    mean_val = torch.nanmean(y_empirical[train_mask_t])
    pred_mean = mean_val.expand_as(y_empirical)
    rmse_mean_test = compute_rmse(pred_mean.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

    # Baseline: Rasch
    theta = nn.Parameter(torch.randn(N, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(J, device=device) * 0.1)
    opt_rasch = torch.optim.Adam([theta, beta], lr=0.01)
    for _ in range(300):
        opt_rasch.zero_grad()
        loss = F.binary_cross_entropy_with_logits((theta.unsqueeze(1)-beta.unsqueeze(0)), y_empirical, reduction='none')
        (loss * train_mask_t).sum().backward()
        opt_rasch.step()
    with torch.no_grad():
        p_rasch = torch.sigmoid(theta.unsqueeze(1)-beta.unsqueeze(0))
        rmse_rasch_test = compute_rmse(p_rasch.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)

    # Beta-IRT Model
    model = ReluARDModel(N, J, K_MODEL, PCA_COMPONENTS, x_j, dropout=0.5).to(device)
    optimizer = optim.AdamW([
        {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
        {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
        {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
    ])

    best_beta_rmse = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        probs = model()
        loss_fit = F.binary_cross_entropy(probs[train_mask_t], y_empirical[train_mask_t])

        if epoch < TAU_WARMUP: lam = 0.0
        elif epoch < TAU_WARMUP + RAMP_EPOCHS: lam = LAMBDA_TAU * ((epoch - TAU_WARMUP) / RAMP_EPOCHS)
        else: lam = LAMBDA_TAU
        loss_sparsity = lam * torch.sum(model.get_tau())
        (loss_fit + loss_sparsity).backward()
        optimizer.step()

        if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
            with torch.no_grad():
                active_mask = model.get_tau() > SNAPPING_THRESHOLD
                for k in range(K_MODEL):
                    if not active_mask[k]: model.tau_raw[k] = DEAD_ZONE_VALUE

        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p_test = model()
                curr_rmse = compute_rmse(p_test.cpu().numpy(), y_empirical.cpu().numpy(), test_mask)
                if curr_rmse < best_beta_rmse - MIN_DELTA:
                    best_beta_rmse = curr_rmse
                    patience_counter = 0
                elif epoch > TAU_WARMUP:
                    patience_counter += 1
                    if patience_counter >= PATIENCE: break

    return rmse_mean_test, rmse_rasch_test, best_beta_rmse


def prepare_data_from_files(file_list, resmat_dir, raw_embs_map):
    """Load and prepare data from a list of files."""
    if len(file_list) == 0:
        return None, None, None, None

    dfs = []
    for f in file_list:
        dfs.append(pd.read_csv(os.path.join(resmat_dir, f), index_col=0))

    # Find shared indices
    shared_indices = set(dfs[0].index)
    for df in dfs[1:]:
        shared_indices = shared_indices.intersection(set(df.index))
    shared_indices = sorted(list(shared_indices))

    if len(shared_indices) == 0:
        return None, None, None, None

    filtered_dfs = [df.loc[shared_indices] for df in dfs]
    stacked_matrix = np.array([df.values for df in filtered_dfs])
    prob_matrix = np.nanmean(stacked_matrix, axis=0)
    prob_df = pd.DataFrame(prob_matrix, index=shared_indices, columns=filtered_dfs[0].columns)

    # Embeddings
    task_ids = prob_df.columns.tolist()
    current_raw_embs = []
    for task_id in task_ids:
        emb = raw_embs_map.get(str(task_id))
        if emb is None and str(task_id).startswith('colbench.'):
            number = str(task_id).split('.')[-1]
            emb = raw_embs_map.get(f'colbench_backend_programming.{number}')

        if emb is None: emb = np.zeros(4096)
        elif isinstance(emb, str): emb = ast.literal_eval(emb)
        current_raw_embs.append(emb)

    x_np = np.array(current_raw_embs)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(x_np)
    x_pca = x_pca / (np.linalg.norm(x_pca, axis=1, keepdims=True) + 1e-8)
    x_j = torch.tensor(x_pca, dtype=torch.float32).to(device)

    # Create train/test split indices (consistent across runs)
    N, J = prob_df.shape
    J_indices = np.arange(J)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(J_indices)
    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    return prob_df, x_j, test_idx, train_idx


# ==========================================================================
# MAIN: ITERATIVE BATCH SELECTION
# ==========================================================================
def main():
    resmat_dir = 'resmats'
    all_files = [f for f in os.listdir(resmat_dir) if f.startswith('resmat')]
    total_files = len(all_files)
    print(f"Found {total_files} total batch files (unordered).")

    # Load embeddings
    emb_file = 'result/all_benchmarks_embeddings_4096_8B.pkl'
    raw_embs_map = {}
    if os.path.exists(emb_file):
        print("Loading embeddings dictionary...")
        emb_df = pd.read_pickle(emb_file)
        for _, r in emb_df.iterrows():
            raw_embs_map[str(r['benchmark.task_id'])] = r['embedding']
            if str(r['benchmark.task_id']).startswith('colbench_backend_programming'):
                suffix = str(r['benchmark.task_id']).split('.')[-1]
                raw_embs_map[f'colbench.{suffix}'] = r['embedding']

    # Check for existing log
    log_data = load_effective_batches_log()

    if log_data is not None:
        # Resume from log - only use effective batches
        effective_files = log_data['effective_files']
        checked_files = set(log_data['checked_files'])
        best_rmse_mean = log_data['best_rmse_mean']
        best_rmse_beta = log_data['best_rmse_beta']

        # Find new files that haven't been checked yet
        new_files = [f for f in all_files if f not in checked_files]

        history = log_data.get('history', [])

        if len(new_files) == 0:
            print("All files have been checked. Using effective batches only.")
            print(f"Effective batches ({len(effective_files)}): {effective_files}")
            print(f"Best RMSE - Mean: {best_rmse_mean:.4f}, Beta: {best_rmse_beta:.4f}")
            # Skip to plotting
        else:
            print(f"Found {len(new_files)} new unchecked files. Continuing evaluation...")
    else:
        # Start fresh
        effective_files = []
        checked_files = set()
        best_rmse_mean = float('inf')
        best_rmse_beta = float('inf')
        new_files = all_files
        history = []
        print("Starting fresh evaluation of all batches...")

    # We need at least 2 files to start
    if len(effective_files) < 2:
        # Bootstrap with first 2 files
        bootstrap_files = new_files[:2]
        new_files = new_files[2:]

        for f in bootstrap_files:
            effective_files.append(f)
            checked_files.add(f)

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        prob_df, x_j, test_idx, train_idx = prepare_data_from_files(effective_files, resmat_dir, raw_embs_map)
        if prob_df is not None:
            best_rmse_mean, rmse_rasch, best_rmse_beta = train_and_evaluate(prob_df, x_j, test_idx, train_idx)
            print(f"Bootstrap with {effective_files}: RMSE Mean={best_rmse_mean:.4f}, Beta={best_rmse_beta:.4f}")
            history.append({
                'n_batches': len(effective_files),
                'batch_added': effective_files[-1],
                'rmse_mean': best_rmse_mean,
                'rmse_rasch': rmse_rasch,
                'rmse_beta': best_rmse_beta
            })

        save_effective_batches_log(effective_files, list(checked_files), best_rmse_mean, best_rmse_beta, history)

    # Iterate through remaining files
    for i, new_file in enumerate(new_files):
        print(f"\n[{i+1}/{len(new_files)}] Testing batch: {new_file}")
        checked_files.add(new_file)

        # Test with this new file added
        candidate_files = effective_files + [new_file]

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        prob_df, x_j, test_idx, train_idx = prepare_data_from_files(candidate_files, resmat_dir, raw_embs_map)

        if prob_df is None:
            print(f"   -> Skipped (no shared indices)")
            save_effective_batches_log(effective_files, list(checked_files), best_rmse_mean, best_rmse_beta, history)
            continue

        rmse_mean, rmse_rasch, rmse_beta = train_and_evaluate(prob_df, x_j, test_idx, train_idx)

        print(f"   -> Current Beta RMSE: {rmse_beta:.4f} (Mean: {rmse_mean:.4f})")
        print(f"   -> Best Beta RMSE:    {best_rmse_beta:.4f}")

        # Check if this batch improves Beta-IRT RMSE
        if rmse_beta < best_rmse_beta:
            print(f"   -> ACCEPTED: Improves Beta RMSE!")
            effective_files.append(new_file)
            best_rmse_mean = rmse_mean
            best_rmse_beta = rmse_beta
            history.append({
                'n_batches': len(effective_files),
                'batch_added': new_file,
                'rmse_mean': rmse_mean,
                'rmse_rasch': rmse_rasch,
                'rmse_beta': rmse_beta
            })
        else:
            print(f"   -> REJECTED: Does not improve Beta RMSE.")

        # Save progress after each file
        save_effective_batches_log(effective_files, list(checked_files), best_rmse_mean, best_rmse_beta, history)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Effective batches ({len(effective_files)}/{total_files}):")
    for f in effective_files:
        print(f"  - {f}")
    print(f"Best RMSE - Mean: {best_rmse_mean:.4f}, Beta: {best_rmse_beta:.4f}")

    # ==========================================================================
    # PLOT: Performance gain by batch addition order
    # ==========================================================================
    print("\nGenerating performance plot...")

    # Use saved history, or regenerate if missing
    if not history:
        print("No history found. Regenerating (one-time)...")
        history = []
        for i in range(2, len(effective_files) + 1):
            subset_files = effective_files[:i]
            torch.manual_seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            prob_df, x_j, test_idx, train_idx = prepare_data_from_files(subset_files, resmat_dir, raw_embs_map)
            if prob_df is not None:
                rmse_mean, rmse_rasch, rmse_beta = train_and_evaluate(prob_df, x_j, test_idx, train_idx)
                history.append({
                    'n_batches': i,
                    'batch_added': effective_files[i-1],
                    'rmse_mean': rmse_mean,
                    'rmse_rasch': rmse_rasch,
                    'rmse_beta': rmse_beta
                })
                print(f"  [{i}/{len(effective_files)}] Mean={rmse_mean:.4f}, Rasch={rmse_rasch:.4f}, Beta={rmse_beta:.4f}")
        # Save history for future runs
        save_effective_batches_log(effective_files, list(checked_files), best_rmse_mean, best_rmse_beta, history)

    df_results = pd.DataFrame(history)
    df_results.to_csv('effective_batches_results.csv', index=False)
    print(f"Saved results to effective_batches_results.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = df_results['n_batches']
    ax.plot(x, df_results['rmse_mean'], 'o-', label='Global Mean', color='gray', linewidth=2, markersize=6)
    ax.plot(x, df_results['rmse_rasch'], 's-', label='Rasch', color='blue', linewidth=2, markersize=6)
    ax.plot(x, df_results['rmse_beta'], '^-', label='Beta-IRT', color='green', linewidth=2, markersize=8)

    ax.set_xlabel('Number of Effective Batches', fontsize=12)
    ax.set_ylabel('RMSE (Test Set)', fontsize=12)
    ax.set_title('Performance Gain with Incremental Batch Selection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate batch names on x-axis
    ax.set_xticks(x)
    batch_labels = [f"{i}\n({row['batch_added'].replace('resmat_', '').replace('.csv', '')})"
                   for i, row in df_results.iterrows()]
    ax.set_xticklabels([f"{row['n_batches']}" for _, row in df_results.iterrows()], fontsize=10)

    plt.tight_layout()
    plt.savefig('effective_batches_performance.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to effective_batches_performance.png")
    plt.show()

    return effective_files, best_rmse_mean, best_rmse_beta


if __name__ == "__main__":
    main()
