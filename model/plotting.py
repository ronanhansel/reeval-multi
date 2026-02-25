#!/usr/bin/env python3
"""
Plotting for Amortized IRT Experiments

Generates plots from CSV results produced by amortized_irt.py:
  - Comparison bar plots (n=1 vs n=max for RMSE and AUC)
  - Convergence line plots (metrics vs number of samples)

Usage:
    python plotting.py                           # Generate all plots
    python plotting.py --csv result/custom.csv   # Use specific CSV
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import subprocess
from huggingface_hub import snapshot_download

# Conditionally import tueplots and check for latex
HAS_TEX = False
try:
    from tueplots import bundles
    # Check if latex is actually installed on the system
    import platform
    if platform.system() != 'Windows':
        result = subprocess.run(['which', 'latex'], capture_output=True, text=True)
        if result.returncode == 0:
            HAS_TEX = True
except ImportError:
    pass

import contextlib

@contextlib.contextmanager
def optional_rc_context():
    if HAS_TEX:
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            yield
    else:
        yield

sys.path.append('..')

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

colors = sns.color_palette("muted")
muted_blue = colors[0]
muted_orange = colors[1]
muted_green = colors[2]
muted_red = colors[3]

# Data paths
HF_REPO_ID = "ronanhansel/data-reeval-multi"


# ══════════════════════════════════════════════════════════════════════════════
# Response Matrix Loading and Visualization
# ══════════════════════════════════════════════════════════════════════════════

def load_response_matrices():
    """
    Load response matrices from local item-editor directory.

    Returns:
        list of DataFrames: Each DataFrame is a binary response matrix (models x items)
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resmat_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix')
    post_rev_dir = os.path.join(resmat_dir, 'post-revision')

    colbench_dir = os.path.join(post_rev_dir, 'colbench_backend_programming', 'resmat')

    all_files = sorted([f for f in os.listdir(colbench_dir) if f.startswith('resmat')])
    if not all_files:
        print(f"No response matrix files found in {colbench_dir}")
        return None

    # Sort files to ensure stable order
    all_dfs = [pd.read_csv(os.path.join(colbench_dir, f), index_col=0) for f in all_files]
    print(f"Loaded {len(all_dfs)} response matrices from {colbench_dir}")
    return all_dfs


def compute_empirical_probability_matrix(response_matrices):
    """
    Compute the empirical probability matrix P_hat by averaging all response matrices.

    Args:
        response_matrices: list of DataFrames (binary response matrices)

    Returns:
        DataFrame: Empirical probability matrix with values in [0, 1]
    """
    if not response_matrices:
        return None

    # Find shared indices (models present in all matrices)
    shared_indices = set(response_matrices[0].index)
    for df in response_matrices[1:]:
        shared_indices = shared_indices.intersection(set(df.index))
    shared_indices = sorted(list(shared_indices))

    # Stack and average
    filtered_dfs = [df.loc[shared_indices] for df in response_matrices]
    stacked = np.array([df.values for df in filtered_dfs], dtype=float)
    p_hat = np.nanmean(stacked, axis=0)

    return pd.DataFrame(p_hat, index=shared_indices, columns=filtered_dfs[0].columns)


def visualize_binary_response_matrix(Y, title="Response Matrix Y", output_path=None, sort_by_mean=True):
    """
    Visualize a binary response matrix with red=0 and blue=1.

    Args:
        Y: DataFrame or numpy array with binary values (0/1)
        title: Plot title
        output_path: Path to save the figure (PNG)
        sort_by_mean: If True, sort rows by model mean (descending) and columns by item mean (descending)
    """
    if isinstance(Y, pd.DataFrame):
        df = Y.copy()
    else:
        df = pd.DataFrame(Y)

    # Sort by mean if requested
    if sort_by_mean:
        # Sort rows (models) by their mean score (descending - best models at top)
        row_means = df.mean(axis=1)
        row_order = row_means.sort_values(ascending=False).index
        df = df.loc[row_order]

        # Sort columns (items) by their mean difficulty (descending - easier items on left)
        col_means = df.mean(axis=0)
        col_order = col_means.sort_values(ascending=False).index
        df = df[col_order]

    values = df.values
    row_labels = df.index.tolist()

    # Colormap: red=0, blue=1
    cmap = mcolors.ListedColormap(["#d62728", "#1f77b4"])  # red, blue
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    values = df.values.astype(float)
    with optional_rc_context():
        fig, ax = plt.subplots(figsize=(6.75, 2.5))  # Full ICML page width
        cax = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_title(title)
    ax.set_xlabel('Items (sorted by difficulty)')
    ax.set_ylabel('Models (sorted by performance)')

    # Add model labels on y-axis
    short_labels = [lbl.split('.')[-1][:25] if '.' in lbl else lbl[:25] for lbl in row_labels]
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(short_labels, fontsize=6)

    # Colorbar
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0 (Fail)', '1 (Pass)'])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OUTPUT] Saved binary response matrix plot: {output_path}")

    plt.close()


def visualize_probability_matrix(P_hat, title="Empirical Probability Matrix $\\hat{P}$", output_path=None, sort_by_mean=True):
    """
    Visualize the empirical probability matrix with a continuous colormap.

    Args:
        P_hat: DataFrame or numpy array with probability values in [0, 1]
        title: Plot title
        output_path: Path to save the figure (PNG)
        sort_by_mean: If True, sort rows by model mean (descending) and columns by item mean (descending)
    """
    if isinstance(P_hat, pd.DataFrame):
        df = P_hat.copy()
    else:
        df = pd.DataFrame(P_hat)

    # Sort by mean if requested
    if sort_by_mean:
        # Sort rows (models) by their mean score (descending - best models at top)
        row_means = df.mean(axis=1)
        row_order = row_means.sort_values(ascending=False).index
        df = df.loc[row_order]

        # Sort columns (items) by their mean difficulty (descending - easier items on left)
        col_means = df.mean(axis=0)
        col_order = col_means.sort_values(ascending=False).index
        df = df[col_order]

    values = df.values
    row_labels = df.index.tolist()

    # Continuous colormap: red (0) -> white (0.5) -> blue (1)
    cmap = plt.cm.RdBu

    with optional_rc_context():
        fig, ax = plt.subplots(figsize=(6.75, 2.5))  # Full ICML page width
        cax = ax.imshow(values, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    ax.set_title(title)
    ax.set_xlabel('Items (sorted by difficulty)')
    ax.set_ylabel('Models (sorted by performance)')

    # Add model labels on y-axis
    short_labels = [lbl.split('.')[-1][:25] if '.' in lbl else lbl[:25] for lbl in row_labels]
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(short_labels, fontsize=6)

    # Colorbar
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_label('Probability')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OUTPUT] Saved probability matrix plot: {output_path}")

    plt.close()


def plot_response_and_probability_matrices(output_dir=None):
    """
    Load response matrices, compute empirical probability, and visualize both.

    Args:
        output_dir: Directory to save plots (default: RESULT_DIR)
    """
    if output_dir is None:
        output_dir = RESULT_DIR

    # Load response matrices
    response_matrices = load_response_matrices()
    if response_matrices is None:
        return

    print(f"\nResponse matrices loaded: {len(response_matrices)}")
    print(f"Shape of first matrix: {response_matrices[0].shape}")

    # Visualize first response matrix Y_1
    Y_1 = response_matrices[0]
    visualize_binary_response_matrix(
        Y_1,
        title=f"Binary Response Matrix $Y_1$ ({Y_1.shape[0]} models x {Y_1.shape[1]} items)",
        output_path=os.path.join(output_dir, 'response_matrix_Y1.png')
    )

    # Compute and visualize empirical probability matrix
    P_hat = compute_empirical_probability_matrix(response_matrices)
    if P_hat is not None:
        print(f"Empirical probability matrix shape: {P_hat.shape}")
        visualize_probability_matrix(
            P_hat,
            title=f"Empirical Probability Matrix $\\hat{{P}}$ ({P_hat.shape[0]} models x {P_hat.shape[1]} items, averaged over {len(response_matrices)} runs)",
            output_path=os.path.join(output_dir, 'empirical_probability_matrix_P_hat.png')
        )


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_amortized_irt_results(csv_path=None):
    """Load results from amortized_irt.py CSV output."""
    if csv_path is None:
        # Try to find any amortized_irt CSV in result directory
        candidates = ['amortized_irt_pca.csv', 'amortized_irt_sae.csv', 'amortized_irt_raw.csv']
        for name in candidates:
            path = os.path.join(RESULT_DIR, name)
            if os.path.exists(path):
                csv_path = path
                break

    if csv_path is None or not os.path.exists(csv_path):
        return None

    print(f"Loading results from {csv_path}")
    return pd.read_csv(csv_path)


# ══════════════════════════════════════════════════════════════════════════════
# Amortized IRT Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_rmse_comparison(df, output_path=None):
    """
    Bar plot comparing RMSE at n=1 vs n=max.

    Shows performance of Global Mean, Rasch-IRT, and Amortized IRT
    when using minimal vs maximal training data.
    """
    if df is None or len(df) == 0:
        print("No data for RMSE comparison plot")
        return

    n_min = df['n_samples'].min()
    n_max = df['n_samples'].max()

    row_min = df[df['n_samples'] == n_min].iloc[0]
    row_max = df[df['n_samples'] == n_max].iloc[0]

    models = ['Global Mean', 'Rasch-IRT', 'Amortized IRT']
    rmse_n1 = [row_min['rmse_mean'], row_min['rmse_rasch'], row_min['rmse_amortized']]
    rmse_nmax = [row_max['rmse_mean'], row_max['rmse_rasch'], row_max['rmse_amortized']]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, rmse_n1, width, label=f'n={n_min}', color=muted_blue)
    bars2 = ax.bar(x + width/2, rmse_nmax, width, label=f'n={n_max}', color=muted_orange)

    ax.set_ylabel('RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()

    if output_path is None:
        emb_type = df['embedding_type'].iloc[0] if 'embedding_type' in df.columns else 'unknown'
        output_path = os.path.join(RESULT_DIR, f'rmse_comparison_{emb_type}.pdf')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OUTPUT] Saved plot: {output_path}")
    plt.close()


def plot_auc_comparison(df, output_path=None):
    """
    Bar plot comparing AUC at n=1 vs n=max.

    Shows performance of Global Mean, Rasch-IRT, and Amortized IRT
    when using minimal vs maximal training data.
    """
    if df is None or len(df) == 0:
        print("No data for AUC comparison plot")
        return

    n_min = df['n_samples'].min()
    n_max = df['n_samples'].max()

    row_min = df[df['n_samples'] == n_min].iloc[0]
    row_max = df[df['n_samples'] == n_max].iloc[0]

    models = ['Global Mean', 'Rasch-IRT', 'Amortized IRT']
    auc_n1 = [row_min['auc_mean'], row_min['auc_rasch'], row_min['auc_amortized']]
    auc_nmax = [row_max['auc_mean'], row_max['auc_rasch'], row_max['auc_amortized']]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, auc_n1, width, label=f'n={n_min}', color=muted_blue)
    bars2 = ax.bar(x + width/2, auc_nmax, width, label=f'n={n_max}', color=muted_orange)

    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()

    if output_path is None:
        emb_type = df['embedding_type'].iloc[0] if 'embedding_type' in df.columns else 'unknown'
        output_path = os.path.join(RESULT_DIR, f'auc_comparison_{emb_type}.pdf')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OUTPUT] Saved plot: {output_path}")
    plt.close()


def plot_rmse_convergence(output_path=None):
    """
    Line plot showing RMSE vs number of training samples for RAW, PCA, and SAE embeddings.
    """
    csv_files = {
        'PCA': os.path.join(RESULT_DIR, 'amortized_irt_pca_beta.csv'),
        'SAE': os.path.join(RESULT_DIR, 'amortized_irt_sae_beta.csv'),
        'RAW': os.path.join(RESULT_DIR, 'amortized_irt_raw_beta.csv')
    }
    
    data = {}
    for name, path in csv_files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)

    if not data:
        print("No CSV data found to plot RMSE convergence.")
        return

    base_df = next(iter(data.values()))
    with optional_rc_context():
        fig, ax = plt.subplots(figsize=(6.75, 3))
    
    ax.plot(base_df['n_samples'], base_df['rmse_mean'], 'k--', label='Global Mean')
    ax.plot(base_df['n_samples'], base_df['rmse_rasch'], 'k-', label='Rasch-IRT')
    
    colors = {'PCA': muted_blue, 'SAE': muted_red, 'RAW': muted_green}
    for name, df in data.items():
        if 'rmse_amortized' in df.columns:
            ax.plot(df['n_samples'], df['rmse_amortized'], '^-', color=colors[name], label=f'Amortized IRT ({name})', markersize=4)

    ax.set_xlabel('Number of Response Matrix Samples (N)')
    ax.set_ylabel('Test RMSE')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(RESULT_DIR, 'rmse_convergence.pdf')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OUTPUT] Saved plot: {output_path}")
    plt.close()

def plot_auc_convergence(output_path=None):
    """
    Line plot showing AUC vs number of training samples for RAW, PCA, and SAE embeddings.
    """
    csv_files = {
        'PCA': os.path.join(RESULT_DIR, 'amortized_irt_pca_beta.csv'),
        'SAE': os.path.join(RESULT_DIR, 'amortized_irt_sae_beta.csv'),
        'RAW': os.path.join(RESULT_DIR, 'amortized_irt_raw_beta.csv')
    }
    
    data = {}
    for name, path in csv_files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)

    if not data:
        print("No CSV data found to plot AUC convergence.")
        return

    base_df = next(iter(data.values()))
    with optional_rc_context():
        fig, ax = plt.subplots(figsize=(6.75, 3))
        
        ax.plot(base_df['n_samples'], base_df['auc_mean'], 'k--', label='Global Mean')
        ax.plot(base_df['n_samples'], base_df['auc_rasch'], 'k-', label='Rasch-IRT')
        
        colors = {'PCA': muted_blue, 'SAE': muted_red, 'RAW': muted_green}
        for name, df in data.items():
            if 'auc_amortized' in df.columns:
                ax.plot(df['n_samples'], df['auc_amortized'], '^-', color=colors[name], label=f'Amortized IRT ({name})', markersize=4)

        ax.set_xlabel('Number of Response Matrix Samples (N)')
        ax.set_ylabel('Test AUC')
        ax.set_ylim(0.5, 1)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(RESULT_DIR, 'auc_convergence.pdf')

        plt.savefig(output_path, bbox_inches='tight')
        print(f"[OUTPUT] Saved plot: {output_path}")
        plt.close()


def plot_active_dims(df, output_path=None):
    """
    Line plot showing number of active dimensions vs n_samples.

    Visualizes how the model's effective dimensionality changes
    with the amount of training data.
    """
    if df is None or len(df) == 0 or 'active_dims' not in df.columns:
        print("No data for active dims plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df['n_samples'], df['active_dims'], 'o-', color=muted_green, markersize=4)

    ax.set_xlabel('Number of Samples (n)')
    ax.set_ylabel('Active Dimensions')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if output_path is None:
        emb_type = df['embedding_type'].iloc[0] if 'embedding_type' in df.columns else 'unknown'
        output_path = os.path.join(RESULT_DIR, f'active_dims_{emb_type}.pdf')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OUTPUT] Saved plot: {output_path}")
    plt.close()





def plot_benchmark_panels(output_dir=None):
    if output_dir is None:
        output_dir = RESULT_DIR
        
    benchmarks = ['scienceagentbench', 'scicode', 'corebench_hard']
    titles = ['ScienceAgentBench', 'SciCode', 'CoreBench-Hard']
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    post_rev_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'post-revision')
    
    data = []
    for b in benchmarks:
        b_dir = os.path.join(post_rev_dir, b, 'resmat')
        if not os.path.exists(b_dir):
            data.append((None, None))
            continue
            
        dfs = []
        prefix0_df = None
        for f in sorted(os.listdir(b_dir)):
            if not f.startswith('resmat'): continue
            df = pd.read_csv(os.path.join(b_dir, f), index_col=0)
            dfs.append(df)
            if '0.csv' in f and prefix0_df is None:
                prefix0_df = df
                
        if not dfs:
            data.append((None, None))
            continue
            
        if prefix0_df is None:
            prefix0_df = dfs[0]

        all_cols = sorted(list(set().union(*[df.columns for df in dfs])))
        shared_idx = sorted(list(set.intersection(*[set(df.index) for df in dfs])))
        
        aligned_dfs = [df.loc[shared_idx].reindex(columns=all_cols) for df in dfs]
        stacked = np.array([df.values for df in aligned_dfs], dtype=float)
        beta_mean = np.nanmean(stacked, axis=0)
        
        beta_df = pd.DataFrame(beta_mean, index=shared_idx, columns=all_cols)
        prefix0_clean = prefix0_df.loc[shared_idx].reindex(columns=all_cols)
        data.append((prefix0_clean, beta_df))

    if not any(pre is not None for pre, _ in data):
        print("No multi-benchmark matrices found for panel.")
        return

    # Panel Pre
    with optional_rc_context():
        fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), layout='constrained')
        for i, (ax, (pre, _), title) in enumerate(zip(axes, data, titles)):
            if pre is not None:
                sns.heatmap(pre, cmap=mcolors.ListedColormap(["#d62728", "#1f77b4"]), ax=ax, cbar=False, vmin=0, vmax=1)
                ax.set_title(f"{title} ($Y_1$)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        plt.savefig(os.path.join(output_dir, 'hal_response_matrix_panel_pre.pdf'))
        plt.close()

    # Panel Post
    with optional_rc_context():
        fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.5), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]}, layout='constrained')
        for i, (ax, (_, post), title) in enumerate(zip(axes[:-1], data, titles)):
            if post is not None:
                sns.heatmap(post, cmap=plt.cm.RdBu, ax=ax, cbar=False, vmin=0, vmax=1)
                ax.set_title(f"{title} ($\\hat{{P}}$)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                
        # Shared color palette bar
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu), cax=axes[-1])
        cbar.set_label('Target Probability')
        
        plt.savefig(os.path.join(output_dir, 'hal_response_matrix_panel_post.pdf'))
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate plots from amortized IRT results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory for final paper figures')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    out_dir = args.output_dir or RESULT_DIR

    # Generate response matrix visualizations (ColBench Y1 & P_hat)
    print("\n[Response Matrix] Generating Colbench response and probability matrix plots...")
    plot_response_and_probability_matrices(out_dir)
    
    # Generate unified panel visuals for scicode, scienceagentbench, corebench
    print("\n[Response Matrix] Generating continuous multi-benchmark panels...")
    plot_benchmark_panels(out_dir)



    # Plot AUC and RMSE curves
    print("\n[Amortized IRT] Generating performance learning curves for all embedding types...")
    plot_rmse_convergence(os.path.join(out_dir, 'rmse_convergence.pdf'))
    plot_auc_convergence(os.path.join(out_dir, 'auc_comparison.pdf'))

    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
