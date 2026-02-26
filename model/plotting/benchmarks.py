#!/usr/bin/env python3
"""
Benchmark Matrix Visualizations for Amortized IRT.
Focuses on raw response matrices (Y) and empirical probability matrices (P_hat).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tueplots import bundles

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")

# ══════════════════════════════════════════════════════════════════════════════
# Logic
# ══════════════════════════════════════════════════════════════════════════════

def load_response_matrices():
    resmat_dir = os.path.join(REPO_ROOT, 'item-editor', 'eval_response_matrix')
    post_rev_dir = os.path.join(resmat_dir, 'post-revision')
    colbench_dir = os.path.join(post_rev_dir, 'colbench_backend_programming', 'resmat')

    if not os.path.exists(colbench_dir):
        return None

    all_files = sorted([f for f in os.listdir(colbench_dir) if f.startswith('resmat')])
    if not all_files:
        return None

    return [pd.read_csv(os.path.join(colbench_dir, f), index_col=0) for f in all_files]

def visualize_binary_response_matrix(Y, output_path):
    df = Y.copy()
    row_order = df.mean(axis=1).sort_values(ascending=False).index
    df = df.loc[row_order]
    col_order = df.mean(axis=0).sort_values(ascending=False).index
    df = df[col_order]

    cmap = mcolors.ListedColormap(["#d62728", "#1f77b4"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    plt.rcParams.update(get_bundle())
    fig, ax = plt.subplots(figsize=(6.75, 2.5))
    cax = ax.imshow(df.values.astype(float), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_title("Binary Response Matrix $Y_1$")
    ax.set_xticks([]); ax.set_yticks([])
    
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0 (Fail)', '1 (Pass)'])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_probability_matrix(P_hat, output_path):
    df = P_hat.copy()
    row_order = df.mean(axis=1).sort_values(ascending=False).index
    df = df.loc[row_order]
    col_order = df.mean(axis=0).sort_values(ascending=False).index
    df = df[col_order]

    plt.rcParams.update(get_bundle())
    fig, ax = plt.subplots(figsize=(6.75, 2.5))
    cax = ax.imshow(df.values, aspect='auto', cmap=plt.cm.RdBu, vmin=0, vmax=1, interpolation='nearest')

    ax.set_title("Empirical Probability Matrix $\\hat{P}$")
    ax.set_xticks([]); ax.set_yticks([])
    
    plt.colorbar(cax, ax=ax, shrink=0.8).set_label('Probability')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmark_panels():
    benchmarks = ['scienceagentbench', 'scicode', 'corebench_hard']
    titles = ['ScienceAgentBench', 'SciCode', 'CoreBench-Hard']
    post_rev_dir = os.path.join(REPO_ROOT, 'item-editor', 'eval_response_matrix', 'post-revision')
    
    data = []
    for b in benchmarks:
        b_dir = os.path.join(post_rev_dir, b, 'resmat')
        if not os.path.exists(b_dir):
            data.append((None, None))
            continue
            
        dfs = [pd.read_csv(os.path.join(b_dir, f), index_col=0) for f in sorted(os.listdir(b_dir)) if f.startswith('resmat')]
        if not dfs:
            data.append((None, None))
            continue
            
        all_cols = sorted(list(set().union(*[df.columns for df in dfs])))
        shared_idx = sorted(list(set.intersection(*[set(df.index) for df in dfs])))
        aligned = [df.loc[shared_idx].reindex(columns=all_cols) for df in dfs]
        
        beta_df = pd.DataFrame(np.nanmean(np.array([df.values for df in aligned], dtype=float), axis=0), 
                              index=shared_idx, columns=all_cols)
        data.append((aligned[0], beta_df))

    plt.rcParams.update(get_bundle())
    
    # Pre Panel
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0), layout='constrained')
    for ax, (pre, _), title in zip(axes, data, titles):
        if pre is not None:
            sns.heatmap(pre, cmap=mcolors.ListedColormap(["#d62728", "#1f77b4"]), ax=ax, cbar=False, vmin=0, vmax=1)
            ax.set_title(f"{title} ($Y_1$)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(os.path.join(FIGURE_DIR, 'hal_response_matrix_panel_pre.pdf'))
    plt.close()

    # Post Panel
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.0), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]}, layout='constrained')
    for ax, (_, post), title in zip(axes[:-1], data, titles):
        if post is not None:
            sns.heatmap(post, cmap=plt.cm.RdBu, ax=ax, cbar=False, vmin=0, vmax=1)
            ax.set_title(f"{title} ($\\hat{{P}}$)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    
    norm = mcolors.Normalize(vmin=0, vmax=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu), cax=axes[-1]).set_label('Prob', fontsize=8)
    plt.savefig(os.path.join(FIGURE_DIR, 'hal_response_matrix_panel_post.pdf'))
    plt.close()

def main():
    print("Generating Benchmark Matrix Plots...")
    rm = load_response_matrices()
    if rm:
        visualize_binary_response_matrix(rm[0], os.path.join(FIGURE_DIR, 'response_matrix_Y1.png'))
        
        shared_indices = set(rm[0].index)
        for df in rm[1:]: shared_indices = shared_indices.intersection(set(df.index))
        shared_indices = sorted(list(shared_indices))
        stacked = np.array([df.loc[shared_indices].values for df in rm], dtype=float)
        p_hat = pd.DataFrame(np.nanmean(stacked, axis=0), index=shared_indices, columns=rm[0].columns)
        
        visualize_probability_matrix(p_hat, os.path.join(FIGURE_DIR, 'empirical_probability_matrix_P_hat.png'))
    
    plot_benchmark_panels()
    print(f"Done. Plots in {FIGURE_DIR}")

if __name__ == "__main__":
    main()
