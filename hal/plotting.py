#!/usr/bin/env python3
"""
Plotting for Amortized IRT Experiments

Generates plots from CSV results produced by amortized_irt.py:
  - Comparison bar plots (n=1 vs n=max for RMSE and AUC)
  - Convergence line plots (metrics vs number of samples)
  - HELM benchmark comparison plots

Usage:
    python plotting.py                           # Generate all plots
    python plotting.py --csv result/custom.csv   # Use specific CSV
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sys.path.append('..')
import style_icml

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


def plot_rmse_convergence(df, output_path=None):
    """
    Line plot showing RMSE vs number of training samples.

    Visualizes how prediction accuracy improves as more
    response matrix samples become available.
    """
    if df is None or len(df) == 0:
        print("No data for RMSE convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df['n_samples'], df['rmse_mean'], 'o-', label='Global Mean',
            color=muted_blue, markersize=4)
    ax.plot(df['n_samples'], df['rmse_rasch'], 's-', label='Rasch-IRT',
            color=muted_orange, markersize=4)
    ax.plot(df['n_samples'], df['rmse_amortized'], '^-', label='Amortized IRT',
            color=muted_green, markersize=4)

    ax.set_xlabel('Number of Samples (n)')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if output_path is None:
        emb_type = df['embedding_type'].iloc[0] if 'embedding_type' in df.columns else 'unknown'
        output_path = os.path.join(RESULT_DIR, f'rmse_convergence_{emb_type}.pdf')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OUTPUT] Saved plot: {output_path}")
    plt.close()


def plot_auc_convergence(df, output_path=None):
    """
    Line plot showing AUC vs number of training samples.

    Visualizes how prediction accuracy improves as more
    response matrix samples become available.
    """
    if df is None or len(df) == 0:
        print("No data for AUC convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df['n_samples'], df['auc_mean'], 'o-', label='Global Mean',
            color=muted_blue, markersize=4)
    ax.plot(df['n_samples'], df['auc_rasch'], 's-', label='Rasch-IRT',
            color=muted_orange, markersize=4)
    ax.plot(df['n_samples'], df['auc_amortized'], '^-', label='Amortized IRT',
            color=muted_green, markersize=4)

    ax.set_xlabel('Number of Samples (n)')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if output_path is None:
        emb_type = df['embedding_type'].iloc[0] if 'embedding_type' in df.columns else 'unknown'
        output_path = os.path.join(RESULT_DIR, f'auc_convergence_{emb_type}.pdf')

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


# ══════════════════════════════════════════════════════════════════════════════
# HELM Benchmark Plot (Hard-coded data)
# ══════════════════════════════════════════════════════════════════════════════

def plot_helm_auc_comparison(output_path=None):
    """
    Bar plot for HELM benchmark AUC comparison.
    Uses hard-coded data from HELM experiments.
    """
    data = {
        'Model': [
            'Average', 'Rasch-IRT', 'Amortised Difficulty',
            'Sub-Amortised IRT', 'Amortised IRT'
        ],
        'AUC': [0.6579, 0.6539, 0.7577, 0.7823, 0.8122],
    }
    df_helm = pd.DataFrame(data)
    model_order = ['Average', 'Rasch-IRT', 'Amortised Difficulty', 'Sub-Amortised IRT', 'Amortised IRT']

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df_helm,
        x='Model',
        y='AUC',
        order=model_order,
        ax=ax,
        color=muted_blue
    )
    ax.set_xlabel('')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=15)

    for i, v in enumerate(df_helm.set_index('Model').loc[model_order]['AUC']):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(RESULT_DIR, 'auc_comparison_helm.pdf')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OUTPUT] Saved plot: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate plots from amortized IRT results')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to amortized_irt CSV results (default: auto-detect)')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Generate HELM comparison plot (always)
    print("\n[HELM] Generating HELM AUC comparison...")
    plot_helm_auc_comparison()

    # Load and plot amortized IRT results
    df = load_amortized_irt_results(args.csv)

    if df is not None and len(df) > 0:
        emb_type = df['embedding_type'].iloc[0] if 'embedding_type' in df.columns else 'unknown'
        print(f"\n[Amortized IRT] Generating plots for {emb_type} embeddings...")

        # Generate all amortized IRT plots
        plot_rmse_comparison(df)
        plot_auc_comparison(df)
        plot_rmse_convergence(df)
        plot_auc_convergence(df)
        plot_active_dims(df)
    else:
        print("\n[Amortized IRT] No CSV results found. Run amortized_irt.py first.")

    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
