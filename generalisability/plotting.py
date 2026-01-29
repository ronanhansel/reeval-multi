"""
Unified Plotting Module

Generates all plots for the generalisability experiments with consistent
ICML 2024 styling via tueplots.

All plots use:
- tueplots icml2024 bundle
- Font size 15
- Consistent color palette

Usage:
    python plotting.py                    # Generate all available plots
    python plotting.py --plot helm        # Generate only HELM plots
    python plotting.py --plot colbench    # Generate only ColBench plots
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Try to use tueplots, fall back to manual settings if not available
try:
    from tueplots import bundles
    HAS_TUEPLOTS = True
except ImportError:
    HAS_TUEPLOTS = False
    warnings.warn("tueplots not installed. Using fallback styling. Install with: pip install tueplots")

warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

# Consistent color palette
COLORS = sns.color_palette("muted")
MUTED_BLUE = COLORS[0]
MUTED_ORANGE = COLORS[1]
MUTED_GREEN = COLORS[2]
MUTED_RED = COLORS[3]


def setup_plotting_style():
    """
    Configure matplotlib with ICML 2024 style.
    Uses tueplots if available, otherwise applies manual configuration.
    """
    if HAS_TUEPLOTS:
        # Use tueplots icml2024 bundle
        style = bundles.icml2024(usetex=False, family='serif')
        plt.rcParams.update(style)

    # Override with our consistent settings
    custom_params = {
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'figure.figsize': (6, 3.5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    plt.rcParams.update(custom_params)


def plot_helm_auc_comparison(results_file=None, output_file=None):
    """
    Generate HELM AUC comparison bar chart.

    Args:
        results_file: Path to helm_results.csv (or uses default)
        output_file: Output PDF path (or uses default)
    """
    if results_file is None:
        results_file = os.path.join(RESULT_DIR, 'helm_results.csv')

    if output_file is None:
        output_file = os.path.join(RESULT_DIR, 'auc_comparison_helm.pdf')

    # Load results or use fallback
    if os.path.exists(results_file):
        print(f"[INFO] Loading HELM results from: {results_file}")
        df = pd.read_csv(results_file)
    else:
        print(f"[WARNING] {results_file} not found. Using fallback values.")
        print("[WARNING] Run 'python models.py --benchmark helm' first for fitted results.")
        df = pd.DataFrame({
            'Model': ['Average', 'Rasch-IRT', 'Amortised Difficulty',
                      'Sub-Amortised IRT', 'Amortised IRT'],
            'AUC': [0.6579, 0.6539, 0.7577, 0.7823, 0.8122]
        })

    # Define model order
    model_order = ['Average', 'Rasch-IRT', 'Amortised Difficulty',
                   'Sub-Amortised IRT', 'Amortised IRT']
    available_models = [m for m in model_order if m in df['Model'].values]

    if not available_models:
        print("[ERROR] No matching models found in results!")
        return

    # Create plot
    fig, ax = plt.subplots()
    bars = ax.bar(
        range(len(available_models)),
        [df.loc[df['Model'] == m, 'AUC'].values[0] for m in available_models],
        color=MUTED_BLUE,
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_xticks(range(len(available_models)))
    ax.set_xticklabels(available_models, rotation=15, ha='right')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[OUTPUT] Saved: {output_file}")
    plt.close()


def plot_colbench_comparison(results_file=None, output_dir=None):
    """
    Generate ColBench comparison plots (RMSE and AUC).

    Args:
        results_file: Path to colbench_results.csv
        output_dir: Directory for output files
    """
    if results_file is None:
        results_file = os.path.join(RESULT_DIR, 'colbench_results.csv')

    if output_dir is None:
        output_dir = RESULT_DIR

    if not os.path.exists(results_file):
        print(f"[WARNING] {results_file} not found. Run 'python models.py --benchmark colbench' first.")
        return

    print(f"[INFO] Loading ColBench results from: {results_file}")
    df = pd.read_csv(results_file)

    model_order = ['Global Mean', 'Rasch-IRT', 'Amortized IRT']
    available_models = [m for m in model_order if m in df['Model'].values]

    # RMSE comparison
    if 'RMSE' in df.columns:
        fig, ax = plt.subplots()
        rmse_vals = [df.loc[df['Model'] == m, 'RMSE'].values[0] for m in available_models]
        bars = ax.bar(range(len(available_models)), rmse_vals, color=MUTED_ORANGE,
                      edgecolor='white', linewidth=0.5)

        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels(available_models, rotation=15, ha='right')
        ax.set_ylabel('RMSE')
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        out = os.path.join(output_dir, 'rmse_comparison_colbench.pdf')
        plt.savefig(out)
        print(f"[OUTPUT] Saved: {out}")
        plt.close()

    # AUC comparison
    if 'AUC' in df.columns:
        fig, ax = plt.subplots()
        auc_vals = [df.loc[df['Model'] == m, 'AUC'].values[0] for m in available_models]
        bars = ax.bar(range(len(available_models)), auc_vals, color=MUTED_BLUE,
                      edgecolor='white', linewidth=0.5)

        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels(available_models, rotation=15, ha='right')
        ax.set_ylabel('AUC')
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        out = os.path.join(output_dir, 'auc_comparison_colbench.pdf')
        plt.savefig(out)
        print(f"[OUTPUT] Saved: {out}")
        plt.close()


def plot_convergence(results_file=None, output_dir=None, metric='rmse'):
    """
    Generate convergence line plots for experiments with varying n_samples.

    Args:
        results_file: Path to convergence results CSV
        output_dir: Output directory
        metric: 'rmse' or 'auc'
    """
    if results_file is None:
        results_file = os.path.join(RESULT_DIR, 'convergence_results.csv')

    if output_dir is None:
        output_dir = RESULT_DIR

    if not os.path.exists(results_file):
        print(f"[WARNING] {results_file} not found. Skipping convergence plots.")
        return

    df = pd.read_csv(results_file)

    if len(df) < 2:
        print("[WARNING] Need multiple n values for convergence plots.")
        return

    fig, ax = plt.subplots()

    if metric == 'rmse':
        ax.plot(df['n_samples'], df['rmse_mean'], 's--', label='Global Mean',
                color='gray', linewidth=1.5, markersize=6)
        ax.plot(df['n_samples'], df['rmse_rasch'], '^--', label='Rasch-IRT',
                color=MUTED_RED, linewidth=1.5, markersize=6)
        ax.plot(df['n_samples'], df['rmse_amortized'], 'o-', label='Amortized IRT',
                color=MUTED_BLUE, linewidth=1.5, markersize=6)
        ax.set_ylabel('RMSE')
        out_name = 'rmse_convergence.pdf'
    else:
        ax.plot(df['n_samples'], df['auc_mean'], 's--', label='Global Mean',
                color='gray', linewidth=1.5, markersize=6)
        ax.plot(df['n_samples'], df['auc_rasch'], '^--', label='Rasch-IRT',
                color=MUTED_RED, linewidth=1.5, markersize=6)
        ax.plot(df['n_samples'], df['auc_amortized'], 'o-', label='Amortized IRT',
                color=MUTED_BLUE, linewidth=1.5, markersize=6)
        ax.set_ylabel('AUC')
        out_name = 'auc_convergence.pdf'

    ax.set_xlabel('Number of samples')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, out_name)
    plt.savefig(out)
    print(f"[OUTPUT] Saved: {out}")
    plt.close()


def plot_n_comparison(results_file=None, output_dir=None):
    """
    Generate grouped bar charts comparing n=1 vs n=max.

    Args:
        results_file: Path to convergence results CSV
        output_dir: Output directory
    """
    if results_file is None:
        results_file = os.path.join(RESULT_DIR, 'convergence_results.csv')

    if output_dir is None:
        output_dir = RESULT_DIR

    if not os.path.exists(results_file):
        print(f"[WARNING] {results_file} not found. Skipping comparison plots.")
        return

    df = pd.read_csv(results_file)
    n_max = df['n_samples'].max()
    df_comp = df[df['n_samples'].isin([1, n_max])].copy()

    if len(df_comp) < 2:
        print("[WARNING] Need both n=1 and n=max for comparison plots.")
        return

    df_comp['n_label'] = df_comp['n_samples'].apply(lambda x: f"n={x}")

    # RMSE comparison
    fig, ax = plt.subplots()

    model_names = ['Global Mean', 'Rasch-IRT', 'Amortized IRT']
    rmse_cols = ['rmse_mean', 'rmse_rasch', 'rmse_amortized']

    x = np.arange(len(model_names))
    width = 0.35

    for i, (_, row) in enumerate(df_comp.iterrows()):
        offset = (i - 0.5) * width
        vals = [row[c] for c in rmse_cols]
        bars = ax.bar(x + offset, vals, width, label=row['n_label'],
                      color=COLORS[i], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, 'rmse_comparison.pdf')
    plt.savefig(out)
    print(f"[OUTPUT] Saved: {out}")
    plt.close()

    # AUC comparison
    fig, ax = plt.subplots()

    auc_cols = ['auc_mean', 'auc_rasch', 'auc_amortized']

    for i, (_, row) in enumerate(df_comp.iterrows()):
        offset = (i - 0.5) * width
        vals = [row[c] for c in auc_cols]
        bars = ax.bar(x + offset, vals, width, label=row['n_label'],
                      color=COLORS[i], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, 'auc_comparison.pdf')
    plt.savefig(out)
    print(f"[OUTPUT] Saved: {out}")
    plt.close()


def plot_all():
    """Generate all available plots."""
    print("=" * 60)
    print("GENERATING ALL PLOTS")
    print("=" * 60)

    setup_plotting_style()

    # HELM plots
    plot_helm_auc_comparison()

    # ColBench plots
    plot_colbench_comparison()

    # Convergence plots (if data available)
    plot_convergence(metric='rmse')
    plot_convergence(metric='auc')
    plot_n_comparison()


def main():
    parser = argparse.ArgumentParser(description='Generate plots')
    parser.add_argument('--plot', type=str, default='all',
                        choices=['all', 'helm', 'colbench', 'convergence'],
                        help='Which plots to generate (default: all)')
    args = parser.parse_args()

    setup_plotting_style()

    if args.plot == 'all':
        plot_all()
    elif args.plot == 'helm':
        plot_helm_auc_comparison()
    elif args.plot == 'colbench':
        plot_colbench_comparison()
    elif args.plot == 'convergence':
        plot_convergence(metric='rmse')
        plot_convergence(metric='auc')
        plot_n_comparison()


if __name__ == '__main__':
    main()
