"""
Plot Judge Iterations

Generates comparison plots showing rubric satisfaction rates
across multiple evaluation iterations for different benchmarks.

Uses tueplots icml2024 bundle for consistent styling.

Usage:
    python plot_judge_iterations.py
    python plot_judge_iterations.py --data-dir /path/to/judge_output
"""

import argparse
import glob
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Try to use tueplots
try:
    from tueplots import bundles
    HAS_TUEPLOTS = True
except ImportError:
    HAS_TUEPLOTS = False
    warnings.warn("tueplots not installed. Using fallback styling.")

warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, '..', 'result')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data-reeval-multi', 'hal', 'judge_output')

# Color palette
COLORS = sns.color_palette("muted")


def setup_plotting_style():
    """Configure matplotlib with ICML 2024 style."""
    if HAS_TUEPLOTS:
        style = bundles.icml2024(usetex=False, family='serif')
        plt.rcParams.update(style)

    custom_params = {
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'figure.figsize': (8, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    plt.rcParams.update(custom_params)


def discover_datasets(data_dir):
    """
    Discover CSV files and group by dataset prefix.

    Returns:
        dict: {prefix: {iteration: filepath}}
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    dataset_iters = {}

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        m = re.match(r'^(.+)_(\d+)\.csv$', fname)
        if not m:
            continue
        prefix, suffix = m.group(1), int(m.group(2))
        dataset_iters.setdefault(prefix, {})[suffix] = fpath

    # Keep only datasets with more than 1 iteration
    multi_iter = {k: v for k, v in dataset_iters.items() if len(v) > 1}
    return multi_iter


def compute_rubric_stats(dataset_iters):
    """
    Compute rubric satisfaction statistics per iteration.

    Returns:
        DataFrame with columns: Dataset, Iteration, Rate, Count, Total
    """
    display_names = {
        'colbench': 'ColBench',
        'corebench': 'CoreBench',
        'sab': 'SAB',
        'scicode': 'SciCode',
    }

    records = []
    for prefix in sorted(dataset_iters):
        iters = dataset_iters[prefix]
        for suffix in sorted(iters):
            df = pd.read_csv(iters[suffix])
            if 'satisfies_rubric' not in df.columns:
                print(f"[WARNING] {iters[suffix]} missing 'satisfies_rubric' column")
                continue

            rate = df['satisfies_rubric'].mean()
            count = df['satisfies_rubric'].sum()
            total = len(df)
            label = display_names.get(prefix, prefix)
            records.append({
                'Dataset': label,
                'Iteration': suffix,
                'Rate': rate,
                'Count': int(count),
                'Total': total,
            })
            print(f"  {label} iter {suffix}: {count}/{total} = {rate:.2%}")

    return pd.DataFrame(records)


def plot_iteration_comparison(df_plot, output_file):
    """
    Generate grouped bar chart comparing iterations across datasets.
    """
    if df_plot.empty:
        print("[WARNING] No data to plot")
        return

    max_iter = df_plot['Iteration'].max()
    iter_colors = [COLORS[i] for i in range(max_iter)]

    datasets = df_plot['Dataset'].unique()
    n_datasets = len(datasets)

    fig, ax = plt.subplots()

    bar_width = 0.22
    offsets = np.arange(n_datasets, dtype=float)

    for i, it in enumerate(sorted(df_plot['Iteration'].unique())):
        subset = df_plot[df_plot['Iteration'] == it]
        counts = []
        positions = []

        for j, ds in enumerate(datasets):
            row = subset[subset['Dataset'] == ds]
            if not row.empty:
                counts.append(row['Count'].values[0])
                positions.append(offsets[j] + i * bar_width)

        ax.bar(positions, counts, width=bar_width, label=f'Iter {it}',
               color=iter_colors[i], edgecolor='white', linewidth=0.5)

        # Value labels
        for pos, count in zip(positions, counts):
            ax.text(pos, count + 0.5, f'{int(count)}', ha='center', va='bottom',
                    fontsize=11)

    # Center tick labels
    center_offset = bar_width * (max_iter - 1) / 2
    ax.set_xticks(offsets + center_offset)
    ax.set_xticklabels(datasets)

    ax.set_ylabel('Number of faulty questions')
    ax.set_ylim(0, df_plot['Count'].max() * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Iteration', loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\n[OUTPUT] Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot judge iterations')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory containing judge output CSVs')
    parser.add_argument('--output-dir', type=str, default=RESULT_DIR,
                        help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_plotting_style()

    print("=" * 60)
    print("JUDGE ITERATION ANALYSIS")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        return

    # Discover datasets
    dataset_iters = discover_datasets(args.data_dir)
    print(f"\n[INFO] Datasets with multiple iterations: {list(dataset_iters.keys())}")

    if not dataset_iters:
        print("[WARNING] No datasets with multiple iterations found")
        return

    # Compute stats
    print("\nComputing rubric satisfaction rates:")
    df_plot = compute_rubric_stats(dataset_iters)

    # Generate plot
    output_file = os.path.join(args.output_dir, 'judge_iteration_comparison.pdf')
    plot_iteration_comparison(df_plot, output_file)

    # Save stats to CSV
    stats_file = os.path.join(args.output_dir, 'judge_iteration_stats.csv')
    df_plot.to_csv(stats_file, index=False)
    print(f"[OUTPUT] Saved stats: {stats_file}")


if __name__ == '__main__':
    main()
