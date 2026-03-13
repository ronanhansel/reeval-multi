#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles
from model.plotting import colors as pc

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESMAT_DIR = os.path.join(REPO_ROOT, 'item-editor', 'eval_response_matrix')
PRE_REV_DIR = os.path.join(RESMAT_DIR, 'pre-revision')
POST_REV_DIR = os.path.join(RESMAT_DIR, 'post-revision')
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)

# Professional Aesthetics from pc (shared colors)

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")

# ══════════════════════════════════════════════════════════════════════════════
# Data Collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_rubric_stats():
    benchmarks = ['colbench_backend_programming', 'corebench_hard', 'scienceagentbench', 'scicode']
    display_names = {
        'colbench_backend_programming': 'ColBench',
        'corebench_hard': 'CoreBench',
        'scienceagentbench': 'SAB',
        'scicode': 'SciCode'
    }
    
    results = []

    for bench in benchmarks:
        # Pre-revision
        pre_path = os.path.join(PRE_REV_DIR, bench, 'verdicts', 'verdict_original.csv')
        pre_val = 0
        if os.path.exists(pre_path):
            df = pd.read_csv(pre_path)
            # Find the row where agent == 'judge_verdict'
            judge_row = df[df['agent'] == 'judge_verdict']
            if not judge_row.empty:
                # Drop the 'agent' column and count 1s
                vals = judge_row.drop(columns=['agent']).values.flatten()
                vals = pd.to_numeric(vals, errors='coerce')
                pre_val = np.nanmean(vals == 1)
        
        # Post-revision
        post_bench_dir = os.path.join(POST_REV_DIR, bench, 'verdicts')
        post_vals = []
        if os.path.exists(post_bench_dir):
            files = [f for f in os.listdir(post_bench_dir) if f.startswith('verdict') and f.endswith('.csv')]
            for f in files:
                df = pd.read_csv(os.path.join(post_bench_dir, f))
                judge_row = df[df['agent'] == 'judge_verdict']
                if not judge_row.empty:
                    # Filter columns to only include those that belong to THIS benchmark
                    cols = [c for c in df.columns if c.startswith(bench + '.') or c == bench]
                    if not cols:
                        cols = [c for c in df.columns if c.startswith(bench)]
                    
                    if cols:
                        vals = judge_row[cols].values.flatten()
                        vals = pd.to_numeric(vals, errors='coerce')
                        post_vals.append(np.nanmean(vals == 1))

        post_mean = np.mean(post_vals) if post_vals else 0
        post_se = np.std(post_vals) / np.sqrt(len(post_vals)) if len(post_vals) > 1 else 0

        results.append({
            'Benchmark': display_names[bench],
            'Pre': pre_val,
            'Post_Mean': post_mean,
            'Post_SE': post_se
        })

    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def add_labels(ax, rects):
    """Add labels on top of bars, avoiding scientific notation for zero values."""
    for rect in rects:
        height = rect.get_height()
        label = f'{height:.2f}'
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

def plot_rubric_statistics(df):
    if df.empty:
        print("No data collected for rubric statistics.")
        return

    plt.rcParams.update(get_bundle())
    
    # Matching the reference image aspect ratio and style
    fig, ax = plt.subplots(figsize=(5, 5))
    
    x = np.arange(len(df['Benchmark']))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df['Pre'], width, label='Pre-Revision', 
                    color=pc.LIGHT_BLUE, alpha=1.0, edgecolor=pc.BAR_EDGE_COLOR, linewidth=1.0)
    rects2 = ax.bar(x + width/2, df['Post_Mean'], width, yerr=df['Post_SE'], label='Post-Revision', 
                    color=pc.LIGHT_GREEN, alpha=1.0, edgecolor=pc.BAR_EDGE_COLOR, linewidth=1.0, capsize=4, 
                    error_kw={'elinewidth': 1, 'capthick': 1})

    # Styling
    ax.set_ylabel('Fraction of Matched Items', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Benchmark'], fontsize=11)

    # Fixed Y-axis to match reference image
    ax.set_ylim(0, 0.30)
    ax.set_yticks(np.arange(0, 0.31, 0.05))
    ax.tick_params(axis='y', labelsize=10)
    
    # Remove top/right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Grid and Legend
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Legend with exact placement and style from image
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, borderpad=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'rubric_statistics.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"[OUTPUT] Saved refined rubric statistics plot: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("GENERATING RUBRIC STATISTICS PLOTS")
    print("=" * 60)
    
    df = collect_rubric_stats()
    print(df)
    plot_rubric_statistics(df)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
