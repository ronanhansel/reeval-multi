#!/usr/bin/env python3
"""
Sample Size vs. AUC Plot.
Shows how AUC improves as the number of test takers (agents) increases.
Generates two versions: Bernoulli (efficient sampling) and Beta (averaged probabilities).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)
RESULT_DIR = os.path.join(MODEL_DIR, 'result')
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Shared colors (matching colors.py)
BLUE = (0.12, 0.47, 0.71)
LIGHT_BLUE = (0.45, 0.62, 0.78)
GREEN = (0.17, 0.63, 0.17)
ORANGE = (1.0, 0.5, 0.05)
BASELINE_GRAY = '#7F8C8D'

# Common sizes for Bernoulli. Beta also includes 4.
SIZES_BERNOULLI = ['4', '8', '16', '32', '64', 'max']
X_VALS_BERNOULLI = [4, 8, 16, 32, 64, 143]

SIZES_BETA = ['4', '8', '16', '32', '64', 'max']
X_VALS_BETA = [4, 8, 16, 32, 64, 143]

MODELS = ['sae', 'pca', 'raw']
# J percentages to plot
J_PERCENTAGES = [0.25, 0.5, 1.0]
J_LABELS = {0.25: '25% Items', 0.5: '50% Items', 1.0: '100% Items'}
J_LINESTYLES = {0.25: ':', 0.5: '--', 1.0: '-'}

MODEL_LABELS = {'sae': 'ARAF (SAE)', 'pca': 'ARAF (PCA)', 'raw': 'ARAF (RAW)'}
MODEL_COLORS = {'sae': "salmon", 'pca': "skyblue", 'raw': "tab:blue"}

# ══════════════════════════════════════════════════════════════════════════════
# Logic
# ══════════════════════════════════════════════════════════════════════════════

def load_auc(setup, model, size, j_percentage=1.0):
    # Bernoulli uses n_1, Beta uses n_max
    n_suffix = "n_1" if setup == 'bernoulli' else "n_max"
    j_suffix = f"_j{j_percentage}" if j_percentage < 1.0 else ""
    filename = f"amortized_irt_{model}_{setup}_pre_{size}_{n_suffix}{j_suffix}.csv"
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        # Silently skip if file doesn't exist for a specific J-percentage
        return (None, None), (None, None)
    
    try:
        df = pd.read_csv(path)
        # Select rows with the best lambda_tau by mean AUC
        if 'lambda_tau' in df.columns:
            tau_stats = df.groupby('lambda_tau')['auc_amortized'].mean()
            best_tau = tau_stats.idxmax()
            best_df = df[df['lambda_tau'] == best_tau]
        else:
            best_df = df
            
        auc_mean = best_df['auc_amortized'].mean()
        auc_se = best_df['auc_amortized'].sem()
        
        # Rasch baseline (usually consistent across models for the same size)
        rasch_mean = best_df['auc_rasch'].mean()
        rasch_se = best_df['auc_rasch'].sem()
        
        return (auc_mean, auc_se), (rasch_mean, rasch_se)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return (None, None), (None, None)

def plot_setup(setup):
    print(f"Generating plot for {setup} setup...")
    
    sizes = SIZES_BERNOULLI if setup == 'bernoulli' else SIZES_BETA
    x_vals = X_VALS_BERNOULLI if setup == 'bernoulli' else X_VALS_BETA
    
    model_data = {m: [] for m in MODELS}
    rasch_data = [] 
    
    for i, size in enumerate(sizes):
        rasch_vals = []
        for model in MODELS:
            m_auc, r_auc = load_auc(setup, model, size)
            if m_auc[0] is not None:
                model_data[model].append((x_vals[i], m_auc[0], m_auc[1]))
            if r_auc[0] is not None:
                rasch_vals.append(r_auc)
        
        if rasch_vals:
            r_means = [v[0] for v in rasch_vals]
            avg_rasch_mean = np.mean(r_means)
            avg_rasch_se = np.mean([v[1] for v in rasch_vals if v[1] is not None])
            rasch_data.append((x_vals[i], avg_rasch_mean, avg_rasch_se))

    # Plotting setup
    plt.rcParams.update(bundles.icml2024(usetex=False, family="serif"))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # 1. Naive Baseline (dashed at 0.5)
    ax.axhline(0.5, color=BASELINE_GRAY, linestyle='--', linewidth=1, alpha=0.8)
    # Adjust position slightly for N=4 if needed, but 8.2 is fine
    label_x = x_vals[0] + 0.2
    ax.text(label_x, 0.503, "Naive Baseline", color=BASELINE_GRAY, fontsize=9, fontweight='bold', va='bottom')
    
    # 2. Rasch
    if rasch_data:
        x_r, y_r, e_r = zip(*rasch_data)
        ax.errorbar(x_r, y_r, yerr=e_r, color='#95a5a6', label='Rasch IRT', 
                   marker='o', linestyle='-', capsize=3, markersize=4, alpha=0.8)
    
    # 3. Amortized Models (Multiple J curves)
    for model in MODELS:
        for j_pct in J_PERCENTAGES:
            j_data = []
            for i, size in enumerate(sizes):
                m_auc, _ = load_auc(setup, model, size, j_percentage=j_pct)
                if m_auc[0] is not None:
                    j_data.append((x_vals[i], m_auc[0], m_auc[1]))
            
            if j_data:
                x, y, e = zip(*j_data)
                label = f"{MODEL_LABELS[model]} ({J_LABELS[j_pct]})" if len(J_PERCENTAGES) > 1 else MODEL_LABELS[model]
                ax.errorbar(x, y, yerr=e, color=MODEL_COLORS[model], label=label, 
                           marker='o', linestyle=J_LINESTYLES[j_pct], capsize=3, markersize=4, 
                           linewidth=1.5, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Number of Test Takers ($N$)', fontsize=10)
    ax.set_ylabel('AUC', fontsize=10)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(x_vals)
    # Style xticks
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    xticklabels = [str(v) for v in x_vals[:-1]] + ['143']
    ax.set_xticklabels(xticklabels)
    
    ax.set_ylim(0.48, 0.78)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend - handle duplicate labels if J_PERCENTAGES > 1
    handles, labels = ax.get_legend_handles_labels()
    # If too many labels, only label J percentages for SAE or something?
    # For now just show all.
    ax.legend(fontsize=6, frameon=True, loc='best', ncol=2)
    
    plt.tight_layout()
    
    # Save
    out_pdf = os.path.join(FIGURE_DIR, f'sample_size_{setup}.pdf')
    out_png = os.path.join(RESULT_DIR, f'sample_size_{setup}.png')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Success! {setup} plot generated: {out_pdf}")

def main():
    for setup in ['bernoulli', 'beta']:
        plot_setup(setup)

if __name__ == "__main__":
    main()
