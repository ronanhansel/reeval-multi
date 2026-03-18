#!/usr/bin/env python3
"""
plot_j_scaling.py — Visualize the Item Scaling Law (Variable J, Fixed N=32).
Plots AUC as a function of the percentage of items (J) used for training.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

# Configuration
RESULT_DIR = 'result'
FIGURE_DIR = '../paper/figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

MODELS = ['sae', 'pca', 'raw', 'rasch_2pl', 'nonamortised_mirt']
MODEL_LABELS = {
    'sae': 'ARAF (SAE)', 'pca': 'ARAF (PCA)', 'raw': 'ARAF (RAW)',
    'rasch_2pl': '2PL IRT', 'nonamortised_mirt': 'Non-Amort. MIRT'
}
MODEL_COLORS = {
    'sae': "salmon", 'pca': "skyblue", 'raw': "tab:blue",
    'rasch_2pl': '#7f8c8d', 'nonamortised_mirt': '#34495e'
}
J_PCTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def load_auc_for_j(setup, model, j_pct):
    n_suffix = "n_1" if setup == 'bernoulli' else "n_max"
    j_suffix = f"_j{j_pct}" if j_pct < 1.0 else ""
    # Note: Pre-32 is our target for N=32 scaling
    filename = f"amortized_irt_{model}_{setup}_pre_32_{n_suffix}{j_suffix}.csv"
    path = os.path.join(RESULT_DIR, filename)
    
    if not os.path.exists(path):
        return None, None
    
    try:
        df = pd.read_csv(path)
        # Assuming lambda_tau was fixed or we take the best
        if 'lambda_tau' in df.columns:
            tau_stats = df.groupby('lambda_tau')['auc_amortized'].mean()
            best_tau = tau_stats.idxmax()
            subset = df[df['lambda_tau'] == best_tau]
        else:
            subset = df
            
        return subset['auc_amortized'].mean(), subset['auc_amortized'].sem()
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def plot_j_scaling(setup):
    plt.rcParams.update(bundles.icml2024(usetex=False, family="serif"))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    for model in MODELS:
        means = []
        ses = []
        valid_js = []
        
        for j in J_PCTS:
            m, s = load_auc_for_j(setup, model, j)
            if m is not None:
                means.append(m)
                ses.append(s)
                valid_js.append(j * 100) # Percentage
        
        if means:
            ls = '--' if 'mirt' in model or '2pl' in model else '-'
            marker = '^' if 'mirt' in model else ('s' if '2pl' in model else 'o')
            ax.errorbar(valid_js, means, yerr=ses, color=MODEL_COLORS[model], 
                       label=MODEL_LABELS[model], marker=marker, linestyle=ls, 
                       capsize=3, markersize=3, linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Percentage of Items ($J\%$) at $N=32$', fontsize=10)
    ax.set_ylabel('AUC', fontsize=10)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_ylim(0.48, 0.80)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=7, frameon=True, loc='lower right')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURE_DIR, f'item_scaling_{setup}.pdf')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    plot_j_scaling('bernoulli')
    plot_j_scaling('beta')
