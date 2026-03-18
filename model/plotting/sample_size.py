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
MODEL_LABELS = {'sae': 'A.SAE', 'pca': 'A.PCA', 'raw': 'A.RAW'}
MODEL_COLORS = {'sae': "lightblue", 'pca': "deepskyblue", 'raw': "steelblue"}

# ══════════════════════════════════════════════════════════════════════════════
# Logic
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics(setup, model, size):
    """Loads AUC and RMSE metrics for a given model and setup."""
    # Bernoulli uses n_1 (minimal sampling), Beta uses n_max (full averaging)
    n_suffix = "n_1" if setup == 'bernoulli' else "n_max"
    
    # Try multiple patterns for the filename
    options = [
        f"amortized_irt_{model}_{setup}_pre_{size}_{n_suffix}.csv",
        f"amortized_irt_{model}_{setup}_n_{size}.csv" if setup == 'bernoulli' else f"amortized_irt_{model}_{setup}_n_max.csv"
    ]
    
    path = None
    for opt in options:
        p = os.path.join(RESULT_DIR, opt)
        if os.path.exists(p):
            path = p
            break
            
    if path is None:
        return None
    
    try:
        df = pd.read_csv(path)
        # Select rows with the best lambda_tau by mean AUC
        if 'lambda_tau' in df.columns:
            tau_stats = df.groupby('lambda_tau')['auc_amortized'].mean()
            best_tau = tau_stats.idxmax()
            best_df = df[df['lambda_tau'] == best_tau]
        else:
            best_df = df
            
        metrics = {}
        for metric in ['auc', 'rmse']:
            for suffix in ['amortized', 'rasch', '2pl', 'mirt']:
                col = f'{metric}_{suffix}'
                if col in best_df.columns:
                    metrics[col] = (best_df[col].mean(), best_df[col].sem())
                else:
                    metrics[col] = (None, None)
        return metrics
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def gather_data(setup):
    sizes = SIZES_BERNOULLI if setup == 'bernoulli' else SIZES_BETA
    x_vals = X_VALS_BERNOULLI if setup == 'bernoulli' else X_VALS_BETA
    
    model_data_auc = {m: [] for m in MODELS}
    model_data_rmse = {m: [] for m in MODELS}
    
    rasch_data_auc, rasch_data_rmse = [], []
    twopl_data_auc, twopl_data_rmse = [], []
    mirt_data_auc, mirt_data_rmse = [], []
    
    for i, size in enumerate(sizes):
        raw_vals = {met: {bm: [] for bm in ['rasch', '2pl', 'mirt']} for met in ['auc', 'rmse']}
        
        # Check standard models (they contain internal baselines)
        for model in MODELS:
            metrics = load_metrics(setup, model, size)
            if metrics:
                if metrics['auc_amortized'][0] is not None:
                    model_data_auc[model].append((x_vals[i], metrics['auc_amortized'][0], metrics['auc_amortized'][1]))
                if metrics['rmse_amortized'][0] is not None:
                    model_data_rmse[model].append((x_vals[i], metrics['rmse_amortized'][0], metrics['rmse_amortized'][1]))
                
                for met in ['auc', 'rmse']:
                    for bm in ['rasch', '2pl', 'mirt']:
                        if metrics[f'{met}_{bm}'][0] is not None:
                            raw_vals[met][bm].append(metrics[f'{met}_{bm}'])
        
        # Check standalone baseline files for scaling data
        for b_name in ['rasch_2pl', 'nonamortised_mirt']:
            metrics = load_metrics(setup, b_name, size)
            if metrics:
                for met in ['auc', 'rmse']:
                    for bm in ['rasch', '2pl', 'mirt']:
                        if metrics[f'{met}_{bm}'][0] is not None:
                            raw_vals[met][bm].append(metrics[f'{met}_{bm}'])

        # Aggregate baselines
        if raw_vals['auc']['rasch']:
            rasch_data_auc.append((x_vals[i], np.mean([v[0] for v in raw_vals['auc']['rasch']]), np.mean([v[1] for v in raw_vals['auc']['rasch'] if v[1] is not None])))
        if raw_vals['auc']['2pl']:
            twopl_data_auc.append((x_vals[i], np.mean([v[0] for v in raw_vals['auc']['2pl']]), np.mean([v[1] for v in raw_vals['auc']['2pl'] if v[1] is not None])))
        if raw_vals['auc']['mirt']:
            mirt_data_auc.append((x_vals[i], np.mean([v[0] for v in raw_vals['auc']['mirt']]), np.mean([v[1] for v in raw_vals['auc']['mirt'] if v[1] is not None])))

        if raw_vals['rmse']['rasch']:
            rasch_data_rmse.append((x_vals[i], np.mean([v[0] for v in raw_vals['rmse']['rasch']]), np.mean([v[1] for v in raw_vals['rmse']['rasch'] if v[1] is not None])))
        if raw_vals['rmse']['2pl']:
            twopl_data_rmse.append((x_vals[i], np.mean([v[0] for v in raw_vals['rmse']['2pl']]), np.mean([v[1] for v in raw_vals['rmse']['2pl'] if v[1] is not None])))
        if raw_vals['rmse']['mirt']:
            mirt_data_rmse.append((x_vals[i], np.mean([v[0] for v in raw_vals['rmse']['mirt']]), np.mean([v[1] for v in raw_vals['rmse']['mirt'] if v[1] is not None])))
            
    return {
        'auc': {
            'model_data': model_data_auc,
            'rasch': rasch_data_auc,
            '2pl': twopl_data_auc,
            'mirt': mirt_data_auc
        },
        'rmse': {
            'model_data': model_data_rmse,
            'rasch': rasch_data_rmse,
            '2pl': twopl_data_rmse,
            'mirt': mirt_data_rmse
        },
        'x_vals': x_vals
    }

def plot_on_ax(ax, data, setup, metric):
    res = data[metric]
    x_vals = data['x_vals']
    
    # 1. Base levels
    if metric == 'auc':
        ax.axhline(0.5, color=BASELINE_GRAY, linestyle='--', linewidth=1, alpha=0.8)
    
    # 2. Baselines
    if res['rasch']:
        x, y, e = zip(*res['rasch'])
        ax.errorbar(x, y, yerr=e, color='#95a5a6', label='Rasch (1PL)', 
                   marker='o', linestyle='--', capsize=2, markersize=3, alpha=0.6)
    
    if res['2pl']:
        x, y, e = zip(*res['2pl'])
        ax.errorbar(x, y, yerr=e, color='#7f8c8d', label='2PL IRT', 
                   marker='s', linestyle='--', capsize=2, markersize=3, alpha=0.6)
                   
    if res['mirt']:
        x, y, e = zip(*res['mirt'])
        ax.errorbar(x, y, yerr=e, color='#34495e', label='Standal. MIRT', 
                   marker='^', linestyle='--', capsize=2, markersize=3, alpha=0.6)
    
    # 3. Amortized Models
    for model in MODELS:
        m_data = res['model_data'][model]
        if m_data:
            x, y, e = zip(*m_data)
            ax.errorbar(x, y, yerr=e, color=MODEL_COLORS[model], label=MODEL_LABELS[model], 
                       marker='o', linestyle='-', capsize=2, markersize=4, 
                       linewidth=1.5, alpha=0.8)
    
    # Formatting
    t_map = {
        ('auc', 'bernoulli'): 'Bernoulli (AUC)',
        ('auc', 'beta'): 'Beta (AUC)',
        ('rmse', 'bernoulli'): 'Bernoulli (RMSE)',
        ('rmse', 'beta'): 'Beta (RMSE)'
    }
    ax.set_title(t_map.get((metric, setup), f"{setup} ({metric})"), fontsize=10, fontweight='bold')
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(x_vals)
    ax.tick_params(axis='both', labelsize=8)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    xticklabels = [str(v) for v in x_vals[:-1]] + ['143']
    ax.set_xticklabels(xticklabels)
    
    if metric == 'auc':
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_ylim(0.48, 0.78)
    else:
        # RMSE typically ranges from 0.4 to 0.6
        ax.set_yticks([0.4, 0.5, 0.6])
        ax.set_ylim(0.38, 0.65)
        
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

def plot_combined():
    print("Generating 4-panel horizontal plot for bernoulli and beta setups (AUC and RMSE)...")
    
    data_bernoulli = gather_data('bernoulli')
    data_beta = gather_data('beta')
    
    plt.rcParams.update(bundles.icml2024(usetex=False, family="serif"))
    # One row, four columns. Share Y within metric pairs.
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    (ax1, ax2, ax3, ax4) = axes
    
    # 1. AUC Panel
    plot_on_ax(ax1, data_bernoulli, 'bernoulli', 'auc')
    plot_on_ax(ax2, data_beta, 'beta', 'auc')
    ax1.set_ylabel('AUC', fontsize=10)
    
    # 2. RMSE Panel
    plot_on_ax(ax3, data_bernoulli, 'bernoulli', 'rmse')
    plot_on_ax(ax4, data_beta, 'beta', 'rmse')
    ax3.set_ylabel('RMSE', fontsize=10)
    
    # Sync Y scales for metric pairs
    ax2.set_ylim(ax1.get_ylim())
    ax4.set_ylim(ax3.get_ylim())
    
    # Shared X label
    fig.supxlabel('Number of Agents ($N$)', fontsize=10, y=0.18)
    
    # Shared legend - extract from one of the axes
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), 
               ncol=6, fontsize=8, frameon=True)
    
    plt.tight_layout()
    # Adjust layout to make room for legend and labels
    plt.subplots_adjust(bottom=0.32, top=0.88, wspace=0.35)
    
    out_pdf = os.path.join(FIGURE_DIR, 'sample_size_quad.pdf')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Success! 4-panel plot generated: {out_pdf}")

def main():
    plot_combined()

if __name__ == "__main__":
    main()
