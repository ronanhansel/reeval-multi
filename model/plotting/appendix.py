#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles


# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)
RESULT_DIR = os.path.join(MODEL_DIR, "result")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures", "appendix")
RESULT_PATH = Path(RESULT_DIR)

os.makedirs(FIGURE_DIR, exist_ok=True)

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def _pick_latest_file(pattern, prefer_plain=True):
    candidates = list(RESULT_PATH.glob(pattern))
    if not candidates:
        return None

    # Tau-sensitivity plots should prefer tau-enabled runs.
    candidates = [p for p in candidates if "notau" not in p.name]
    if not candidates:
        return None

    # Prefer the canonical run without j-suffix when available.
    if prefer_plain:
        plain = [p for p in candidates if "_j" not in p.name]
        if plain:
            candidates = plain

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_suite_file(embedding_type, suite_name, variant):
    if suite_name == "max":
        patterns = {
            "bernoulli_pre": f"amortized_irt_{embedding_type}_bernoulli_pre_max_n_1*.csv",
            "beta_pre": f"amortized_irt_{embedding_type}_beta_pre_max_n_max*.csv",
            "bernoulli_post": f"amortized_irt_{embedding_type}_bernoulli_n_1*.csv",
            "beta_post": f"amortized_irt_{embedding_type}_beta_n_max*.csv",
        }
    else:  # n32
        patterns = {
            "bernoulli_pre": f"amortized_irt_{embedding_type}_bernoulli_pre_32_n_1*.csv",
            "beta_pre": f"amortized_irt_{embedding_type}_beta_pre_32_n_max*.csv",
            "bernoulli_post": f"amortized_irt_{embedding_type}_bernoulli_n_1*.csv",
            "beta_post": f"amortized_irt_{embedding_type}_beta_n_max*.csv",
        }

    pattern = patterns.get(variant)
    if pattern is None:
        return None
    return _pick_latest_file(pattern, prefer_plain=True)


# ══════════════════════════════════════════════════════════════════════════════
# Core Plotting Function
# ══════════════════════════════════════════════════════════════════════════════

def plot_merged_sensitivity_all_embeddings(suite_name):
    """
    Generates a fixed 3x3 sensitivity plot (3 metrics x 3 embeddings)
    for a specific suite (max or n32).
    """
    plt.rcParams.update(get_bundle())
    embedding_types = ["raw", "pca", "sae"]
    
    # Selection logic to mirror tau_sensitivity_n32_log_appendix.pdf datasets
    if suite_name == 'max':
        title_suffix = "(All Agents)"
    else:  # n32
        title_suffix = "(32 Agents)"

    # Rows will be metrics
    metrics = [
        ('auc_amortized', 'AUC', (0.58, 0.82)),
        ('rmse_amortized', 'RMSE', (0.20, 0.55)),
        ('active_dims', 'Active Dimensions ($K$)', (-1.5, 31.5))
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5), constrained_layout=True)

    legend_map = {}

    # Outer loop: Metrics (Rows)
    for row_idx, (col_name, metric_title, ylim) in enumerate(metrics):
        # Inner loop: Embedding types (Columns)
        for col_idx, embedding_type in enumerate(embedding_types):
            ax = axes[row_idx, col_idx]
            
            configs = [
                ('Bernoulli Pre-Revision', 'bernoulli_pre', '-', 'salmon'),
                ('Beta Pre-Revision', 'beta_pre', '-', 'tab:red'),
                ('Bernoulli Post-Revision', 'bernoulli_post', '-', 'skyblue'),
                ('Beta Post-Revision', 'beta_post', '-', 'tab:blue'),
            ]

            data = []
            for label, variant, ls, color in configs:
                file_path = _resolve_suite_file(embedding_type, suite_name, variant)
                if file_path is None:
                    continue
                
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip')
                    df.columns = df.columns.str.strip()
                    
                    required_cols = ['lambda_tau', 'auc_amortized', 'rmse_amortized', 'active_dims']
                    for col in required_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna(subset=['lambda_tau', 'auc_amortized', 'active_dims'])
                    if df.empty: continue

                    cols_to_agg = ['auc_amortized', 'active_dims']
                    if 'rmse_amortized' in df.columns:
                        cols_to_agg.append('rmse_amortized')

                    df_mean = df.groupby('lambda_tau')[cols_to_agg].mean().reset_index().sort_values('lambda_tau')
                    df_sem = df.groupby('lambda_tau')[cols_to_agg].sem().reset_index().sort_values('lambda_tau').fillna(0)
                    
                    data.append({
                        'label': label, 'color': color, 'ls': ls,
                        'mean': df_mean, 'sem': df_sem
                    })
                except Exception:
                    pass

            for d in data:
                if col_name not in d['mean'].columns: continue
                taus = d['mean']['lambda_tau'].values
                means = d['mean'][col_name].values
                error_band = d['sem'][col_name].values
                
                line, = ax.plot(taus, means, color=d['color'], marker='o', markersize=1.5, label=d['label'], linewidth=1.2, linestyle=d['ls'], alpha=0.8)
                ax.fill_between(taus, means - error_band, means + error_band, color=d['color'], alpha=0.1)

                if d['label'] not in legend_map:
                    legend_map[d['label']] = line
            
            # Subtitle (Embedding type) only on top row
            if row_idx == 0:
                ax.set_title(embedding_type.upper(), fontsize=10)
            
            # Y-axis labels only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(metric_title, fontsize=10)
                ax.tick_params(axis='y', left=True, labelleft=True)
            else:
                ax.tick_params(axis='y', left=True, labelleft=False)
                
            # X-axis labels only on bottom row
            if row_idx == 2:
                ax.tick_params(axis='x', bottom=True, labelbottom=True, which='both')
            else:
                ax.tick_params(axis='x', bottom=True, labelbottom=False, which='both')
                
            ax.set_xscale('log')
            ax.set_xlim(0.001, 10)
            ax.set_ylim(ylim)
            ax.grid(linestyle=':', alpha=0.8, which='both')
            ax.tick_params(labelsize=8)
            
    fig.supxlabel(r'Regularization Strength ($\tau$)', fontsize=10)
    
    # Legend settings
    handles = [legend_map[k] for k in legend_map]
    labels_legend = list(legend_map.keys())
    fig.legend(handles, labels_legend, loc='lower center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    
    output_base = f"sensitivity_all_{suite_name}_merged"
    output_name = f"{output_base}_appendix.pdf"
    plt.savefig(os.path.join(FIGURE_DIR, output_name), bbox_inches='tight')
    plt.close()
    print(f"Generated reorganized 3x3 plot: {output_name}")

# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for suite in ['max', 'n32']:
        plot_merged_sensitivity_all_embeddings(suite)

if __name__ == "__main__":
    main()
