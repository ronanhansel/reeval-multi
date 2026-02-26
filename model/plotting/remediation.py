#!/usr/bin/env python3
import os
import sys
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
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)

# Main Blue Color (from Pre_max)
MAIN_BLUE = '#1f77b4'

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif") # usetex=False for safer local run

# ══════════════════════════════════════════════════════════════════════════════
# Data Collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_results():
    """
    Collect metrics for Naive, Rasch, Pre_8, Pre_max, Post_1, and Post_max.
    """
    file_map = {
        'Pre_8': "amortized_irt_sae_bernoulli_pre_8_n_1.csv",
        'Pre_max': "amortized_irt_sae_beta_pre_max_n_1.csv",
        'Post_1': "amortized_irt_sae_bernoulli_n_1.csv",
        'Post_max': "amortized_irt_sae_beta_n_max.csv"
    }
    
    # Check if Pre_max exists to get Naive/Rasch benchmarks
    pm_path = os.path.join(RESULT_DIR, file_map['Pre_max'])
    if not os.path.exists(pm_path):
        print(f"Warning: {pm_path} not found. Cannot collect full results.")
        return pd.DataFrame()

    pm_df = pd.read_csv(pm_path)
    
    res = {
        'Naive': {'auc': pm_df['auc_naive'].tolist(), 'rmse': pm_df['rmse_naive'].tolist()},
        'Rasch': {'auc': pm_df['auc_rasch'].tolist(), 'rmse': pm_df['rmse_rasch'].tolist()},
    }
    
    # Load others
    for col, filename in file_map.items():
        path = os.path.join(RESULT_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            res[col] = {
                'auc': df['auc_amortized'].tolist(),
                'rmse': df['rmse_amortized'].tolist()
            }
        else:
            print(f"Warning: {path} not found.")
            res[col] = {'auc': [0.0], 'rmse': [0.0]}
            
    # Aggregate stats
    final_data = []
    columns = ['Naive', 'Rasch', 'Pre_8', 'Pre_max', 'Post_1', 'Post_max']
    for col in columns:
        auc_vals = res[col]['auc']
        rmse_vals = res[col]['rmse']
        
        final_data.append({
            'Model': col,
            'AUC_Mean': np.mean(auc_vals),
            'AUC_SE': np.std(auc_vals) / np.sqrt(len(auc_vals)) if len(auc_vals) > 1 else 0.0,
            'RMSE_Mean': np.mean(rmse_vals),
            'RMSE_SE': np.std(rmse_vals) / np.sqrt(len(rmse_vals)) if len(rmse_vals) > 1 else 0.0
        })
        
    return pd.DataFrame(final_data)

# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def format_label(val, se):
    if se > 1e-6: # Threshold for showing SE
        return f"{val:.3f} ± {se:.3f}"
    else:
        return f"{val:.3f}"

def plot_refined_results(df):
    if df.empty:
        return

    plt.rcParams.update(get_bundle())
    
    # AUC Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(df['Model'], df['AUC_Mean'], yerr=df['AUC_SE'], 
                  capsize=2, color=MAIN_BLUE, error_kw={'elinewidth': 0.6, 'capthick': 0.6})
    
    ax.set_ylabel("Predictive AUC")
    ax.set_ylim(0.4, 0.9)
    ax.set_xticks(range(len(df['Model'])))
    ax.set_xticklabels(df['Model'], rotation=15)
    
    for bar, val, se in zip(bars, df['AUC_Mean'], df['AUC_SE']):
        height = bar.get_height()
        label = format_label(val, se)
        ax.text(bar.get_x() + bar.get_width() / 2, height + se + 0.01, label, 
                ha='center', va='bottom', fontsize=5)

    plt.savefig(os.path.join(FIGURE_DIR, "refined_auc_comparison.pdf"), bbox_inches='tight')
    plt.close()

    # RMSE Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(df['Model'], df['RMSE_Mean'], yerr=df['RMSE_SE'], 
                  capsize=2, color=MAIN_BLUE, error_kw={'elinewidth': 0.6, 'capthick': 0.6})
    
    ax.set_ylabel("Predictive RMSE")
    ax.set_ylim(0, 0.6)
    ax.set_xticks(range(len(df['Model'])))
    ax.set_xticklabels(df['Model'], rotation=15)
    
    for bar, val, se in zip(bars, df['RMSE_Mean'], df['RMSE_SE']):
        height = bar.get_height()
        label = format_label(val, se)
        ax.text(bar.get_x() + bar.get_width() / 2, height + se + 0.01, label, 
                ha='center', va='bottom', fontsize=5)

    plt.savefig(os.path.join(FIGURE_DIR, "refined_rmse_comparison.pdf"), bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("GENERATING REMEDIATION PLOTS")
    print("=" * 60)
    
    df = collect_results()
    if not df.empty:
        print(df)
        plot_refined_results(df)
        print(f"Refined plots generated in {FIGURE_DIR}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
