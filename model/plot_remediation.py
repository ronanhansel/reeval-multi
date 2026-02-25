import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════
RESULT_DIR = "model/result"
FIGURE_DIR = "paper/figures"
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
    Using the new consolidated CSV structure.
    """
    
    # 1. Naive & Rasch (from pre_max)
    # We'll use the amortized_irt_sae_beta_pre_max_n_1.csv for benchmarks
    file_map = {
        'Pre_8': "amortized_irt_sae_bernoulli_pre_8_n_1.csv",
        'Pre_max': "amortized_irt_sae_beta_pre_max_n_1.csv",
        'Post_1': "amortized_irt_sae_bernoulli_n_1.csv",
        'Post_max': "amortized_irt_sae_beta_n_max.csv"
    }
    
    # Get Naive/Rasch from Pre_max file
    pm_df = pd.read_csv(os.path.join(RESULT_DIR, file_map['Pre_max']))
    
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
    plt.rcParams.update(get_bundle())
    
    # AUC Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(df['Model'], df['AUC_Mean'], yerr=df['AUC_SE'], 
                  capsize=2, color=MAIN_BLUE, error_kw={'elinewidth': 0.6, 'capthick': 0.6})
    
    ax.set_ylabel("Predictive AUC")
    ax.set_ylim(0.4, 0.9)
    ax.set_title("Model Predictability (AUC)")
    ax.set_xticks(range(len(df['Model'])))
    ax.set_xticklabels(df['Model'], rotation=15)
    
    for bar, val, se in zip(bars, df['AUC_Mean'], df['AUC_SE']):
        height = bar.get_height()
        label = format_label(val, se)
        ax.text(bar.get_x() + bar.get_width() / 2, height + se + 0.01, label, 
                ha='center', va='bottom', fontsize=5, fontweight='bold')

    plt.savefig(os.path.join(FIGURE_DIR, "refined_auc_comparison.pdf"), bbox_inches='tight')
    plt.close()

    # RMSE Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(df['Model'], df['RMSE_Mean'], yerr=df['RMSE_SE'], 
                  capsize=2, color=MAIN_BLUE, error_kw={'elinewidth': 0.6, 'capthick': 0.6})
    
    ax.set_ylabel("Predictive RMSE")
    ax.set_ylim(0, 0.6)
    ax.set_title("Model Predictability (RMSE)")
    ax.set_xticks(range(len(df['Model'])))
    ax.set_xticklabels(df['Model'], rotation=15)
    
    for bar, val, se in zip(bars, df['RMSE_Mean'], df['RMSE_SE']):
        height = bar.get_height()
        label = format_label(val, se)
        ax.text(bar.get_x() + bar.get_width() / 2, height + se + 0.01, label, 
                ha='center', va='bottom', fontsize=5, fontweight='bold')

    plt.savefig(os.path.join(FIGURE_DIR, "refined_rmse_comparison.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = collect_results()
    print(df)
    plot_refined_results(df)
    print(f"Refined plots generated in {FIGURE_DIR}")
