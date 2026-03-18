#!/usr/bin/env python3
"""
Result Comparison Plotting for Amortized IRT.
Focuses on paper-ready bar charts:
1. Remediation Comparison (Pre-revision vs Post-revision SAE)
2. Embedding Comparison (SAE vs PCA vs RAW at N=max)
"""

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

RESULT_DIR = os.path.join(MODEL_DIR, "result")
BASELINE_PATH = os.path.join(RESULT_DIR, "baselines", "baseline_metrics.csv")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Config & Paths from pc (plotting colors) are now preferred

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_baseline_cache():
    if not os.path.exists(BASELINE_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(BASELINE_PATH, on_bad_lines='skip')
    except Exception:
        return pd.DataFrame()

    expected = ['seed', 'model_type', 'n_samples', 'pre_revision', 'j_percentage']
    expected += ['rmse_naive', 'rmse_rasch', 'rmse_2pl', 'rmse_mirt', 'auc_naive', 'auc_rasch', 'auc_2pl', 'auc_mirt']
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    df['model_type'] = df['model_type'].astype(str)
    df['pre_revision'] = df['pre_revision'].astype(str).str.strip().str.lower().replace('', 'none')
    df['n_samples'] = pd.to_numeric(df['n_samples'], errors='coerce')
    return df


def resolve_n_samples_from_df(df):
    if 'n_samples' not in df.columns:
        return None
    vals = pd.to_numeric(df['n_samples'], errors='coerce').dropna()
    if vals.empty:
        return None
    return int(vals.iloc[0])


def lookup_baseline_stats(baseline_df, model_type, n_samples, pre_revision, metric_col):
    if baseline_df.empty or n_samples is None or metric_col not in baseline_df.columns:
        return np.nan, np.nan

    pre_key = str(pre_revision).strip().lower() if pre_revision is not None else 'none'
    if not pre_key:
        pre_key = 'none'

    sub = baseline_df[
        (baseline_df['model_type'] == str(model_type)) &
        (baseline_df['n_samples'] == int(n_samples)) &
        (baseline_df['pre_revision'] == pre_key)
    ]
    vals = pd.to_numeric(sub[metric_col], errors='coerce').dropna()
    if vals.empty:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.sem() if len(vals) > 1 else 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# Data Loading & Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def load_aggregated_results(embedding_type, n_samples='max', model_type='beta'):
    fname = f"amortized_irt_{embedding_type}_{model_type}_n_{n_samples}.csv"
    path = os.path.join(RESULT_DIR, fname)
    if not os.path.exists(path): return None

    df = pd.read_csv(path, on_bad_lines='skip')
    metrics = ['rmse_naive', 'rmse_rasch', 'rmse_2pl', 'rmse_mirt', 'rmse_amortized', 
               'auc_naive', 'auc_rasch', 'auc_2pl', 'auc_mirt', 'auc_amortized']

    # Backward/forward compatibility: older or specialized CSVs may miss some baseline columns.
    for col in metrics:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Require core model metrics; keep rows even when some baseline metrics are absent.
    df = df.dropna(subset=['rmse_amortized', 'auc_amortized'])
    
    if df.empty: return None
    agg = df.groupby('n_samples')[metrics].agg(['mean', 'sem']).reset_index()
    agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg.columns]
    return agg.fillna(0)

def format_label(val, se):
    return f"{val:.3f}±{se:.3f}" if se > 1e-4 else f"{val:.3f}"

# ══════════════════════════════════════════════════════════════════════════════
# Plotting Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_remediation_summary():
    """6-bar plot comparing benchmarks before and after remediation (SAE only)."""
    configs = {
        'Pre_32': 'amortized_irt_sae_bernoulli_pre_32_n_1.csv',
        'Pre_max': 'amortized_irt_sae_beta_pre_max_n_max.csv',
        'Post_1': 'amortized_irt_sae_bernoulli_n_1.csv',
        'Post_max': 'amortized_irt_sae_beta_n_max.csv',
    }
    
    # Baseline from Post_max
    base_path = os.path.join(RESULT_DIR, configs['Post_max'])
    if not os.path.exists(base_path): return
    base_df = pd.read_csv(base_path, on_bad_lines='skip')
    baseline_df = load_baseline_cache()
    n_post_max = resolve_n_samples_from_df(base_df)

    def fallback_mean(df, col):
        return float(pd.to_numeric(df[col], errors='coerce').mean()) if col in df.columns else 0.0

    naive_auc_m, naive_auc_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'auc_naive')
    naive_rmse_m, naive_rmse_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'rmse_naive')
    rasch_auc_m, rasch_auc_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'auc_rasch')
    rasch_rmse_m, rasch_rmse_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'rmse_rasch')

    if np.isnan(naive_auc_m):
        naive_auc_m, naive_auc_s = fallback_mean(base_df, 'auc_naive'), 0.0
    if np.isnan(naive_rmse_m):
        naive_rmse_m, naive_rmse_s = fallback_mean(base_df, 'rmse_naive'), 0.0
    if np.isnan(rasch_auc_m):
        rasch_auc_m, rasch_auc_s = fallback_mean(base_df, 'auc_rasch'), 0.0
    if np.isnan(rasch_rmse_m):
        rasch_rmse_m, rasch_rmse_s = fallback_mean(base_df, 'rmse_rasch'), 0.0
    
    res = {
        'Naive': {'a_m': naive_auc_m, 'a_s': naive_auc_s, 'r_m': naive_rmse_m, 'r_s': naive_rmse_s},
        'Rasch': {'a_m': rasch_auc_m, 'a_s': rasch_auc_s, 'r_m': rasch_rmse_m, 'r_s': rasch_rmse_s},
    }
    
    # Try to load standalone 2PL
    standalone_2pl = os.path.join(RESULT_DIR, 'amortized_irt_rasch_2pl_beta_n_max.csv')
    if os.path.exists(standalone_2pl):
        df_2l = pd.read_csv(standalone_2pl)
        res['2PL'] = {'a_m': df_2l['auc_amortized'].mean(), 'a_s': df_2l['auc_amortized'].sem(), 'r_m': df_2l['rmse_amortized'].mean(), 'r_s': df_2l['rmse_amortized'].sem()}
    else:
        twopl_auc_m, twopl_auc_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'auc_2pl')
        twopl_rmse_m, twopl_rmse_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'rmse_2pl')
        if np.isnan(twopl_auc_m):
            twopl_auc_m, twopl_auc_s = fallback_mean(base_df, 'auc_2pl'), 0.0
        if np.isnan(twopl_rmse_m):
            twopl_rmse_m, twopl_rmse_s = fallback_mean(base_df, 'rmse_2pl'), 0.0
        res['2PL'] = {'a_m': twopl_auc_m, 'a_s': twopl_auc_s, 'r_m': twopl_rmse_m, 'r_s': twopl_rmse_s}

    # Try to load standalone MIRT
    standalone_mirt = os.path.join(RESULT_DIR, 'amortized_irt_nonamortised_mirt_beta_n_max.csv')
    if os.path.exists(standalone_mirt):
        df_mi = pd.read_csv(standalone_mirt)
        res['MIRT'] = {'a_m': df_mi['auc_amortized'].mean(), 'a_s': df_mi['auc_amortized'].sem(), 'r_m': df_mi['rmse_amortized'].mean(), 'r_s': df_mi['rmse_amortized'].sem()}
    else:
        mirt_auc_m, mirt_auc_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'auc_mirt')
        mirt_rmse_m, mirt_rmse_s = lookup_baseline_stats(baseline_df, 'beta', n_post_max, 'none', 'rmse_mirt')
        if np.isnan(mirt_auc_m):
            mirt_auc_m, mirt_auc_s = fallback_mean(base_df, 'auc_mirt'), 0.0
        if np.isnan(mirt_rmse_m):
            mirt_rmse_m, mirt_rmse_s = fallback_mean(base_df, 'rmse_mirt'), 0.0
        res['MIRT'] = {'a_m': mirt_auc_m, 'a_s': mirt_auc_s, 'r_m': mirt_rmse_m, 'r_s': mirt_rmse_s}
    for label, fname in configs.items():
        path = os.path.join(RESULT_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, on_bad_lines='skip')
            df['auc_amortized'] = pd.to_numeric(df['auc_amortized'], errors='coerce')
            df = df.dropna(subset=['auc_amortized'])
            if df.empty: continue
            
            # Use fillna(0) because .std() on a single row returns NaN
            a_mean = df['auc_amortized'].mean()
            a_se = df['auc_amortized'].std() / np.sqrt(len(df)) if len(df) > 1 else 0.0
            r_mean = df['rmse_amortized'].mean()
            r_se = df['rmse_amortized'].std() / np.sqrt(len(df)) if len(df) > 1 else 0.0
            
            res[label] = {
                'a_m': a_mean, 'a_s': a_se,
                'r_m': r_mean, 'r_s': r_se
            }
        else:
            print(f"Warning: {path} not found.")
            res[label] = {'a_m': 0.0, 'a_s': 0.0, 'r_m': 0.0, 'r_s': 0.0}

    order = ['Naive', 'Rasch', '2PL', 'MIRT', 'Pre_32', 'Pre_max', 'Post_1', 'Post_max']
    df_p = pd.DataFrame([{
        'Model': l, 'AUC': res[l]['a_m'], 'AUC_SE': res[l]['a_s'], 'RMSE': res[l]['r_m'], 'RMSE_SE': res[l]['r_s']
    } for l in order])

    plt.rcParams.update(get_bundle())
    
    # All bars use the same light blue color
    clrs = [pc.PRIMARY_LIGHT_BLUE] * len(df_p)

    for metric in ['AUC', 'RMSE']:
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        bars = ax.bar(df_p['Model'], df_p[metric], yerr=df_p[f'{metric}_SE'], capsize=2, color=clrs, 
                      edgecolor=pc.BAR_EDGE_COLOR, linewidth=pc.BAR_LINEWIDTH, alpha=pc.BAR_ALPHA)
        ax.set_ylabel(f"Predictive {metric}", fontweight='bold', fontsize=9)
        ax.set_ylim(0.4 if metric == 'AUC' else 0, 0.95 if metric == 'AUC' else 0.6)
        ax.set_xticklabels(df_p['Model'], rotation=15, fontsize=8)
        
        for bar, val, se in zip(bars, df_p[metric], df_p[f'{metric}_SE']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.005, format_label(val, se), ha='center', va='bottom', fontsize=5, fontweight='bold')
        
        plt.savefig(os.path.join(FIGURE_DIR, f"refined_{metric.lower()}_comparison.pdf"), bbox_inches='tight')
        plt.close()

def plot_embedding_comparison():
    """3-bar plot comparing Post_max performance across SAE, PCA, and RAW."""
    plot_data = []
    for emb in ['sae', 'pca', 'raw']:
        agg = load_aggregated_results(emb)
        if agg is not None:
            row = agg.iloc[-1]
            plot_data.append({'Label': emb.upper(), 'AUC': row['auc_amortized_mean'], 'AUC_SE': row['auc_amortized_sem'], 'RMSE': row['rmse_amortized_mean'], 'RMSE_SE': row['rmse_amortized_sem']})
    
    if not plot_data: return
    df_p = pd.DataFrame(plot_data)
    clrs = [pc.PRIMARY_LIGHT_BLUE] * len(df_p)

    plt.rcParams.update(get_bundle())
    for metric in ['AUC', 'RMSE']:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        bars = ax.bar(df_p['Label'], df_p[metric], yerr=df_p[f'{metric}_SE'], capsize=3, color=clrs, 
                      edgecolor=pc.BAR_EDGE_COLOR, linewidth=pc.BAR_LINEWIDTH, alpha=pc.BAR_ALPHA)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f"Post-max Performance ({metric})", fontsize=10)
        
        if metric == 'AUC':
            ax.set_ylim(0.6, 0.8)
        else: # RMSE
            ax.set_ylim(0.2, 0.4)
        
        for bar, val, se in zip(bars, df_p[metric], df_p[f'{metric}_SE']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.005, format_label(val, se), ha='center', va='bottom', fontsize=7, fontweight='bold')
            
        plt.savefig(os.path.join(FIGURE_DIR, f'post_max_embedding_comparison_{metric.lower()}.pdf'), bbox_inches='tight')
        plt.close()

def generate_comprehensive_table():
    """Generate a comprehensive Markdown table of all model results."""
    configs = [
        # (Label, Embedding, N, Model_Type, Pre_Revision_Suffix)
        # --- Post-Revision (Standard) ---
        ('Naive (Post-max Baseline)', 'sae', 'max', 'beta', ''),
        ('Rasch IRT (Post-max Baseline)', 'sae', 'max', 'beta', ''),
        ('Naive (Post-1 Baseline)', 'sae', '1', 'bernoulli', ''),
        ('Rasch IRT (Post-1 Baseline)', 'sae', '1', 'bernoulli', ''),
        ('SAE Post (N=1)', 'sae', '1', 'bernoulli', ''),
        ('SAE Post (N=max)', 'sae', 'max', 'beta', ''),
        ('PCA Post (N=1)', 'pca', '1', 'bernoulli', ''),
        ('PCA Post (N=max)', 'pca', 'max', 'beta', ''),
        ('RAW Post (N=1)', 'raw', '1', 'bernoulli', ''),
        ('RAW Post (N=max)', 'raw', 'max', 'beta', ''),

        # --- Pre-Revision (Baseline Phase) ---
        ('Naive-32 (Pre Baseline)', 'sae', '1', 'bernoulli', 'pre_32'),
        ('Rasch-32 (Pre Baseline)', 'sae', '1', 'bernoulli', 'pre_32'),
        ('SAE Pre-32 (N=1)', 'sae', '1', 'bernoulli', 'pre_32'),
        ('PCA Pre-32 (N=1)', 'pca', '1', 'bernoulli', 'pre_32'),
        ('RAW Pre-32 (N=1)', 'raw', '1', 'bernoulli', 'pre_32'),
        
        ('Naive Pre-max (Baseline)', 'sae', 'max', 'beta', 'pre_max'),
        ('Rasch Pre-max (Baseline)', 'sae', 'max', 'beta', 'pre_max'),
        ('SAE Pre-max (N=max)', 'sae', 'max', 'beta', 'pre_max'),
        ('PCA Pre-max (N=max)', 'pca', 'max', 'beta', 'pre_max'),
        ('RAW Pre-max (N=max)', 'raw', 'max', 'beta', 'pre_max'),

        # --- Ablations: No Tau (Post) ---
        ('SAE Post (N=1, No-TAU)', 'sae', '1_notau', 'bernoulli', ''),
        ('SAE Post (N=max, No-TAU)', 'sae', 'max_notau', 'beta', ''),
        ('PCA Post (N=max, No-TAU)', 'pca', 'max_notau', 'beta', ''),
        ('RAW Post (N=max, No-TAU)', 'raw', 'max_notau', 'beta', ''),

        # --- Ablations: No Embedding (ONES) ---
        ('ONES Post (N=1)', 'ones', '1', 'bernoulli', ''),
        ('ONES Post (N=max)', 'ones', 'max', 'beta', ''),
        ('ONES Pre-32 (N=1)', 'ones', '1', 'bernoulli', 'pre_32'),
        ('ONES Pre-max (N=max)', 'ones', 'max', 'beta', 'pre_max'),
    ]
    
    table_data = []
    baseline_df = load_baseline_cache()
    
    for label, emb, n, mtype, pre in configs:
        # Construct filename
        if pre:
            fname = f"amortized_irt_{emb}_{mtype}_{pre}_n_{n}.csv"
        else:
            fname = f"amortized_irt_{emb}_{mtype}_n_{n}.csv"
            
        path = os.path.join(RESULT_DIR, fname)
        if not os.path.exists(path):
            continue
            
        df = pd.read_csv(path, on_bad_lines='skip')
        for col in ['auc_amortized', 'auc_naive', 'auc_rasch', 'rmse_amortized', 'rmse_naive', 'rmse_rasch']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Baseline' in label:
            metric_prefix = 'auc_naive' if 'Naive' in label else 'auc_rasch'
            rmse_prefix = 'rmse_naive' if 'Naive' in label else 'rmse_rasch'
            n_samples = resolve_n_samples_from_df(df)
            pre_revision = pre if pre else 'none'
            
            auc_m, auc_se = lookup_baseline_stats(baseline_df, mtype, n_samples, pre_revision, metric_prefix)
            rmse_m, rmse_se = lookup_baseline_stats(baseline_df, mtype, n_samples, pre_revision, rmse_prefix)

            if np.isnan(auc_m):
                auc_m = df[metric_prefix].mean() if metric_prefix in df.columns else 0.0
                auc_se = df[metric_prefix].std() / np.sqrt(len(df)) if metric_prefix in df.columns and len(df) > 1 else 0.0
            if np.isnan(rmse_m):
                rmse_m = df[rmse_prefix].mean() if rmse_prefix in df.columns else 0.0
                rmse_se = df[rmse_prefix].std() / np.sqrt(len(df)) if rmse_prefix in df.columns and len(df) > 1 else 0.0
        else:
            auc_m = df['auc_amortized'].mean()
            auc_se = df['auc_amortized'].std() / np.sqrt(len(df)) if len(df) > 1 else 0.0
            rmse_m = df['rmse_amortized'].mean()
            rmse_se = df['rmse_amortized'].std() / np.sqrt(len(df)) if len(df) > 1 else 0.0
            
        table_data.append({
            'Model Configuration': label,
            'AUC': format_label(auc_m, auc_se),
            'RMSE': format_label(rmse_m, rmse_se)
        })
    
    if not table_data:
        print("No results found for table generation.")
        return
        
    res_df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(RESULT_DIR, "comprehensive_results.csv")
    res_df.to_csv(csv_path, index=False)
    
    # Print as Markdown Table
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print(res_df.to_markdown(index=False))
    print("="*80)
    
    # Save markdown version as well
    md_path = os.path.join(RESULT_DIR, "comprehensive_results.md")
    with open(md_path, 'w') as f:
        f.write("# Comprehensive Model Comparison\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n")

def main():
    print("Generating Result Comparison Plots...")
    plot_remediation_summary()
    plot_embedding_comparison()
    generate_comprehensive_table()
    
    # Explicitly print baselines for quick verification
    csv_path = os.path.join(RESULT_DIR, "comprehensive_results.csv")
    if os.path.exists(csv_path):
        res_df = pd.read_csv(csv_path)
        print("\n[SUMMARY] Key Post-Revision Baselines:")
        for cfg in ['Naive (Post-max Baseline)', 'Rasch IRT (Post-max Baseline)', 
                    'Naive (Post-1 Baseline)', 'Rasch IRT (Post-1 Baseline)']:
            row = res_df[res_df['Model Configuration'] == cfg]
            if not row.empty:
                print(f"  {cfg:30} | AUC {row['AUC'].values[0]:8} | RMSE {row['RMSE'].values[0]}")
            
    print(f"\nDone. Plots in {FIGURE_DIR}")

if __name__ == "__main__":
    main()
