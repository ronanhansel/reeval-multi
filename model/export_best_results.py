import pandas as pd
import os
import numpy as np

RESULT_DIR = '/Users/ronan/Developer/agent-eval/model/result'
OUTPUT_CSV = os.path.join(RESULT_DIR, 'comprehensive_results.csv')
OUTPUT_MD = os.path.join(RESULT_DIR, 'comprehensive_results.md')

def format_res(mean, sem):
    if sem == 0 or np.isnan(sem):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{sem:.3f}"

def get_best_results(filename, label_prefix):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found.")
        return []
    
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        # Coerce numeric
        cols_to_fix = ['lambda_tau', 'auc_amortized', 'rmse_amortized', 'auc_rasch', 'rmse_rasch', 'auc_2pl', 'rmse_2pl', 'auc_mirt', 'rmse_mirt', 'rmse_naive']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['auc_amortized', 'lambda_tau'])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

    # 1. Models (SAE/PCA/RAW)
    # Group by tau to find the best one based on mean AUC
    tau_stats = df.groupby('lambda_tau')['auc_amortized'].mean()
    best_tau = tau_stats.idxmax()
    
    best_df = df[df['lambda_tau'] == best_tau]
    
    auc_mean = best_df['auc_amortized'].mean()
    auc_se = best_df['auc_amortized'].sem()
    
    rmse_mean = best_df['rmse_amortized'].mean()
    rmse_se = best_df['rmse_amortized'].sem()
    
    results = [{
        'Model Configuration': label_prefix,
        'AUC': format_res(auc_mean, auc_se),
        'RMSE': format_res(rmse_mean, rmse_se)
    }]
    
    # 2. Extract Baselines (Skip if this IS a standalone baseline already)
    if any(b in label_prefix for b in ["Rasch 2PL", "MIRT", "ONES"]):
        return results

    unique_seeds = df.drop_duplicates(subset=['seed'])
    
    # Determine the specific baseline label to avoid collisions
    condition = label_prefix.split(' ', 1)[1] if ' ' in label_prefix else label_prefix
    
    # Rasch 1PL
    if 'auc_rasch' in df.columns:
        rasch_auc_mean = unique_seeds['auc_rasch'].mean()
        rasch_auc_se = unique_seeds['auc_rasch'].sem()
        rasch_rmse_mean = unique_seeds['rmse_rasch'].mean()
        rasch_rmse_se = unique_seeds['rmse_rasch'].sem()
        
        results.append({
            'Model Configuration': f"Rasch IRT ({condition} Baseline)",
            'AUC': format_res(rasch_auc_mean, rasch_auc_se),
            'RMSE': format_res(rasch_rmse_mean, rasch_rmse_se)
        })

    # 2PL (internal)
    if 'auc_2pl' in df.columns:
        twopl_auc_mean = unique_seeds['auc_2pl'].mean()
        twopl_auc_se = unique_seeds['auc_2pl'].sem()
        twopl_rmse_mean = unique_seeds['rmse_2pl'].mean()
        twopl_rmse_se = unique_seeds['rmse_2pl'].sem()
        
        results.append({
            'Model Configuration': f"2PL IRT ({condition} Baseline)",
            'AUC': format_res(twopl_auc_mean, twopl_auc_se),
            'RMSE': format_res(twopl_rmse_mean, twopl_rmse_se)
        })

    # MIRT (internal)
    if 'auc_mirt' in df.columns:
        mirt_auc_mean = unique_seeds['auc_mirt'].mean()
        mirt_auc_se = unique_seeds['auc_mirt'].sem()
        mirt_rmse_mean = unique_seeds['rmse_mirt'].mean()
        mirt_rmse_se = unique_seeds['rmse_mirt'].sem()
        
        results.append({
            'Model Configuration': f"Non-Amortized MIRT ({condition} Baseline)",
            'AUC': format_res(mirt_auc_mean, mirt_auc_se),
            'RMSE': format_res(mirt_rmse_mean, mirt_rmse_se)
        })

    # Naive
    if 'rmse_naive' in df.columns:
        rmse_naive_mean = unique_seeds['rmse_naive'].mean()
        rmse_naive_se = unique_seeds['rmse_naive'].sem()
        results.append({
            'Model Configuration': f"Naive ({condition} Baseline)",
            'AUC': "0.500",
            'RMSE': format_res(rmse_naive_mean, rmse_naive_se)
        })

    return results

# Mapping table
configs = [
    # Post Revision
    ('amortized_irt_sae_beta_n_max.csv', 'SAE Post (N=max)'),
    ('amortized_irt_pca_beta_n_max.csv', 'PCA Post (N=max)'),
    ('amortized_irt_raw_beta_n_max.csv', 'RAW Post (N=max)'),
    ('amortized_irt_sae_bernoulli_n_1.csv', 'SAE Post (N=1)'),
    ('amortized_irt_pca_bernoulli_n_1.csv', 'PCA Post (N=1)'),
    ('amortized_irt_raw_bernoulli_n_1.csv', 'RAW Post (N=1)'),
    
    # Standalone Baselines
    ('amortized_irt_rasch_2pl_bernoulli_n_32.csv', 'Rasch 2PL (N=32)'),
    ('amortized_irt_rasch_2pl_beta_n_max.csv', 'Rasch 2PL (N=max)'),
    ('amortized_irt_nonamortised_mirt_bernoulli_n_32.csv', 'Non-Amortized MIRT (N=32)'),
    ('amortized_irt_nonamortised_mirt_beta_n_max.csv', 'Non-Amortized MIRT (N=max)'),

    # Pre Revision
    ('amortized_irt_sae_beta_pre_max_n_max.csv', 'SAE Pre-max (N=max)'),
    ('amortized_irt_pca_beta_pre_max_n_max.csv', 'PCA Pre-max (N=max)'),
    ('amortized_irt_raw_beta_pre_max_n_max.csv', 'RAW Pre-max (N=max)'),
    ('amortized_irt_sae_bernoulli_pre_32_n_1.csv', 'SAE Pre-32 (N=1)'),
    ('amortized_irt_pca_bernoulli_pre_32_n_1.csv', 'PCA Pre-32 (N=1)'),
    ('amortized_irt_raw_bernoulli_pre_32_n_1.csv', 'RAW Pre-32 (N=1)'),
    
    # Ablations (Post Revision)
    ('amortized_irt_sae_beta_n_max_notau.csv', 'SAE Post (N=max, No-TAU)'),
    ('amortized_irt_pca_beta_n_max_notau.csv', 'PCA Post (N=max, No-TAU)'),
    ('amortized_irt_raw_beta_n_max_notau.csv', 'RAW Post (N=max, No-TAU)'),
    ('amortized_irt_ones_beta_n_max.csv', 'ONES Post (N=max)'),
    ('amortized_irt_ones_beta_n_max_notau.csv', 'ONES Post (N=max, No-TAU)'),
    
    # Ablations (Post Revision Bernoulli N=1)
    ('amortized_irt_sae_bernoulli_n_1_notau.csv', 'SAE Post (N=1, No-TAU)'),
    ('amortized_irt_pca_bernoulli_n_1_notau.csv', 'PCA Post (N=1, No-TAU)'),
    ('amortized_irt_raw_bernoulli_n_1_notau.csv', 'RAW Post (N=1, No-TAU)'),
    ('amortized_irt_ones_bernoulli_n_1.csv', 'ONES Post (N=1)'),
    ('amortized_irt_ones_bernoulli_n_1_notau.csv', 'ONES Post (N=1, No-TAU)'),

    # Ablations (Pre Revision N=max)
    ('amortized_irt_sae_beta_pre_max_n_max_notau.csv', 'SAE Pre-max (N=max, No-TAU)'),
    ('amortized_irt_pca_beta_pre_max_n_max_notau.csv', 'PCA Pre-max (N=max, No-TAU)'),
    ('amortized_irt_raw_beta_pre_max_n_max_notau.csv', 'RAW Pre-max (N=max, No-TAU)'),
    ('amortized_irt_ones_beta_pre_max_n_max.csv', 'ONES Pre-max (N=max)'),
    ('amortized_irt_ones_beta_pre_max_n_max_notau.csv', 'ONES Pre-max (N=max, No-TAU)'),

    # Ablations (Pre Revision Pre-32 N=1)
    ('amortized_irt_sae_bernoulli_pre_32_n_1_notau.csv', 'SAE Pre-32 (N=1, No-TAU)'),
    ('amortized_irt_pca_bernoulli_pre_32_n_1_notau.csv', 'PCA Pre-32 (N=1, No-TAU)'),
    ('amortized_irt_raw_bernoulli_pre_32_n_1_notau.csv', 'RAW Pre-32 (N=1, No-TAU)'),
    ('amortized_irt_ones_bernoulli_pre_32_n_1.csv', 'ONES Pre-32 (N=1)'),
    ('amortized_irt_ones_bernoulli_pre_32_n_1_notau.csv', 'ONES Pre-32 (N=1, No-TAU)'),
    
    # Standalone Baselines
    ('amortized_irt_rasch_2pl_bernoulli_n_32.csv', '2PL Standalone (N=32)'),
    ('amortized_irt_rasch_2pl_beta_n_max.csv', '2PL Standalone (N=max)'),
    ('amortized_irt_nonamortised_mirt_bernoulli_n_32.csv', 'MIRT Standalone (N=32)'),
    ('amortized_irt_nonamortised_mirt_beta_n_max.csv', 'MIRT Standalone (N=max)'),
]

all_results = []
for filename, label in configs:
    all_results.extend(get_best_results(filename, label))

# Convert to DF and drop duplicates (baselines will no longer collide incorrectly)
final_df = pd.DataFrame(all_results).drop_duplicates(subset=['Model Configuration'])

# Better sorting for the CSV/MD
def sort_key(label):
    if 'Naive' in label: return 0
    if 'Rasch' in label: return 1
    if '2PL' in label: return 2
    if 'MIRT' in label: return 3
    if 'SAE' in label: return 4
    if 'PCA' in label: return 5
    if 'RAW' in label: return 6
    if 'ONES' in label: return 7
    return 8

# Priority for Pre/Post and Max/8
def cond_key(label):
    score = 0
    if "Pre-max" in label: score += 10
    if "Pre-32" in label: score += 20
    if "Post (N=1)" in label: score += 30
    if "Post (N=max)" in label: score += 40
    return score

final_df['sort_order'] = final_df['Model Configuration'].apply(sort_key)
final_df['cond_order'] = final_df['Model Configuration'].apply(cond_key)
final_df = final_df.sort_values(['cond_order', 'sort_order']).drop(columns=['sort_order', 'cond_order'])

# Save
final_df.to_csv(OUTPUT_CSV, index=False)
with open(OUTPUT_MD, 'w') as f:
    f.write(final_df.to_markdown(index=False))

print(f"Exported all results to {OUTPUT_CSV}")
