#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import bundles
import torch
import ast
import pickle
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from matplotlib.patches import FancyBboxPatch
from model.baseline_cache import load_baseline_store
from model.result_paths import ensure_main_result_dir, main_result_dir


# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = str(main_result_dir())
BASELINE_PATH = str(Path(RESULT_DIR) / "baselines" / "baseline_metrics.csv")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures", "interpretability")
EMB_DIR = os.path.join(MODEL_DIR, "processed_embeddings")
PREFERRED_BASELINE_EMBEDDING = 'raw'

os.makedirs(FIGURE_DIR, exist_ok=True)

# Add repo root to path for imports if needed
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from model.plotting import colors as pc

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_baseline_cache():
    df = load_baseline_store(BASELINE_PATH)

    if 'n_samples' in df.columns:
        df['n_samples'] = pd.to_numeric(df['n_samples'], errors='coerce')
    if 'pre_revision' in df.columns:
        df['pre_revision'] = df['pre_revision'].astype(str).str.strip().str.lower().replace('', 'none')
    if 'model_type' in df.columns:
        df['model_type'] = df['model_type'].astype(str)
    if 'baseline_embedding_type' in df.columns:
        df['baseline_embedding_type'] = (
            df['baseline_embedding_type'].astype(str).str.strip().str.lower().replace({'': 'pca', 'nan': 'pca'})
        )
    else:
        df['baseline_embedding_type'] = 'pca'
    return df


def lookup_baseline_auc(baseline_df, model_type, n_samples, pre_revision, metric_col):
    if baseline_df.empty or metric_col not in baseline_df.columns or n_samples is None:
        return None, 0.0
    sub = baseline_df[
        (baseline_df['model_type'] == str(model_type)) &
        (baseline_df['n_samples'] == int(n_samples)) &
        (baseline_df['pre_revision'] == str(pre_revision).strip().lower())
    ]
    if 'baseline_embedding_type' in sub.columns:
        available = set(sub['baseline_embedding_type'].dropna().astype(str))
        target = PREFERRED_BASELINE_EMBEDDING if PREFERRED_BASELINE_EMBEDDING in available else ('pca' if 'pca' in available else None)
        if target is not None:
            sub = sub[sub['baseline_embedding_type'] == target]
    vals = pd.to_numeric(sub[metric_col], errors='coerce').dropna()
    if vals.empty:
        return None, 0.0
    return float(vals.mean()), float(vals.sem() if len(vals) > 1 else 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# Optimal SOTA Tau mappings (when filtering sweep CSVs for robustness plots)
# ══════════════════════════════════════════════════════════════════════════════

SOTA_TAUS = {
    "amortized_irt_sae_bernoulli_pre_32_n_1.csv": 0.0159,
    "amortized_irt_sae_beta_pre_max_n_max.csv": 0.16,
    "amortized_irt_sae_beta_n_max.csv": 0.0535,
    "amortized_irt_pca_bernoulli_n_1.csv": 0.0155,
    "amortized_irt_pca_bernoulli_pre_32_n_1.csv": 0.0155,
    "amortized_irt_pca_beta_n_max.csv": 0.054,
    "amortized_irt_pca_beta_pre_max_n_max.csv": 0.054,
    "amortized_irt_raw_beta_n_max.csv": 0.029,
    "amortized_irt_raw_bernoulli_n_1.csv": 0.0151,
}

def load_filtered_csv(filename):
    """Loads CSV and filters it to only the SOTA tau value if it contains a tau sweep."""
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        return None
        
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
    except Exception:
        return None
    
    if 'lambda_tau' in df.columns and filename in SOTA_TAUS:
        df['lambda_tau'] = pd.to_numeric(df['lambda_tau'], errors='coerce')
        df = df.dropna(subset=['lambda_tau'])
        target_tau = SOTA_TAUS[filename]
        # Allow small floating point tolerance
        df = df[np.isclose(df['lambda_tau'], target_tau, atol=1e-4)]
        
    if 'active_dims' in df.columns:
        df['active_dims'] = pd.to_numeric(df['active_dims'], errors='coerce')
    if 'auc_amortized' in df.columns:
        df['auc_amortized'] = pd.to_numeric(df['auc_amortized'], errors='coerce')
        
    df = df.dropna(subset=['active_dims', 'auc_amortized'])
        
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Dimensionality Stability (K-Consistency)
# ══════════════════════════════════════════════════════════════════════════════

def plot_stability_comparison():
    plt.rcParams.update(get_bundle())
    
    file_map = {
        'Pre-Rev (Best)': "amortized_irt_sae_beta_pre_max_n_max.csv",
        'Post-Rev (N=max)': "amortized_irt_sae_beta_n_max.csv"
    }
    
    all_rows = []
    for label, filename in file_map.items():
        df = load_filtered_csv(filename)
        if df is not None:
            for k in df['active_dims']:
                all_rows.append({'Model': label, 'K': k})
    
    if not all_rows:
        return

    df_plot = pd.DataFrame(all_rows)
    
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    # Use a boxplot for distribution and swarmplot for individual seeds
    sns.boxplot(data=df_plot, x='Model', y='K', color='#f0f0f0', width=0.5, ax=ax, showfliers=False)
    sns.stripplot(data=df_plot, x='Model', y='K', palette='viridis', alpha=0.7, size=6, ax=ax, jitter=True)
    
    # Calculate seeds for title
    seed_counts = df_plot.groupby('Model').size().unique()
    seed_str = f"{seed_counts[0]}" if len(seed_counts) == 1 else f"{seed_counts.min()}-{seed_counts.max()}"
    
    ax.set_ylabel("Active Dimensions ($K$)")
    ax.set_ylim(-1, 31)
    
    plt.savefig(os.path.join(FIGURE_DIR, "k_stability_distribution.pdf"), bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Loading Cleanliness & Semantic Alignment
# ══════════════════════════════════════════════════════════════════════════════

def load_data_and_weights(weight_path, embedding_type='sae', pre_revision='none'):
    resmat_dir = os.path.join(REPO_ROOT, 'item-editor', 'eval_response_matrix')
    
    if pre_revision != 'none':
        pre_rev_dir = os.path.join(resmat_dir, 'pre-revision')
        b_names = ['colbench_backend_programming', 'corebench_hard', 'scicode', 'scienceagentbench']
        combined_dfs = []
        for b in b_names:
            possible_files = ['raw_score.csv', 'benchmark.csv', 'success_rate.csv', 'written_score.csv']
            df = None
            for f in possible_files:
                p = os.path.join(pre_rev_dir, b, f)
                if os.path.exists(p):
                    df = pd.read_csv(p, index_col=0)
                    break
            
            if df is not None:
                df.columns = [f"{b}.{c}" if not str(c).startswith(b) and not str(c).startswith(b.replace('_hard','')) else c for c in df.columns]
                combined_dfs.append(df)
        if combined_dfs:
            final_df = pd.concat(combined_dfs, axis=1, join='outer')
            oracle_df = final_df.dropna(axis=1, how='all')
        else:
            print("Warning: No pre-revision data found.")
            return None, None
    else:
        post_rev_dir = os.path.join(resmat_dir, 'post-revision')
        b_names = ['colbench_backend_programming', 'corebench_hard', 'scicode', 'scienceagentbench']
        combined_dfs = []
        for b in b_names:
            b_resmat_dir = os.path.join(post_rev_dir, b, 'resmat')
            if os.path.exists(b_resmat_dir):
                files = sorted([f for f in os.listdir(b_resmat_dir) if f.startswith('resmat')])
                if files:
                    df = pd.read_csv(os.path.join(b_resmat_dir, files[0]), index_col=0)
                    
                    # First, keep only columns that belong to this benchmark
                    valid_cols = [c for c in df.columns if str(c).startswith(b)]
                    if valid_cols:
                        df = df[valid_cols]
                        # Now rename properly (some might be pure ints, some might be "benchmark.id")
                        df.columns = [f"{b}.{c}" if not str(c).startswith(b) and not str(c).startswith(b.replace('_hard','')) else c for c in df.columns]
                        combined_dfs.append(df)
        if combined_dfs:
            oracle_df = pd.concat(combined_dfs, axis=1, join='outer')
        else:
            print("Warning: No post-revision data found.")
            return None, None

    # Load embeddings
    emb_file = os.path.join(EMB_DIR, f'embeddings_{embedding_type}_48.pkl')
    if not os.path.exists(emb_file):
        print(f"Embedding file not found: {emb_file}")
        return None, None
        
    emb_df = pd.read_pickle(emb_file)
    id_col = 'task_id' if 'task_id' in emb_df.columns else 'benchmark.task_id'
    raw_embs_map = {str(r[id_col]): r['embedding'] for _, r in emb_df.iterrows()}

    # Align embeddings
    task_ids = oracle_df.columns.tolist()
    embeddings = []
    for tid in task_ids:
        emb = raw_embs_map.get(str(tid))
        if emb is None and tid.startswith('colbench.'):
            emb = raw_embs_map.get(f'colbench_backend_programming.{tid.split(".")[-1]}')
        if emb is None: emb = np.zeros(48)
        embeddings.append(np.array(emb, dtype=np.float32))
    
    x_j = torch.tensor(np.stack(embeddings))
    x_j = x_j / (torch.norm(x_j, dim=1, keepdim=True) + 1e-8)
    
    # Load weights
    if not os.path.exists(weight_path):
        print(f"Weights file not found: {weight_path}")
        return None, None
        
    state = torch.load(weight_path)
    W = state['W']
    tau = torch.relu(state['tau_raw'])
    
    # Calc loadings: A = tau * (x_j @ W_norm.T)
    W_norm = torch.nn.functional.normalize(W, dim=1)
    loadings = (x_j @ W_norm.T) * tau.unsqueeze(0)
    
    return loadings.numpy(), task_ids

def get_item_inputs(tids):
    input_path = os.path.join(REPO_ROOT, "item-editor/eval_response_matrix/all_benchmarks_embeddings_4096_8B.pkl")
    if not os.path.exists(input_path): return {}
    df = pd.read_pickle(input_path)
    # Create lookup benchmark.task_id -> text_input
    lookup = {}
    for _, row in df.iterrows():
        key = str(row['benchmark.task_id'])
        lookup[key] = row['text_input']
    return lookup

def plot_loading_heatmap():
    plt.rcParams.update(get_bundle())
    
    configs = [
        ('Pre-Revision', os.path.join(RESULT_DIR, "amortized_irt_sae_beta_pre_max_n_max_seed_42_weights_final.pkl"), 'max'),
        ('Post-Revision', os.path.join(RESULT_DIR, "amortized_irt_sae_beta_n_max_seed_42_weights_final.pkl"), 'none')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 1.5), sharey=True, constrained_layout=False)
    
    any_plotted = False
    
    bench_map = {
        'colbench_backend_programming': 'ColBench',
        'corebench_hard': 'CoreBench',
        'scicode': 'SciCode',
        'scienceagentbench': 'SAB'
    }
    
    # Enforce left-to-right sorting independently of alphabet
    ordered_benchmarks = ['colbench_backend_programming', 'scienceagentbench', 'corebench_hard', 'scicode']
    b_order = {b: i for i, b in enumerate(ordered_benchmarks)}
    
    # 1. Load all data and find global max for symmetric colorbar
    loaded_data = []
    common_tids = None
    max_val = 0.0
    
    for i, (label, w_path, pre_rev) in enumerate(configs):
        if not os.path.exists(w_path): 
            print(f"Skipping {label} loading heatmap: {os.path.basename(w_path)} not found.")
            loaded_data.append(None)
            continue
        A, tids = load_data_and_weights(w_path, pre_revision=pre_rev)
        if A is None: 
            loaded_data.append(None)
            continue
            
        loaded_data.append((A, tids))
        max_val = max(max_val, np.abs(A).max())
        
        if common_tids is None:
            common_tids = set(tids)
        else:
            common_tids = common_tids.intersection(set(tids))
            
    if not common_tids:
        print("Skipping loading cleanliness comparison: No common items found or weights missing.")
        plt.close()
        return
        
    for i, (label, w_path, pre_rev) in enumerate(configs):
        data = loaded_data[i]
        if data is None: continue
        
        A, tids = data
        any_plotted = True
        
        # Filter for active dims only
        active = np.where(np.abs(A).max(axis=0) > 0.005)[0]
        if len(active) == 0: active = np.arange(A.shape[1])
        A_sub = A[:, active]
        
        # Filter to only the intersection
        tids_list = list(tids)
        intersect_mask = [tids_list.index(t) for t in tids_list if t in common_tids]
        A_intersect = A_sub[intersect_mask]
        intersect_tids = [tids_list[idx] for idx in intersect_mask]
        
        # Sort items explicitly by predefined benchmark order
        benchmarks = [t.split('.')[0] for t in intersect_tids]
        sort_idx = np.argsort([b_order.get(b, 99) for b in benchmarks])
        A_sorted = A_intersect[sort_idx]
        sorted_benchmarks = np.array(benchmarks)[sort_idx]
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        
        v_limit = 1.0 if 'Pre' in label else 0.05
        ticks = [-0.8, -0.4, 0.0, 0.4, 0.8] if 'Pre' in label else [-0.04, -0.02, 0.00, 0.02, 0.04]
        cbar_fmt = '%.1f' if 'Pre' in label else '%.2f'
        
        hm = sns.heatmap(A_sorted.T, ax=axes[i], cmap='RdBu_r', center=0, 
                        vmin=-v_limit, vmax=v_limit,
                        cbar=True, cbar_ax=cax,
                        cbar_kws={'ticks': ticks, 'format': cbar_fmt})
        
        # Shrink tick marks
        axes[i].tick_params(axis='both', which='both', length=2, width=0.5)
        
        # Shrink colorbar tick marks
        cax.tick_params(axis='both', which='both', length=2, width=0.5)
        
        # Remove redundant labels for the second plot
        if i == 1:
            axes[i].tick_params(left=False, labelleft=False)
            
        axes[i].set_xlabel(label, fontsize=7)
        if i == 0:
            axes[i].set_ylabel("Active Latent Dims ($K$)", fontsize=7)
            
        # Add vertical black lines and custom x-ticks at cluster centers
        unique_b = []
        counts = []
        for b in sorted_benchmarks:
            if not unique_b or unique_b[-1] != b:
                unique_b.append(b)
                counts.append(1)
            else:
                counts[-1] += 1
                
        boundaries = [0]
        centers = []
        for count in counts:
            boundaries.append(boundaries[-1] + count)
            centers.append(boundaries[-2] + count / 2.0)
            
        for b_idx in boundaries[1:-1]:
            axes[i].axvline(x=b_idx, color='black', linewidth=0.3, linestyle='--')
            
        axes[i].set_xticks([])
    
    if any_plotted:
        fig.subplots_adjust(wspace=0.15)
        out_path = os.path.join(FIGURE_DIR, "loading_cleanliness_comparison.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Loading cleanliness comparison generated: {out_path}")
    plt.close()

def plot_semantic_alignment():
    desc_path = os.path.join(EMB_DIR, "feature_descriptions_sae.pkl")
    
    configs = [
        ('Pre_max', os.path.join(RESULT_DIR, "amortized_irt_sae_beta_pre_max_n_max_seed_42_weights_final.pkl"), 'max'),
        ('Post_max', os.path.join(RESULT_DIR, "amortized_irt_sae_beta_n_max_seed_42_weights_final.pkl"), 'none')
    ]
    
    if not os.path.exists(desc_path):
        print(f"WARNING: {desc_path} not found. Semantic Alignment report will have 'N/A' for topics.")
        descriptions = pd.DataFrame(columns=['neuron_idx', 'top_example_1'])
    else:
        with open(desc_path, "rb") as f:
            descriptions = pickle.load(f)
        if isinstance(descriptions, list):
            descriptions = pd.DataFrame(descriptions)

    for label, w_path, pre_rev in configs:
        if not os.path.exists(w_path): 
            print(f"Skipping {label} semantic alignment: {os.path.basename(w_path)} not found.")
            continue
        A, tids = load_data_and_weights(w_path, pre_revision=pre_rev)
        if A is None: continue
        
        weights = torch.load(w_path)
        tau = torch.relu(weights['tau_raw']).cpu().numpy()
        
        active = np.where(np.abs(A).max(axis=0) > 1e-6)[0]
        
        report_lines = []
        report_lines.append(f"\n### Semantic Alignment & Clarity: {label} (K={len(active)})")
        report_lines.append("| Dim | Tau Strength | Top 10 Loader Items + Context | Primary Benchmark | Purity |")
        report_lines.append("|---|---|---|---|---|")
        
        benchmarks = np.array([t.split('.')[0] for t in tids])
        
        all_purities = []
        all_entropies = []
        
        input_lookup = get_item_inputs(tids)
        
        for idx in active:
            # Loaders
            loadings = np.abs(A[:, idx])
            top_10_indices = np.argsort(loadings)[-10:][::-1]
            
            loader_details = []
            for tid_idx in top_10_indices:
                tid = tids[tid_idx]
                val = loadings[tid_idx]
                # Only include if loading is significant
                if val < 1e-6: continue 
                desc = input_lookup.get(tid, "")[:3500].replace('\n', ' ').strip()
                loader_details.append(f"**{tid}**: {desc}...")
            
            loaders_str = "<br>".join(loader_details) if loader_details else "None"
            
            # Purity/Stats
            top_20_indices = np.argsort(loadings)[-20:]
            top_20_benchmarks = benchmarks[top_20_indices]
            unique, counts = np.unique(top_20_benchmarks, return_counts=True)
            primary = unique[np.argmax(counts)] if len(unique) > 0 else "N/A"
            purity = np.max(counts) / 20 if len(counts) > 0 else 0
            all_purities.append(purity)
            
            # Entropy
            bench_loadings = []
            for b in np.unique(benchmarks):
                curr_b_loadings = loadings[benchmarks == b]
                bench_sum = curr_b_loadings.sum() if len(curr_b_loadings) > 0 else 0
                bench_loadings.append(bench_sum)
            bench_loadings = np.array(bench_loadings) / (loadings.sum() + 1e-9)
            ent = entropy(bench_loadings) if loadings.sum() > 1e-6 else 0
            all_entropies.append(ent)
            
            report_lines.append(f"| {idx} | {tau[idx]:.3f} | {loaders_str} | {primary} | {purity:.2f} |")
            
        report_lines.append(f"\n**{label} Aggregate Metrics:**")
        report_lines.append(f"- Mean Benchmark Purity: {np.mean(all_purities):.4f}")
        report_lines.append(f"- Mean Loading Entropy (lower is cleaner): {np.mean(all_entropies):.4f}")
        report_lines.append("-" * 40)
        
        report_content = "\n".join(report_lines)
        print(report_content)
        
        report_path = os.path.join(RESULT_DIR, f"semantic_alignment_{label}.md")
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"Detailed report saved to {report_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Tau Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def plot_tau_sensitivity():
    plt.rcParams.update(get_bundle())
    
    # Legend order: Pre Revision (32), Bernoulli, Beta
    # Dark on top, Light on bottom
    comparison_sets = [
        ('max_merged', [
            ('Pre Revision', 'amortized_irt_sae_bernoulli_pre_max_n_1.csv', '-', 'salmon'),
            ('Bernoulli', 'amortized_irt_sae_bernoulli_n_1.csv', '-', 'skyblue'),
            ('Beta', 'amortized_irt_sae_beta_n_max.csv', '-', 'tab:blue'),
        ]),
        ('n32_merged', [
            ('Pre Revision', 'amortized_irt_sae_bernoulli_pre_32_n_1.csv', '-', 'salmon'),
            ('Bernoulli', 'amortized_irt_sae_bernoulli_n_1.csv', '-', 'skyblue'),
            ('Beta', 'amortized_irt_sae_beta_n_max.csv', '-', 'tab:blue'),
        ])
    ]

    # Only paper linear scale now (appendix handled in appendix.py)
    scale_name, x_start, x_data_limit, x_axis_limit, x_scale = ('linear', 0.0, 0.1, 0.102, 'linear')
    
    for suffix, configs in comparison_sets:
        data = []
        for label, filename, ls, color in configs:
            path_data = os.path.join(RESULT_DIR, filename)
            if not os.path.exists(path_data): continue
            
            try:
                df = pd.read_csv(path_data, on_bad_lines='skip')
                df.columns = df.columns.str.strip()
                
                required_cols = ['lambda_tau', 'auc_amortized', 'rmse_amortized', 'active_dims']
                for col in required_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['lambda_tau', 'auc_amortized', 'active_dims'])
                df = df[df['lambda_tau'] <= x_data_limit]
                if df.empty: continue

                cols_to_agg = ['auc_amortized', 'active_dims']
                if 'rmse_amortized' in df.columns:
                    cols_to_agg.append('rmse_amortized')

                df_mean = df.groupby('lambda_tau')[cols_to_agg].mean().reset_index()
                df_sem = df.groupby('lambda_tau')[cols_to_agg].sem().reset_index().fillna(0)
                
                data.append({
                    'label': label, 'color': color, 'ls': ls,
                    'mean': df_mean.sort_values('lambda_tau'),
                    'sem': df_sem.sort_values('lambda_tau')
                })
            except Exception: continue

        if not data: continue

        # Determine layout: Paper (linear) is 2-panel
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), constrained_layout=True)
        metrics = [
            ('auc_amortized', 'AUC', (0.58, 0.82)),
            ('active_dims', 'Active Dimensions', (-1.5, 31.5))
        ]

        for i, (col, title, ylim) in enumerate(metrics):
            ax = axes[i]
            for d in data:
                if col not in d['mean'].columns: continue
                taus = d['mean']['lambda_tau'].values
                means = d['mean'][col].values
                error_band = d['sem'][col].values
                ax.plot(taus, means, color=d['color'], label=d['label'], linewidth=1.2, linestyle=d['ls'], alpha=0.8)
                ax.fill_between(taus, means - error_band, means + error_band, color=d['color'], alpha=0.1)
            
            ax.set_xscale(x_scale)
            ax.set_xlim(x_start, x_axis_limit)
            ax.set_ylim(ylim)
            ax.grid(linestyle=':', alpha=0.8, which='both')
            ax.tick_params(labelsize=8)
            if i == 1:
                ax.set_title(title, fontsize=9)

        fig.supxlabel(r'Regularization Strength ($\tau$)', fontsize=9)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.2))
        
        fname = f"tau_sensitivity_{suffix.split('_')[0]}_{scale_name}.pdf"
        plt.savefig(os.path.join(FIGURE_DIR, fname), bbox_inches='tight')
        
        # New: Export Active Dimensions (K) separately for n32 linear
        if suffix == 'n32_merged' and scale_name == 'linear':
            fig_k, ax_k = plt.subplots(figsize=(3.5, 2))
            
            for d in data:
                if 'active_dims' not in d['mean'].columns: continue
                taus = d['mean']['lambda_tau'].values
                means = d['mean']['active_dims'].values
                error_band = d['sem']['active_dims'].values
                ax_k.plot(taus, means, color=d['color'], label=d['label'], linewidth=1.2, linestyle=d['ls'], alpha=0.8)
                ax_k.fill_between(taus, means - error_band, means + error_band, color=d['color'], alpha=0.1)
            
            # ax_k.set_xlabel(r'Regularization Strength ($\tau$)', fontsize=10)
            ax_k.set_ylabel("Active Dimensions ($K$)", fontsize=10)
            ax_k.set_xlim(x_start, x_axis_limit)
            ax_k.set_ylim((-1.5, 31.5))
            ax_k.grid(linestyle=':', alpha=0.8)
            ax_k.tick_params(labelsize=9)
            
            ax_k.legend(loc='best', fontsize=8, frameon=True)
            
            k_fname = f"tau_sensitivity_{suffix.split('_')[0]}_k_only.pdf"
            plt.savefig(os.path.join(FIGURE_DIR, k_fname), bbox_inches='tight')
            plt.close(fig_k)

        plt.close()

    print(f"Paper 2-panel sensitivity plots generated in {FIGURE_DIR}")

def plot_dimensionality_bar():
    """Create a side-by-side bar chart comparison of K for key model stages."""
    plt.rcParams.update(get_bundle())
    
    # Configuration for the specific bars
    # Label, Filename, Target Tau
    configs = [
        ('Pre_32 (Beta)', 'amortized_irt_sae_beta_pre_32_n_max.csv', 0.16),
        ('Post_max', 'amortized_irt_sae_beta_n_max.csv', 0.0535)
    ]
    
    names = []
    means = []
    errors = []
    
    for label, filename, target_tau in configs:
        path = os.path.join(RESULT_DIR, filename)
        if not os.path.exists(path):
            continue
            
        df = pd.read_csv(path, on_bad_lines='skip')
        df['lambda_tau'] = pd.to_numeric(df['lambda_tau'], errors='coerce')
        df['active_dims'] = pd.to_numeric(df['active_dims'], errors='coerce')
        df = df.dropna(subset=['lambda_tau', 'active_dims'])
        
        # Filter for the specific tau
        df_target = df[np.isclose(df['lambda_tau'], target_tau, atol=1e-4)]
        
        if len(df_target) > 0:
            # Group by seed to get distribution across seeds
            seed_means = df_target.groupby('seed')['active_dims'].mean()
            names.append(label)
            means.append(seed_means.mean())
            errors.append(seed_means.sem()) # Use SEM for error bars
            
    if not names:
        return
        
    fig, ax = plt.subplots(figsize=(4, 3.2))
    
    # Create the bar plot
    bars = ax.bar(names, means, yerr=errors, color=pc.BLUE, alpha=0.8, capsize=3, ecolor='black', error_kw={'lw': 0.5, 'capthick': 0.5, 'capsize': 2})
    
    ax.set_ylabel('Active Dimensions ($K$)')
    ax.set_ylim(0, 32)
    ax.grid(axis='y', linestyle=':', alpha=0.8)
    
    # Polish axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "dimensionality_bar_comparison.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"Dimensionality bar comparison generated: {os.path.join(FIGURE_DIR, 'dimensionality_bar_comparison.pdf')}")


def plot_double_stacked_performance_bars():
    """Double-bar architecture: Pre (Left, Red) vs Post (Right, Blue). Exports both Stacked and Base versions."""
    plt.rcParams.update(get_bundle())
    
    architectures = ['Naive', 'Rasch', 'kNN', 'MIRT', 'SAE', 'PCA', 'RAW']
    
    baseline_df = load_baseline_cache()

    def get_val_stats(filename, arch_type):
        """Returns (mean, sem) for the best tau or baseline."""
        path = os.path.join(RESULT_DIR, filename)
        if not os.path.exists(path): return 0.5, 0.0
        try:
            df = pd.read_csv(path, on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            # Coerce numeric
            numeric_cols = ['lambda_tau', 'auc_amortized', 'auc_rasch', 'auc_naive', 'auc_2pl', 'auc_mirt', 'auc_knn']
            for c in numeric_cols:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Find best tau or unique baseline
            if arch_type in ['SAE', 'PCA', 'RAW']:
                # For amortized, find best tau based on AUC
                df = df.dropna(subset=['auc_amortized', 'lambda_tau'])
                if df.empty: return 0.5, 0.0
                best_tau = df.groupby('lambda_tau')['auc_amortized'].mean().idxmax()
                best_df = df[df['lambda_tau'] == best_tau]
                metric_col = 'auc_amortized'
            else:
                # Classical baselines are sourced from the dedicated baseline cache.
                model_type = 'bernoulli' if '_bernoulli_' in filename else 'beta'
                if 'scenario' in df.columns and df['scenario'].notna().any():
                    pre_revision = str(df['scenario'].dropna().iloc[0]).replace('Pre-', '').strip().lower()
                elif '_pre_' in filename:
                    pre_revision = filename.split('_pre_')[1].split('_')[0].strip().lower()
                else:
                    pre_revision = 'none'

                n_vals = pd.to_numeric(df['n_samples'], errors='coerce').dropna() if 'n_samples' in df.columns else pd.Series(dtype=float)
                n_samples = int(n_vals.iloc[0]) if not n_vals.empty else None

                metric_col_map = {
                    'Naive': 'auc_naive',
                    'Rasch': 'auc_rasch',
                    'kNN': 'auc_knn',
                    'MIRT': 'auc_mirt',
                }
                metric_col = metric_col_map.get(arch_type)
                if metric_col is None:
                    return 0.5, 0.0

                m, s = lookup_baseline_auc(baseline_df, model_type, n_samples, pre_revision, metric_col)
                if m is not None:
                    return m, s

                # Fallback to historical inline columns if still present.
                best_df = df.drop_duplicates(subset=['seed'])
            
            if metric_col not in best_df.columns: return 0.5, 0.0
            vals = best_df[metric_col].dropna()
            return vals.mean(), vals.sem() if len(vals) > 1 else 0.0
        except: return 0.5, 0.0

    # Data Collection
    pre32_m, pre32_s = [], []
    premx_m, premx_s = [], []
    pbern_m, pbern_s = [], []
    pbeta_m, pbeta_s = [], []
    
    file_variants = {
        'Naive': {
            'pre32': 'amortized_irt_ones_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_ones_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_ones_bernoulli_n_1.csv',
            'beta': 'amortized_irt_ones_beta_n_max.csv',
        },
        'Rasch': {
            # Rasch is sourced from baseline cache via lookup_baseline_auc.
            # Use neutral sweep files only to provide scenario/n_samples context.
            'pre32': 'amortized_irt_ones_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_ones_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_ones_bernoulli_n_1.csv',
            'beta': 'amortized_irt_ones_beta_n_max.csv',
        },
        'kNN': {
            # kNN is sourced from baseline cache via lookup_baseline_auc.
            # Use the same context files used by other classical baselines.
            'pre32': 'amortized_irt_ones_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_ones_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_ones_bernoulli_n_1.csv',
            'beta': 'amortized_irt_ones_beta_n_max.csv',
        },
        'MIRT': {
            # Use the same setup context as other baselines for apples-to-apples
            # comparisons in hybrid bars (Pre-32, Post Bernoulli N=1, Post Beta N=max).
            'pre32': 'amortized_irt_ones_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_ones_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_ones_bernoulli_n_1.csv',
            'beta': 'amortized_irt_ones_beta_n_max.csv',
        },
        'SAE': {
            'pre32': 'amortized_irt_sae_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_sae_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_sae_bernoulli_n_1.csv',
            'beta': 'amortized_irt_sae_beta_n_max.csv',
        },
        'PCA': {
            'pre32': 'amortized_irt_pca_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_pca_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_pca_bernoulli_n_1.csv',
            'beta': 'amortized_irt_pca_beta_n_max.csv',
        },
        'RAW': {
            'pre32': 'amortized_irt_raw_bernoulli_pre_32_n_1.csv',
            'premax': 'amortized_irt_raw_beta_pre_max_n_max.csv',
            'bernoulli': 'amortized_irt_raw_bernoulli_n_1.csv',
            'beta': 'amortized_irt_raw_beta_n_max.csv',
        },
    }

    for arch in architectures:
        variants = file_variants.get(arch, {})
        m32, s32 = get_val_stats(variants.get('pre32', ''), arch)
        mmx, smx = get_val_stats(variants.get('premax', ''), arch)
        mbr, sbr = get_val_stats(variants.get('bernoulli', ''), arch)
        mbt, sbt = get_val_stats(variants.get('beta', ''), arch)
        
        pre32_m.append(m32); pre32_s.append(s32)
        premx_m.append(mmx); premx_s.append(smx)
        pbern_m.append(mbr); pbern_s.append(sbr)
        pbeta_m.append(mbt); pbeta_s.append(sbt)

    pre_base = np.array(pre32_m)
    pre_stack = np.maximum(0, np.array(premx_m) - pre_base)
    post_base = np.array(pbern_m)
    post_stack = np.maximum(0, np.array(pbeta_m) - post_base)

    # Export Loop: mode can be 'full', 'base', or 'hybrid'
    plot_configs = [
        ('full', 'best_auc_double_bars.pdf'),
        ('base', 'best_auc_base_only.pdf'),
        ('hybrid', 'best_auc_hybrid_stacked.pdf')
    ]

    for mode, fname in plot_configs:
        fig, ax = plt.subplots(figsize=(3.5, 2))
        
        # Filter architectures for hybrid mode
        curr_archs = list(architectures)
        curr_pre_base = np.array(pre_base)
        curr_pre_stack = np.array(pre_stack)
        curr_post_base = np.array(post_base)
        curr_post_stack = np.array(post_stack)
        curr_pre32_s = np.array(pre32_s)
        curr_premx_s = np.array(premx_s)
        curr_pbern_s = np.array(pbern_s)
        curr_pbeta_s = np.array(pbeta_s)

        if mode in ['full', 'base']:
            mask = [i for i, a in enumerate(curr_archs) if a != 'MIRT']
            curr_archs = [curr_archs[i] for i in mask]
            curr_pre_base = curr_pre_base[mask]
            curr_pre_stack = curr_pre_stack[mask]
            curr_post_base = curr_post_base[mask]
            curr_post_stack = curr_post_stack[mask]
            curr_pre32_s = curr_pre32_s[mask]
            curr_premx_s = curr_premx_s[mask]
            curr_pbern_s = curr_pbern_s[mask]
            curr_pbeta_s = curr_pbeta_s[mask]

        if mode == 'hybrid':
            # Hybrid order: Naive, Rasch, MIRT, kNN, ARAF (RAW)
            hybrid_order = ['Naive', 'Rasch', 'MIRT', 'kNN', 'RAW']
            index_by_arch = {a: i for i, a in enumerate(curr_archs)}
            order_idx = [index_by_arch[a] for a in hybrid_order if a in index_by_arch]

            curr_archs = [curr_archs[i] for i in order_idx]
            curr_pre_base = curr_pre_base[order_idx]
            curr_pre_stack = curr_pre_stack[order_idx]
            curr_post_base = curr_post_base[order_idx]
            curr_post_stack = curr_post_stack[order_idx]
            curr_pre32_s = curr_pre32_s[order_idx]
            curr_premx_s = curr_premx_s[order_idx]
            curr_pbern_s = curr_pbern_s[order_idx]
            curr_pbeta_s = curr_pbeta_s[order_idx]

        x = np.arange(len(curr_archs))
        width = 0.35

        # Determine what to plot for Pre
        # Pre: 32 Base (Light Red / Salmon)
        label_pre = f'{"32 Test Takers" if mode == "full" else "Pre Revision"}'
        ax.bar(x - width/2, curr_pre_base, width, color='salmon', 
               edgecolor='black', linewidth=0.5, label=label_pre, alpha=0.8,
               yerr=curr_pre32_s, error_kw={'lw': 0.5, 'capsize': 2})
        if mode == 'full':
            ax.bar(x - width/2, curr_pre_stack, width, bottom=curr_pre_base, color='tab:red', 
                   edgecolor='black', linewidth=0.5, label='124 Test Takers', alpha=0.8,
                   yerr=curr_premx_s, error_kw={'lw': 0.5, 'capsize': 2})
        
        # Determine what to plot for Post
        # Post: Bernoulli (Yellow / Gold)
        ax.bar(x + width/2, curr_post_base, width, color='skyblue', 
               edgecolor='black', linewidth=0.5, label='Bernoulli', alpha=0.8,
               yerr=curr_pbern_s, error_kw={'lw': 0.5, 'capsize': 2}) 
        if mode in ['full', 'hybrid']:
            ax.bar(x + width/2, curr_post_stack, width, bottom=curr_post_base, color='tab:blue', 
                   edgecolor='black', linewidth=0.5, label='Beta', alpha=0.8,
                   yerr=curr_pbeta_s, error_kw={'lw': 0.5, 'capsize': 2})

        ax.set_ylabel('AUC', fontsize=10)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_xticks(x)
        display_labels = []
        for arch in curr_archs:
            if mode == 'hybrid' and arch == 'RAW':
                display_labels.append('ARAF')
            elif arch in ['SAE', 'PCA', 'RAW']:
                display_labels.append(f"ARAF\n({arch})")
            else:
                display_labels.append(arch)
        ax.set_xticklabels(display_labels, fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylim(0.48, 0.82)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        # Consistent legend
        handles, labels = ax.get_legend_handles_labels()
        ncol = 2 if mode in ['full', 'hybrid'] else 1
        ax.legend(handles, labels, loc='best', fontsize=8, ncol=ncol, frameon=True)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
        plt.savefig(os.path.join(FIGURE_DIR, fname), bbox_inches='tight')
        plt.close()

    print(f"Update: All AUC plot variants (Full, Base, Hybrid) generated in {FIGURE_DIR}")

def main():
    print("=" * 60)
    print("GENERATING INTERPRETABILITY PLOTS")
    print("=" * 60)
    ensure_main_result_dir()
    
    plot_stability_comparison()
    plot_dimensionality_bar()
    plot_loading_heatmap()
    plot_semantic_alignment()
    plot_tau_sensitivity()
    plot_double_stacked_performance_bars()
    
    print(f"Interpretability plots generated in {FIGURE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
