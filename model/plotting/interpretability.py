#!/usr/bin/env python3
import os
import sys
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

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures", "interpretability")
EMB_DIR = os.path.join(MODEL_DIR, "processed_embeddings")

os.makedirs(FIGURE_DIR, exist_ok=True)

# Add repo root to path for imports if needed
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Colors
MAIN_BLUE = '#1f77b4'
STABLE_GREEN = '#2ca02c'
UNSTABLE_RED = '#d62728'

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Dimensionality Stability (K-Consistency)
# ══════════════════════════════════════════════════════════════════════════════

def plot_stability_comparison():
    plt.rcParams.update(get_bundle())
    
    file_map = {
        'Pre_8': "amortized_irt_sae_bernoulli_pre_8_n_1.csv",
        'Pre_max': "amortized_irt_sae_beta_pre_max_n_1.csv",
        'Post_1': "amortized_irt_sae_bernoulli_n_1.csv",
        'Post_max': "amortized_irt_sae_beta_n_max.csv"
    }
    
    data = []
    for label, filename in file_map.items():
        path = os.path.join(RESULT_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            k_vals = df['active_dims'].tolist()
            data.append({
                'Model': label,
                'K_Mean': np.mean(k_vals),
                'K_SE': np.std(k_vals) / np.sqrt(len(k_vals)) if len(k_vals) > 1 else 0.0
            })
    
    if not data:
        print("No data found for stability comparison.")
        return

    df_plot = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(df_plot['Model'], df_plot['K_Mean'], yerr=df_plot['K_SE'], 
                  capsize=3, color=MAIN_BLUE, alpha=0.8)
    
    ax.set_ylabel("Active Dimensions ($K$)")
    ax.set_ylim(0, 32)
    ax.set_title("Latent Factor Stability")
    
    # Highlight Post_max stability
    for i, label in enumerate(df_plot['Model']):
        if label == 'Post_max':
            bars[i].set_color(MAIN_BLUE)
            
    plt.savefig(os.path.join(FIGURE_DIR, "k_stability_comparison.pdf"), bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Razor's Edge (Phase Transition)
# ══════════════════════════════════════════════════════════════════════════════

def plot_razors_edge():
    plt.rcParams.update(get_bundle())
    
    # Load PCA Bernoulli N=1 (the most unstable)
    path = os.path.join(RESULT_DIR, "amortized_irt_pca_bernoulli_n_1.csv")
    if not os.path.exists(path):
        return
        
    df = pd.read_csv(path)
    df = df.sort_values(by='seed')
    
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    
    # Scatter plot of seeds vs K
    seeds = range(len(df))
    k_vals = df['active_dims']
    
    ax.scatter(seeds, k_vals, color=UNSTABLE_RED, s=50, edgecolors='black', zorder=3)
    ax.plot(seeds, k_vals, color=UNSTABLE_RED, linestyle='--', alpha=0.4, zorder=2)
    
    ax.set_xlabel("Random Seed Rank")
    ax.set_ylabel("Active Dims ($K$)")
    ax.set_title("The Bernoulli $N=1$ 'Razor's Edge'")
    ax.set_yticks([0, 15, 30])
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Annotate collapse vs saturation
    ax.text(len(df)//4, 2, "Collapse", color=UNSTABLE_RED, fontweight='bold', ha='center')
    ax.text(len(df)//4, 27, "Saturation", color=UNSTABLE_RED, fontweight='bold', ha='center')
    
    plt.savefig(os.path.join(FIGURE_DIR, "razors_edge_instability.pdf"), bbox_inches='tight')
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
    
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.0), sharey=True)
    
    any_plotted = False
    for i, (label, w_path, pre_rev) in enumerate(configs):
        if not os.path.exists(w_path): continue
        A, tids = load_data_and_weights(w_path, pre_revision=pre_rev)
        if A is None: continue
        
        any_plotted = True
        # Filter for active dims only
        active = np.where(np.abs(A).max(axis=0) > 0.001)[0]
        if len(active) == 0: active = np.arange(A.shape[1])
        A_sub = A[:, active]
        
        # Sort items by benchmark for visual clumping
        benchmarks = [t.split('.')[0] for t in tids]
        sort_idx = np.argsort(benchmarks)
        A_sorted = A_sub[sort_idx]
        sorted_benchmarks = np.array(benchmarks)[sort_idx]
        
        sns.heatmap(A_sorted.T, ax=axes[i], cmap='RdBu_r', center=0, cbar=False if i==0 else True)
        axes[i].set_title(label, fontsize=11)
        if i == 0:
            axes[i].set_ylabel("Active Latent Dims ($K$)", fontsize=9)
            
        # Add vertical lines and custom x ticks
        unique_b = []
        counts = []
        for b in sorted_benchmarks:
            if not unique_b or unique_b[-1] != b:
                unique_b.append(b)
                counts.append(1)
            else:
                counts[-1] += 1
                
        bench_map = {
            'colbench_backend_programming': 'ColBench',
            'corebench_hard': 'CoreBench',
            'scienceagentbench': 'SAB',
            'scicode': 'SciCode'
        }
                
        boundaries = [0]
        centers = []
        for count in counts:
            boundaries.append(boundaries[-1] + count)
            centers.append(boundaries[-2] + count / 2.0)
            
        for b_idx in boundaries[1:-1]:
            axes[i].axvline(x=b_idx, color='black', linewidth=1)
            
        axes[i].set_xticks(centers)
        axes[i].set_xticklabels([bench_map.get(b, b) for b in unique_b], rotation=45, ha='right', fontsize=8)
    
    if any_plotted:
        plt.savefig(os.path.join(FIGURE_DIR, "loading_cleanliness_comparison.pdf"), bbox_inches='tight')
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
        if not os.path.exists(w_path): continue
        A, tids = load_data_and_weights(w_path, pre_revision=pre_rev)
        if A is None: continue
        
        weights = torch.load(w_path)
        tau = torch.relu(weights['tau_raw']).cpu().numpy()
        
        active = np.where(np.abs(A).max(axis=0) > 1e-6)[0]
        
        report_lines = []
        report_lines.append(f"\n### Semantic Alignment & Clarity: {label} (K={len(active)})")
        report_lines.append("| Dim | Tau Strength | Top 5 Loader Items + Context | Primary Benchmark | Purity |")
        report_lines.append("|---|---|---|---|---|")
        
        benchmarks = np.array([t.split('.')[0] for t in tids])
        
        all_purities = []
        all_entropies = []
        
        input_lookup = get_item_inputs(tids)
        
        for idx in active:
            # Loaders
            loadings = np.abs(A[:, idx])
            top_5_indices = np.argsort(loadings)[-5:][::-1]
            
            loader_details = []
            for i in top_5_indices:
                tid = tids[i]
                if loadings[i] < 1e-6: continue
                prompt = input_lookup.get(tid, "No description found")
                # Truncate and clean prompt
                prompt_snippet = prompt[:3500].replace('\n', ' ').strip() + "..."
                loader_details.append(f"**{tid}**: {prompt_snippet}")
            
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

def plot_2d_projections():
    plt.rcParams.update(get_bundle())
    
    configs = [
        ('Pre-Revision', os.path.join(RESULT_DIR, "amortized_irt_sae_beta_pre_max_n_max_seed_42_weights_final.pkl"), 'max'),
        ('Post-Revision', os.path.join(RESULT_DIR, "amortized_irt_sae_beta_n_max_seed_42_weights_final.pkl"), 'none')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    
    any_plotted = False
    for i, (label, w_path, pre_rev) in enumerate(configs):
        if not os.path.exists(w_path): continue
        A, tids = load_data_and_weights(w_path, pre_revision=pre_rev)
        if A is None: continue
            
        any_plotted = True
        # Filter for active dims
        active = np.where(np.abs(A).max(axis=0) > 1e-6)[0]
        if len(active) < 2:
            print(f"Warning: {label} has < 2 active dims. Using all {A.shape[1]}.")
            A_sub = A
        else:
            A_sub = A[:, active]
            
        print(f"Projecting {label} with {A_sub.shape[1]} active dims...")
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tids)-1))
        X_2d = tsne.fit_transform(A_sub)
        
        # Map benchmarks to cleaner names for legend
        bench_map = {
            'colbench_backend_programming': 'ColBench',
            'corebench_hard': 'CoreBench',
            'scienceagentbench': 'SAB',
            'scicode': 'SciCode'
        }
        
        # Color by benchmark
        benchmarks = [t.split('.')[0] for t in tids]
        unique_b = sorted(list(set(benchmarks)))
        palette = sns.color_palette("husl", len(unique_b))
        
        # Calculate Quantitative Clarity (Silhouette Score)
        if len(unique_b) > 1:
            score = silhouette_score(X_2d, benchmarks)
            print(f"| {label} | Benchmark Separation (Silhouette): {score:.4f} |")
        
        for b, color in zip(unique_b, palette):
            mask = [bench == b for bench in benchmarks]
            label_name = bench_map.get(b, b)
            axes[i].scatter(X_2d[mask, 0], X_2d[mask, 1], label=label_name, color=color, s=15, alpha=0.7, edgecolors='none')
            
        axes[i].set_title(f"{label} (K={A_sub.shape[1]})", fontsize=11)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        if i == 1:
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True)
            
    if any_plotted:
        plt.savefig(os.path.join(FIGURE_DIR, "loading_2d_projections.pdf"), bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("GENERATING INTERPRETABILITY PLOTS")
    print("=" * 60)
    
    plot_stability_comparison()
    plot_razors_edge()
    plot_loading_heatmap()
    plot_semantic_alignment()
    plot_2d_projections()
    
    print(f"Interpretability plots generated in {FIGURE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
