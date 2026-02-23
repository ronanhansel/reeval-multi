import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure figures directory exists
FIGURES_DIR = os.path.join('paper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. AUC and RMSE Curves
def plot_learning_curves():
    csv_files = {
        'PCA': 'model/result/amortized_irt_pca_beta.csv',
        'SAE': 'model/result/amortized_irt_sae_beta.csv',
        'RAW': 'model/result/amortized_irt_raw_beta.csv'
    }
    
    # Load data
    data = {}
    for name, path in csv_files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found. Ensure sweeps have run.")
            
    if not data:
        print("No CSV data found to plot learning curves!")
        return

    # Assuming all dfs share the baseline columns, use the first one available
    base_df = next(iter(data.values()))
    
    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(base_df['n_samples'], base_df['auc_mean'], 'k--', label='Global Mean')
    plt.plot(base_df['n_samples'], base_df['auc_rasch'], 'k-', label='Rasch IRT')
    
    colors = {'PCA': 'blue', 'SAE': 'red', 'RAW': 'green'}
    for name, df in data.items():
        if 'auc_amortized' in df.columns:
            plt.plot(df['n_samples'], df['auc_amortized'], color=colors[name], label=f'Amortized IRT ({name})', linewidth=2)
            
    plt.xlabel('Number of Response Matrix Samples (N)')
    plt.ylabel('Test AUC')
    plt.title('Test AUC vs Matrix Capacity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'auc_comparison.pdf'), bbox_inches='tight')
    plt.close()

    # Plot RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(base_df['n_samples'], base_df['rmse_mean'], 'k--', label='Global Mean')
    plt.plot(base_df['n_samples'], base_df['rmse_rasch'], 'k-', label='Rasch IRT')
    
    for name, df in data.items():
        if 'rmse_amortized' in df.columns:
            plt.plot(df['n_samples'], df['rmse_amortized'], color=colors[name], label=f'Amortized IRT ({name})', linewidth=2)
            
    plt.xlabel('Number of Response Matrix Samples (N)')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE vs Matrix Capacity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'rmse_convergence.pdf'), bbox_inches='tight')
    plt.close()
    
# 2. Resmat Helper
def load_and_average_resmats(benchmark):
    dir_path = os.path.join('item-editor', 'eval_response_matrix', 'post-revision', benchmark, 'resmat')
    if not os.path.exists(dir_path):
        return None, None
    files = sorted([f for f in os.listdir(dir_path) if f.startswith('resmat')])
    if not files:
        return None, None
        
    dfs = []
    prefix0_df = None
    
    for f in files:
        df = pd.read_csv(os.path.join(dir_path, f), index_col=0)
        # Ensure colbench items are uniformly named to prevent shape mismatch if some are missing
        df.columns = [f"{benchmark}.{c}" if not str(c).startswith(benchmark) else c for c in df.columns]
        dfs.append(df)
        
        # '0.csv' signifies the original binary iteration
        if '0.csv' in f and prefix0_df is None:
            prefix0_df = df
            
    if prefix0_df is None:
        prefix0_df = dfs[0]

    all_cols = sorted(list(set().union(*[df.columns for df in dfs])))
    shared_idx = sorted(list(set.intersection(*[set(df.index) for df in dfs])))
    
    aligned_dfs = [df.loc[shared_idx].reindex(columns=all_cols) for df in dfs]
    stacked = np.array([df.values for df in aligned_dfs], dtype=float)
    beta_mean = np.nanmean(stacked, axis=0)
    
    beta_df = pd.DataFrame(beta_mean, index=shared_idx, columns=all_cols)
    prefix0_clean = prefix0_df.loc[shared_idx].reindex(columns=all_cols)
    return prefix0_clean, beta_df

# 3. ColBench Plotting
def plot_colbench():
    pre, post = load_and_average_resmats('colbench_backend_programming')
    if pre is None:
        return
        
    # Colbench Binary (pre)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pre, cmap='coolwarm', cbar=False, vmin=0, vmax=1)
    plt.title("ColBench Binary Responses (Iteration 0)")
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(FIGURES_DIR, 'colbench_response_matrix_Y1.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Colbench Beta (post)
    plt.figure(figsize=(12, 6))
    sns.heatmap(post, cmap='coolwarm', cbar=True, vmin=0, vmax=1)
    plt.title("ColBench Subskill Density (Averaged $\hat{P}$)")
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(FIGURES_DIR, 'colbench_empirical_probability_matrix_P_hat.png'), bbox_inches='tight', dpi=300)
    plt.close()

# 4. Other Benchmarks Panels (SAB, Scicode, CoreBench)
def plot_benchmark_panels():
    benchmarks = ['scienceagentbench', 'scicode', 'corebench_hard']
    titles = ['ScienceAgentBench', 'SciCode', 'CoreBench-Hard']
    
    data = []
    for b in benchmarks:
        pre, post = load_and_average_resmats(b)
        data.append((pre, post))
        
    if not any(pre is not None for pre, _ in data):
        return

    # Panel Pre (Binary)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, (pre, _), title) in enumerate(zip(axes, data, titles)):
        if pre is not None:
            sns.heatmap(pre, cmap='coolwarm', ax=ax, cbar=(i==2), vmin=0, vmax=1)
            ax.set_title(f"{title} (Initial Binary)")
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hal_response_matrix_panel_pre.pdf'))
    plt.close()

    # Panel Post (Beta / continuous gradients for fixed items)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, (_, post), title) in enumerate(zip(axes, data, titles)):
        if post is not None:
            sns.heatmap(post, cmap='coolwarm', ax=ax, cbar=(i==2), vmin=0, vmax=1)
            ax.set_title(f"{title} (Interpolated Gradients)")
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hal_response_matrix_panel_post.pdf'))
    plt.close()

if __name__ == '__main__':
    print("Plotting figures...")
    plot_learning_curves()
    plot_colbench()
    plot_benchmark_panels()
    print("Done generating paper figures.")
