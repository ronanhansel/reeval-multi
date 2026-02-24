import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ast
import random
import warnings

from utils import compute_rmse, evaluate_auc, get_valid_item_mask
from amortized_irt import AmortizedIRTModel, train_amortized_irt, RESULT_DIR, RANDOM_SEED, K_MODEL, LAMBDA_TAU, TEST_SIZE, BETA_PHI, device

warnings.filterwarnings('ignore')

def load_post_revision_data():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    post_rev_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'post-revision')

    # 1. Load ColBench (main anchors)
    dfs = []
    colbench_resmat_dir = os.path.join(post_rev_dir, 'colbench_backend_programming', 'resmat')
    for f in sorted(os.listdir(colbench_resmat_dir)):
        if not f.startswith('resmat'): continue
        df = pd.read_csv(os.path.join(colbench_resmat_dir, f), index_col=0)
        df.columns = [f"colbench_backend_programming.{c}" if not str(c).startswith("colbench") else c for c in df.columns]
        dfs.append(df)
        
    # 2. Load other benchmarks response matrices
    other_benchmarks = [b for b in os.listdir(post_rev_dir) if b != 'colbench_backend_programming' and os.path.isdir(os.path.join(post_rev_dir, b))]
    
    max_other_runs = 0
    for benchmark in other_benchmarks:
        b_resmat_dir = os.path.join(post_rev_dir, benchmark, 'resmat')
        if os.path.exists(b_resmat_dir):
            b_files = [f for f in os.listdir(b_resmat_dir) if f.startswith('resmat')]
            max_other_runs = max(max_other_runs, len(b_files))
            
    other_dfs = []
    for i in range(max_other_runs):
        combined_df = None
        for benchmark in other_benchmarks:
            b_resmat_dir = os.path.join(post_rev_dir, benchmark, 'resmat')
            if not os.path.exists(b_resmat_dir): continue
            
            b_files = sorted([f for f in os.listdir(b_resmat_dir) if f.startswith('resmat')])
            if i < len(b_files):
                df = pd.read_csv(os.path.join(b_resmat_dir, b_files[i]), index_col=0)
                df.columns = [f"{benchmark}.{c}" if not str(c).startswith(benchmark) else c for c in df.columns]
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.join(df, how='outer')
        other_dfs.append(combined_df)

    # Combine ColBench with the rest
    final_dfs = []
    for i, colbench_df in enumerate(dfs):
        if i < len(other_dfs) and other_dfs[i] is not None:
            other_df_to_join = other_dfs[i]
        elif len(other_dfs) > 0 and other_dfs[-1] is not None:
            other_df_to_join = other_dfs[-1]
        else:
            other_df_to_join = None
            
        if other_df_to_join is not None:
            final_df = colbench_df.join(other_df_to_join, how='outer')
        else:
            final_df = colbench_df
            
        final_dfs.append(final_df)
        
    global_shared = sorted(list(set(final_dfs[0].index).intersection(*[set(d.index) for d in final_dfs[1:]])))
    
    print(f"Loaded {len(final_dfs)} post-revision matrices")
    print(f"Agents in first run: {len(final_dfs[0].index)}")
    print(f"Global shared agents: {len(global_shared)}")
    
    return final_dfs, global_shared

def load_pre_revision_data():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pre_rev_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'pre-revision')
    
    benchmarks = ['colbench_backend_programming', 'corebench_hard', 'scicode', 'scienceagentbench']
    
    combined_df = None
    for b in benchmarks:
        raw_score_path = os.path.join(pre_rev_dir, b, 'raw_score.csv')
        if not os.path.exists(raw_score_path): continue
            
        df = pd.read_csv(raw_score_path, index_col=0)
        # Prefix columns if they don't have the benchmark string
        df.columns = [f"{b}.{c}" if not str(c).startswith(b) else c for c in df.columns]
        
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')
            
    return combined_df

def get_embeddings(target_df, embedding_type='sae', embedding_dim=48):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_emb_dir = os.path.join(repo_root, 'model', 'processed_embeddings')
    emb_file = os.path.join(processed_emb_dir, f'embeddings_{embedding_type}_{embedding_dim}.pkl')
    if not os.path.exists(emb_file):
        emb_file = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'all_benchmarks_embeddings_4096_8B.pkl')
    
    emb_df = pd.read_pickle(emb_file)
    raw_embs_map = {}
    id_col = 'task_id' if 'task_id' in emb_df.columns else 'benchmark.task_id'
    for _, r in emb_df.iterrows():
        task_id = str(r[id_col])
        raw_embs_map[task_id] = r['embedding']
        if task_id.startswith('colbench_backend_programming'):
            suffix = task_id.split('.')[-1]
            raw_embs_map[f'colbench.{suffix}'] = r['embedding']

    task_ids = target_df.columns.tolist()
    embeddings = []
    for task_id in task_ids:
        emb = raw_embs_map.get(str(task_id))
        if emb is None and task_id.startswith('colbench.'):
            number = task_id.split('.')[-1]
            emb = raw_embs_map.get(f'colbench_backend_programming.{number}')
        if emb is None:
            sample_emb = next(iter(raw_embs_map.values()))
            emb = np.zeros(len(sample_emb) if hasattr(sample_emb, '__len__') else 4096)
        elif isinstance(emb, str):
            emb = ast.literal_eval(emb)
        embeddings.append(np.array(emb, dtype=np.float32))

    embeddings = np.stack(embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return torch.tensor(embeddings, dtype=torch.float32).to(device)

def prepare_tensor_data(target_df, oracle_df, x_j):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    N, J = oracle_df.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    oracle_values_clean = np.nan_to_num(oracle_df.values, nan=0.5)
    y_oracle = torch.from_numpy(oracle_values_clean.astype(np.float32)).to(device)

    train_values_clean = np.nan_to_num(target_df.values, nan=0.5)
    y_train = torch.from_numpy(train_values_clean.astype(np.float32)).to(device)

    train_mask = np.zeros_like(oracle_df.values, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(oracle_df.values)[:, train_idx]
    
    test_mask = np.zeros_like(oracle_df.values, dtype=bool)
    test_mask[:, test_idx] = ~np.isnan(oracle_df.values)[:, test_idx]

    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    # Variance-aware mask
    item_mask = get_valid_item_mask(y_train)

    return {
        'y_train': y_train,
        'y_oracle': y_oracle,
        'train_mask_t': train_mask_t,
        'test_mask': test_mask,
        'test_mask_t': test_mask_t,
        'item_mask': item_mask,
        'x_j': x_j,
        'N': N,
        'J': J,
        'embedding_dim': x_j.shape[1]
    }

def run_scenario(scenario_name, target_df, oracle_df, lambda_tau=LAMBDA_TAU, model_type='beta', epochs=1000):
    print(f"\n{'=' * 50}")
    print(f"Running Scenario: {scenario_name} with LAMBDA_TAU={lambda_tau} (Model: {model_type})")
    print(f"{'=' * 50}")
    
    x_j = get_embeddings(target_df)
    data = prepare_tensor_data(target_df, oracle_df, x_j)
    
    # Do not evaluate if there are no valid items
    if not data['item_mask'].any():
        print("No valid items with variance! Skipping...")
        return None
        
    model = AmortizedIRTModel(data['N'], data['J'], K_MODEL, data['embedding_dim'], data['x_j'], dropout=0.5).to(device)

    best_rmse = train_amortized_irt(model, data['y_train'], data['train_mask_t'], data['y_oracle'], data['test_mask'],
                                    model_type=model_type, beta_phi=BETA_PHI, epochs=epochs, lambda_tau=lambda_tau)

    model.eval()
    with torch.no_grad():
        p_amortized = model()
        final_auc = evaluate_auc(p_amortized, data['y_oracle'], data['test_mask_t'], item_mask=data['item_mask'])
        final_rmse = compute_rmse(p_amortized.cpu().numpy(), data['y_oracle'].cpu().numpy(), data['test_mask'], item_mask=data['item_mask'])
        
        tau_val = model.get_tau()
        active_mask = tau_val > 0.01
        active_dims = active_mask.sum().item()
        
        active_dim_indices = torch.nonzero(active_mask).squeeze().cpu().tolist()
        if isinstance(active_dim_indices, int):
            active_dim_indices = [active_dim_indices]
            
        if active_dims == 0:
            active_dim_indices = []

    print(f"Results for {scenario_name}:")
    print(f"  Test RMSE (valid items): {final_rmse:.4f}")
    print(f"  Test AUC (valid items):  {final_auc:.4f}")
    print(f"  Active Dims:             {active_dims} -> {active_dim_indices}")
    
    return {
        'scenario': scenario_name,
        'test_rmse': final_rmse,
        'test_auc': final_auc,
        'active_dims': active_dims,
        'active_indices': str(active_dim_indices)
    }

def main():
    np.random.seed(RANDOM_SEED)
    
    print("Loading Post-Revision Data...")
    post_dfs, post_shared = load_post_revision_data()
    print(f"Loaded {len(post_dfs)} post-revision matrices from pristine set")

    print("\nLoading Pre-Revision Data...")
    pre_df = load_pre_revision_data()
    print(f"Loaded pre-revision matrix with {pre_df.shape[0]} agents ('broken matrices')")

    results = []

    # Scenario A: Post-8
    # Selection: the first run as the target, restricted strictly to the 24 global_shared agents.
    # This precisely matches the 24-agent intersection used by SOTA to attain 0.83 AUC.
    all_columns = sorted(list(set().union(*[df.columns for df in post_dfs])))
    
    sampled_index = 0
    target_A = post_dfs[sampled_index].loc[post_shared].reindex(columns=all_columns).copy()
    
    # Oracle is the true probabilistic target across all 54 runs for these 24 agents.
    oracle_A_stacked = np.array([df.loc[post_shared].reindex(columns=all_columns).values for df in post_dfs], dtype=float)
    oracle_A = pd.DataFrame(np.nanmean(oracle_A_stacked, axis=0), index=post_shared, columns=all_columns)
    
    # We use 1.38 lambda tau with 250 epochs (matching SOTA amortized_irt.py exactly)
    res_A = run_scenario('Post-8', target_A, oracle_A, lambda_tau=1.38, model_type='beta', epochs=250)  
    if res_A: results.append(res_A)

    # Scenario B: Pre-8
    # Pre-revision data is extremely disjoint. To ensure we can test on colbench,
    # sample 8 agents that actually evaluated colbench.
    colbench_cols = [c for c in pre_df.columns if c.startswith('colbench')]
    agents_with_colbench = pre_df.dropna(subset=colbench_cols, how='all').index
    sampled_agents = np.random.choice(agents_with_colbench, 8, replace=False)
    target_B = pre_df.loc[sampled_agents].copy()
    oracle_B = target_B.copy()
    res_B = run_scenario('Pre-8', target_B, oracle_B, lambda_tau=0.20, model_type='beta')
    if res_B: results.append(res_B)

    # Scenario C: Pre-Max (formerly Pre-22)
    # Use all agents from the pre-revision datasets (N=286 disjoint rows, outer joined).
    target_C = pre_df.copy()
    oracle_C = target_C.copy()
    res_C = run_scenario('Pre-max', target_C, oracle_C, lambda_tau=0.20, model_type='beta')
    if res_C: results.append(res_C)

    # Export
    out_path = os.path.join(RESULT_DIR, 'remediation_impact.csv')
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == '__main__':
    main()
