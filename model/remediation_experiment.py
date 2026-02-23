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
    resmat_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'post-revision', 'colbench_backend_programming', 'resmat')

    colbench_files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
    dfs = []
    for f in colbench_files:
        df = pd.read_csv(os.path.join(resmat_dir, f), index_col=0)
        if len(df) < 5:
            continue
        # Fix agents formatting
        import re
        normalized_indices = []
        for idx in df.index:
            name = str(idx).replace("colbench.", "")
            name = re.sub(r'^(?:moon|sun)\d+_', '', name)
            normalized_indices.append(f"colbench.{name}")
        df.index = normalized_indices
        df.columns = [f"colbench_backend_programming.{c}" if not str(c).startswith("colbench") else c for c in df.columns]
        dfs.append(df)
        
    global_shared = sorted(list(set(dfs[0].index).intersection(*[set(d.index) for d in dfs[1:]])))
    return dfs, global_shared

def load_pre_revision_data():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_score_path = os.path.join(repo_root, 'item-editor', 'eval_response_matrix', 'pre-revision', 'colbench_backend_programming', 'raw_score.csv')
    df = pd.read_csv(raw_score_path, index_col=0)
    df.columns = [f"colbench_backend_programming.{c}" if not str(c).startswith("colbench") else c for c in df.columns]
    return df

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

    N, J = target_df.shape
    J_indices = np.arange(J)
    np.random.shuffle(J_indices)

    n_test = int(TEST_SIZE * J)
    test_idx = J_indices[:n_test]
    train_idx = J_indices[n_test:]

    oracle_values_clean = np.nan_to_num(oracle_df.values, nan=0.5)
    y_oracle = torch.from_numpy(oracle_values_clean.astype(np.float32)).to(device)

    train_values_clean = np.nan_to_num(target_df.values, nan=0.5)
    y_train = torch.from_numpy(train_values_clean.astype(np.float32)).to(device)

    train_mask = np.zeros_like(target_df.values, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(target_df.values)[:, train_idx]
    
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

def run_scenario(scenario_name, target_df, oracle_df):
    print(f"\n{'=' * 50}")
    print(f"Running Scenario: {scenario_name}")
    print(f"{'=' * 50}")
    
    x_j = get_embeddings(target_df)
    data = prepare_tensor_data(target_df, oracle_df, x_j)
    
    # Do not evaluate if there are no valid items
    if not data['item_mask'].any():
        print("No valid items with variance! Skipping...")
        return None
        
    model = AmortizedIRTModel(data['N'], data['J'], K_MODEL, data['embedding_dim'], data['x_j'], dropout=0.5).to(device)

    best_rmse = train_amortized_irt(model, data['y_train'], data['train_mask_t'], data['y_oracle'], data['test_mask'],
                                    model_type='beta', beta_phi=BETA_PHI, epochs=1000)

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

    # Scenario A: Post-revision, N=8
    # Randomly sample 8 runs and average them as the target.
    # The Oracle is all 54 runs averaged.
    sampled_indices = np.random.choice(len(post_dfs), 8, replace=False)
    target_dfs_A = [post_dfs[i].loc[post_shared] for i in sampled_indices]
    oracle_dfs_A = [df.loc[post_shared] for df in post_dfs]
    
    all_col_A = sorted(list(set().union(*[df.columns for df in target_dfs_A])))
    target_A_stacked = np.array([df.reindex(columns=all_col_A).values for df in target_dfs_A], dtype=float)
    target_A = pd.DataFrame(np.nanmean(target_A_stacked, axis=0), index=post_shared, columns=all_col_A)
    
    oracle_A_stacked = np.array([df.reindex(columns=all_col_A).values for df in oracle_dfs_A], dtype=float)
    oracle_A = pd.DataFrame(np.nanmean(oracle_A_stacked, axis=0), index=post_shared, columns=all_col_A)
    
    res_A = run_scenario('Post-8', target_A, oracle_A)
    if res_A: results.append(res_A)

    # Scenario B: Pre-revision, N=8
    # Randomly sample 8 agents from the 22 pre-revision agents. 
    # Use this single run as both target and oracle (with random train/test items).
    sampled_agents = np.random.choice(pre_df.index, 8, replace=False)
    target_B = pre_df.loc[sampled_agents]
    oracle_B = target_B.copy()
    res_B = run_scenario('Pre-8', target_B, oracle_B)
    if res_B: results.append(res_B)

    # Scenario C: Pre-revision, N=22
    # Use all 22 agents as target and oracle.
    target_C = pre_df.copy()
    oracle_C = target_C.copy()
    res_C = run_scenario('Pre-22', target_C, oracle_C)
    if res_C: results.append(res_C)

    # Export
    out_path = os.path.join(RESULT_DIR, 'remediation_impact.csv')
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == '__main__':
    main()
