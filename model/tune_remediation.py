import os
import numpy as np
import pandas as pd
import torch

from utils import compute_rmse, evaluate_auc, get_valid_item_mask
from amortized_irt import AmortizedIRTModel, train_amortized_irt, RESULT_DIR, RANDOM_SEED, K_MODEL, TEST_SIZE, BETA_PHI, device
from remediation_experiment import load_post_revision_data, load_pre_revision_data, get_embeddings, prepare_tensor_data

def tune_scenario(scenario_name, target_df, oracle_df):
    print(f"\n{'=' * 50}")
    print(f"Tuning Scenario: {scenario_name}")
    print(f"{'=' * 50}")
    
    x_j = get_embeddings(target_df)
    data = prepare_tensor_data(target_df, oracle_df, x_j)
    
    if not data['item_mask'].any():
        print("No valid items with variance! Skipping...")
        return
        
    lambda_taus = [1.3, 1.0, 0.8, 0.5, 0.2, 0.1, 0.05, 0.01]
    
    results = []
    
    for l_tau in lambda_taus:
        print(f"\n--- Testing LAMBDA_TAU = {l_tau} ---")
        model = AmortizedIRTModel(data['N'], data['J'], K_MODEL, data['embedding_dim'], data['x_j'], dropout=0.5).to(device)
        
        train_amortized_irt(model, data['y_train'], data['train_mask_t'], data['y_oracle'], data['test_mask'],
                            model_type='beta', beta_phi=BETA_PHI, epochs=1000, lambda_tau=l_tau)
                            
        model.eval()
        with torch.no_grad():
            p_amortized = model()
            final_auc = evaluate_auc(p_amortized, data['y_oracle'], data['test_mask_t'], item_mask=data['item_mask'])
            final_rmse = compute_rmse(p_amortized.cpu().numpy(), data['y_oracle'].cpu().numpy(), data['test_mask'], item_mask=data['item_mask'])
            
            tau_val = model.get_tau()
            active_mask = tau_val > 0.01
            active_dims = active_mask.sum().item()
            
        print(f"  Test RMSE: {final_rmse:.4f} | Test AUC: {final_auc:.4f} | Active Dims: {active_dims}")
        results.append({
            'scenario': scenario_name,
            'lambda_tau': l_tau,
            'test_auc': final_auc,
            'test_rmse': final_rmse,
            'active_dims': active_dims
        })
    return results

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    print("Loading Post-Revision Data...")
    post_dfs, post_shared = load_post_revision_data()
    print("\nLoading Pre-Revision Data...")
    pre_df = load_pre_revision_data()

    all_results = []

    # Scenario A: Post-8
    sampled_indices = np.random.choice(len(post_dfs), 8, replace=False)
    target_dfs_A = [post_dfs[i].loc[post_shared] for i in sampled_indices]
    oracle_dfs_A = [df.loc[post_shared] for df in post_dfs]
    
    all_col_A = sorted(list(set().union(*[df.columns for df in target_dfs_A])))
    target_A_stacked = np.array([df.reindex(columns=all_col_A).values for df in target_dfs_A], dtype=float)
    target_A = pd.DataFrame(np.nanmean(target_A_stacked, axis=0), index=post_shared, columns=all_col_A)
    
    oracle_A_stacked = np.array([df.reindex(columns=all_col_A).values for df in oracle_dfs_A], dtype=float)
    oracle_A = pd.DataFrame(np.nanmean(oracle_A_stacked, axis=0), index=post_shared, columns=all_col_A)
    
    res_A = tune_scenario('Post-8', target_A, oracle_A)
    all_results.extend(res_A)

    # Scenario B: Pre-8
    np.random.seed(RANDOM_SEED)
    sampled_agents = np.random.choice(pre_df.index, 8, replace=False)
    target_B = pre_df.loc[sampled_agents]
    oracle_B = target_B.copy()
    res_B = tune_scenario('Pre-8', target_B, oracle_B)
    all_results.extend(res_B)

    # Scenario C: Pre-22
    target_C = pre_df.copy()
    oracle_C = target_C.copy()
    res_C = tune_scenario('Pre-22', target_C, oracle_C)
    all_results.extend(res_C)
    
    out_df = pd.DataFrame(all_results)
    print("\n\nFINAL TUNING RESULTS:")
    print(out_df.to_string())
    out_df.to_csv('model/result/remediation_tuning.csv', index=False)

if __name__ == '__main__':
    main()
