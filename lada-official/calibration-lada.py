#!/usr/bin/env python3
"""
LADA Calibration Script (Latent Ability Dirichlet Allocation)

Fits the LADA model using Coordinate Gradient Ascent as described in the paper.
Replaces standard MIRT calibration while maintaining the same input/output structure.

Model Specification:
    P(y=1) = sigmoid( d_j + w_j^T * theta_i )
    w_j = Softmax(phi_j)  (Dirichlet constraint via reparameterization)
    
Optimization (Coordinate Descent):
    1. Update Theta (Abilities)
    2. Update d (Difficulties)
    3. Update phi (Unconstrained weights) via Inner Adam Loop

Usage:
    python calibration-lada.py
"""

# Import and Setup
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle
import json
from collections import defaultdict

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
K_VALUES = [3, 4, 5]  
N_EPOCHS = 1000
BATCH_SIZE = 50000

# Hyperparameters
# Adjusted for Mean Reduction and faster convergence
ALPHA_DIRICHLET = 1.1   
LR_THETA = 0.05        # Increased for faster convergence
LR_D = 0.05            
LR_PHI = 0.1           # High LR needed to push through Softmax saturation
REG_STRENGTH = 0.01    # Weak regularization (matches MIRT script)
PHI_INNER_STEPS = 3    # Reduced inner steps as LR is higher

# Prior hyperparameters
MU_D = 0.0             
SIGMA_D = 1.0          

RESULT_DIR = "../result/lada-fitting"
os.makedirs(RESULT_DIR, exist_ok=True)

# For reproducibility
FULL_SEED = 86
EARLY_STOPPING_THRESHOLD = 1e-4
EARLY_STOPPING_PATIENCE = 20

# Data Loading
print("Loading data...")
try:
    with open("../data-reeval-multi/resmat.pkl", "rb") as f:
        results = pickle.load(f)
except FileNotFoundError:
    print("Error: Data file '../data-reeval-multi/resmat.pkl' not found.")
    print("Please ensure the data path is correct.")
    sys.exit(1)

print(f"Data shape: {results.shape}")

# Work directly with numpy values
resmat_values = results.values
n_persons, n_items = resmat_values.shape
scenarios = results.columns.get_level_values("scenario").unique()

print(f"n_persons: {n_persons}, n_items: {n_items}")
print(f"Total observations: {(~np.isnan(resmat_values)).sum()}")

# ------------------------------------------------------------------------------
# LADA Model Definition
# ------------------------------------------------------------------------------

class LADAModel(nn.Module):
    def __init__(self, n_users, n_items, k_dims, alpha=1.1, mu_d=0.0, sigma_d=1.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.k_dims = k_dims
        self.alpha = alpha
        self.mu_d = mu_d
        self.sigma_d = sigma_d

        # Eq 4: Theta ~ MVN(0, I)
        self.theta = nn.Parameter(torch.randn(n_users, k_dims))
        
        # Eq 2: d ~ N(mu, sigma)
        self.d = nn.Parameter(torch.zeros(n_items))
        
        # Eq 11: Reparameterization w = Softmax(phi)
        self.phi = nn.Parameter(torch.randn(n_items, k_dims))

    def get_w(self, item_indices=None):
        """Compute Dirichlet-constrained weights from phi using Softmax (Eq 11)."""
        if item_indices is not None:
            phi_subset = self.phi[item_indices]
            return F.softmax(phi_subset, dim=1)
        return F.softmax(self.phi, dim=1)

    def forward(self, user_indices, item_indices):
        """
        P(y=1) = sigma(d_j + w_j^T theta_i)  (Eq 15)
        """
        batch_theta = self.theta[user_indices]    # (Batch, K)
        batch_d = self.d[item_indices]            # (Batch)
        batch_w = self.get_w(item_indices)        # (Batch, K)
        
        # Dot product: sum(w * theta)
        interaction = (batch_w * batch_theta).sum(dim=1)
        logits = interaction + batch_d
        return logits

# ------------------------------------------------------------------------------
# LADA Fitting Function
# ------------------------------------------------------------------------------

def fit_lada_model(resmat_values, k_dims, seed=None, name="model"):
    """
    Fit LADA model using Coordinate Gradient Ascent.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    print(f"\nFitting LADA model '{name}' with K={k_dims}...")
    
    n_total_persons, n_total_items = resmat_values.shape
    
    # 1. Prepare Data
    observed_pairs = np.argwhere(~np.isnan(resmat_values))
    np.random.shuffle(observed_pairs)
    
    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    test_pairs = observed_pairs[:n_test]
    train_pairs = observed_pairs[n_test:]
    
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat_values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat_values[test_rows, test_cols]
    
    # Convert to tensors
    train_rows_t = torch.from_numpy(train_rows).long()
    train_cols_t = torch.from_numpy(train_cols).long()
    train_ys_t = torch.from_numpy(train_ys).float()
    
    print(f"  Train obs: {len(train_ys)}, Test obs: {len(test_ys)}")
    
    # Weights for class balancing
    item_counts = pd.Series(train_cols).value_counts().reindex(range(n_total_items), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights = inv_freq_weights.iloc[train_cols].values
    train_weights_t = torch.from_numpy(train_weights).float()
    
    # 2. Initialize Model
    model = LADAModel(n_total_persons, n_total_items, k_dims, 
                      alpha=ALPHA_DIRICHLET, mu_d=MU_D, sigma_d=SIGMA_D).to(device)
    
    # 3. Optimizers
    # Using Adam with higher LRs for faster convergence
    opt_theta = torch.optim.Adam([model.theta], lr=LR_THETA)
    opt_d = torch.optim.Adam([model.d], lr=LR_D)
    opt_phi = torch.optim.Adam([model.phi], lr=LR_PHI)
    
    train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t, train_weights_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )
    
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    training_history = []
    
    # 4. Training Loop (Coordinate Descent)
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{N_EPOCHS}", leave=False)
        
        for batch_rows, batch_cols, batch_ys, batch_wts in pbar:
            batch_rows = batch_rows.to(device)
            batch_cols = batch_cols.to(device)
            batch_ys = batch_ys.to(device)
            batch_wts = batch_wts.to(device)
            
            # Use MEAN reduction for scale stability (like MIRT script)
            
            # --- BLOCK 1: Update Theta (Eq 35) ---
            opt_theta.zero_grad()
            logits = model(batch_rows, batch_cols)
            
            nll = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            nll_weighted = (batch_wts * nll).mean() # Mean reduction
            
            # Scaled Prior (Reg strength * Mean L2)
            prior_theta = REG_STRENGTH * torch.mean(model.theta[batch_rows]**2)
            
            loss_theta = nll_weighted + prior_theta
            loss_theta.backward()
            opt_theta.step()
            
            # --- BLOCK 2: Update Difficulty (Eq 36) ---
            opt_d.zero_grad()
            logits = model(batch_rows, batch_cols)
            
            nll = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            nll_weighted = (batch_wts * nll).mean()
            
            d_val = model.d[batch_cols]
            # Simple L2 regularization for d, scaled by REG_STRENGTH
            prior_d = REG_STRENGTH * torch.mean(d_val**2) 
            
            loss_d = nll_weighted + prior_d
            loss_d.backward()
            opt_d.step()
            
            # --- BLOCK 3: Update Weights/Phi (Eq 37) ---
            for _ in range(PHI_INNER_STEPS):
                opt_phi.zero_grad()
                logits = model(batch_rows, batch_cols)
                
                nll = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
                nll_weighted = (batch_wts * nll).mean()
                
                # Weight Prior (Dirichlet)
                # Scaled down to match loss magnitude
                phi_batch = model.phi[batch_cols]
                term1 = torch.sum(phi_batch, dim=1)
                lse = torch.logsumexp(phi_batch, dim=1)
                term2 = model.k_dims * lse
                # Prior is typically small, scaling by REG_STRENGTH to keep it controlled
                dirichlet_term = (model.alpha - 1) * (term1 - term2).mean()
                
                # Minimize: Loss - Prior (since prior is + log prob)
                loss_phi = nll_weighted - (REG_STRENGTH * dirichlet_term)
                loss_phi.backward()
                opt_phi.step()

            epoch_loss += nll_weighted.item()
            pbar.set_postfix({'loss': f"{nll_weighted.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Validation
        with torch.no_grad():
            logits_test = model(torch.from_numpy(test_rows).to(device), 
                                torch.from_numpy(test_cols).to(device))
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
            try:
                val_auc = roc_auc_score(test_ys, probs_test)
            except ValueError:
                val_auc = 0.5
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_epoch_loss,
            'val_auc': val_auc
        })
        
        print(f"    Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_epoch_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early Stopping Logic
        if val_auc > best_auc + EARLY_STOPPING_THRESHOLD:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                'theta': model.theta.detach().clone(),
                'phi': model.phi.detach().clone(),
                'd': model.d.detach().clone()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                best_state = {
                    'theta': model.theta.detach().clone(),
                    'phi': model.phi.detach().clone(),
                    'd': model.d.detach().clone()
                }
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}. Best Val AUC: {best_auc:.4f}")
                break
    
    # Restore best model
    if best_state is None:
        best_state = {'theta': model.theta, 'phi': model.phi, 'd': model.d}
        
    model.theta.data = best_state['theta']
    model.phi.data = best_state['phi']
    model.d.data = best_state['d']
    
    # Final Metrics
    with torch.no_grad():
        logits_test = model(torch.from_numpy(test_rows).to(device), 
                            torch.from_numpy(test_cols).to(device))
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        test_auc = roc_auc_score(test_ys, probs_test)
        
        full_logits = model(train_rows_t.to(device), train_cols_t.to(device))
        final_log_likelihood = -F.binary_cross_entropy_with_logits(
            full_logits, train_ys_t.to(device), reduction='sum').item()

    num_params = model.theta.numel() + model.phi.numel() + model.d.numel()
    aic = 2 * num_params - 2 * final_log_likelihood
    bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood
    
    final_w = model.get_w().detach().cpu()
    
    history_path = os.path.join(RESULT_DIR, f"training_history_{name}_k{k_dims}.csv")
    pd.DataFrame(training_history).to_csv(history_path, index=False)
    
    return {
        'theta': model.theta.detach().cpu(),
        'phi': model.phi.detach().cpu(),
        'w': final_w,  # Derived parameter W
        'd': model.d.detach().cpu(),
        'test_auc': test_auc,
        'best_val_auc': best_auc,
        'best_epoch': best_epoch,
        'log_likelihood': final_log_likelihood,
        'aic': aic,
        'bic': bic,
        'num_params': num_params,
        'n_train': len(train_ys),
        'n_test': len(test_ys)
    }

# ------------------------------------------------------------------------------
# Main Execution (Identical structure to MIRT script)
# ------------------------------------------------------------------------------

def main(target_scenario=None):
    print("Starting LADA calibration for multiple K values...")
    
    # Determine which scenarios to run
    if target_scenario:
        if target_scenario not in scenarios:
            print(f"Error: Scenario '{target_scenario}' not found in dataset.")
            print("Available scenarios:", list(sorted(scenarios)))
            sys.exit(1)
        scenarios_to_run = [target_scenario]
        run_full_dataset = False
        print(f"Running ONLY for scenario: {target_scenario}")
    else:
        scenarios_to_run = sorted(scenarios)
        run_full_dataset = True
        print(f"Running for combined dataset and all {len(scenarios)} scenarios.")

    all_results = {}
    
    for K in K_VALUES:
        print(f"\n{'='*80}")
        print(f"PROCESSING K = {K} DIMENSIONS (LADA)")
        print(f"{'='*80}")
        
        lada_results = defaultdict(dict)
        
        # 1. Full Dataset (Only if no specific scenario requested)
        if run_full_dataset:
            print(f"\nFitting LADA on full dataset with K={K}...")
            full_result = fit_lada_model(resmat_values, K, seed=FULL_SEED, name="combined_data")
            
            lada_results["combined_data"] = full_result
            print(f"Test AUC: {full_result['test_auc']:.4f}")
            
            # Save full model
            torch.save({
                'theta': full_result['theta'],
                'phi': full_result['phi'],
                'w': full_result['w'],
                'd': full_result['d'],
                'K': K,
                'metrics': full_result
            }, os.path.join(RESULT_DIR, f"lada_model_k{K}_combined_data.pt"))

        # 2. Per Scenario
        print(f"\nFitting LADA for {len(scenarios_to_run)} scenarios with K={K}...")
        for i, scenario in enumerate(scenarios_to_run):
            mask = (results.columns.get_level_values("scenario") == scenario)
            scenario_data = resmat_values[:, mask]
            
            n_obs = (~np.isnan(scenario_data)).sum()
            if n_obs < 100:
                print(f"Skipping {scenario} - insufficient data ({n_obs} obs)")
                continue
                
            print(f"Processing {scenario} ({i+1}/{len(scenarios_to_run)})...")
            try:
                res = fit_lada_model(scenario_data, K, seed=i+1, name=scenario)
                lada_results[scenario] = res
                
                torch.save({
                    'theta': res['theta'],
                    'phi': res['phi'],
                    'w': res['w'],
                    'd': res['d'],
                    'K': K,
                    'scenario': scenario
                }, os.path.join(RESULT_DIR, f"lada_model_k{K}_{scenario}.pt"))
                
            except Exception as e:
                print(f"Error fitting {scenario}: {e}")
        
        # 3. Summaries for this K
        summary_data = []
        for name, res in lada_results.items():
            summary_data.append({
                'scenario': name,
                'K': K,
                'test_auc': res['test_auc'],
                'aic': res['aic'],
                'bic': res['bic'],
                'log_likelihood': res['log_likelihood'],
                'best_epoch': res['best_epoch']
            })
            
        if not summary_data:
            print(f"No successful fittings for K={K}")
            continue

        summary_df = pd.DataFrame(summary_data).sort_values('test_auc', ascending=False)
        print(f"\nLADA Fitting Results Summary for K={K}:")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        summary_suffix = f"_{target_scenario}" if target_scenario else ""
        summary_df.to_csv(os.path.join(RESULT_DIR, f"lada_k{K}_summary{summary_suffix}.csv"), index=False)
        
        # Save JSON
        json_res = {}
        for name, res in lada_results.items():
            json_res[name] = {k: v for k, v in res.items() if not torch.is_tensor(v)}
            # Helper for shapes
            json_res[name]['theta_shape'] = list(res['theta'].shape)
        
        with open(os.path.join(RESULT_DIR, f"lada_k{K}_complete_results{summary_suffix}.json"), 'w') as f:
            json.dump(json_res, f, indent=2)

        all_results[f"K_{K}"] = {
            'best_auc': summary_df.iloc[0]['test_auc'],
            'best_model': summary_df.iloc[0]['scenario']
        }

    # Final Overall Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY (LADA)")
    print(f"{'='*80}")
    
    summary_list = []
    for k_key, k_val in all_results.items():
        summary_list.append({
            'K': k_key,
            'Best AUC': k_val['best_auc'],
            'Best Model': k_val['best_model']
        })
    print(pd.DataFrame(summary_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-values', nargs='+', type=int, default=K_VALUES)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--scenario', '-s', type=str, default=None, 
                        help="Specific scenario to fit (skips full dataset and other scenarios)")
    args = parser.parse_args()
    
    K_VALUES = args.k_values
    N_EPOCHS = args.epochs
    
    main(target_scenario=args.scenario)