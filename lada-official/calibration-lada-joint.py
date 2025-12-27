#!/usr/bin/env python3
"""
LADA Calibration Script - JOINT OPTIMIZATION VARIANT (GPU OPTIMIZED)

Optimized for A100:
- Pre-loads entire dataset to GPU VRAM to prevent PCIe bottlenecks.
- Uses massive batch sizes (1M+) to maximize GPU compute utilization.
- Replaces CPU-bound DataLoader with manual GPU tensor slicing.

Usage:
    python calibration-lada-joint.py --k-values 3 4 5 --epochs 1000
    python calibration-lada-joint.py -s lsat_qa
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle
import json
from collections import defaultdict

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration Defaults
DEFAULT_K_VALUES = [3, 4, 5]  
DEFAULT_EPOCHS = 1000
# Huge batch size for A100 (adjust if you run out of VRAM, but A100 80GB can handle this easily)
BATCH_SIZE = 1000000 

# Hyperparameters
ALPHA_DIRICHLET = 1.1   
LR_JOINT = 0.05        
REG_STRENGTH = 0.01    

# Prior hyperparameters
MU_D = 0.0             
SIGMA_D = 1.0          

RESULT_DIR = "../result/lada-fitting-joint"
os.makedirs(RESULT_DIR, exist_ok=True)

# For reproducibility
FULL_SEED = 86
EARLY_STOPPING_THRESHOLD = 1e-4
EARLY_STOPPING_PATIENCE = 20

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
print("Loading data...")
try:
    with open("../data-reeval-multi/resmat.pkl", "rb") as f:
        results = pickle.load(f)
except FileNotFoundError:
    print("Error: Data file '../data-reeval-multi/resmat.pkl' not found.")
    sys.exit(1)

resmat_values = results.values
n_persons, n_items = resmat_values.shape
scenarios = results.columns.get_level_values("scenario").unique()

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

        # Parameters
        self.theta = nn.Parameter(torch.randn(n_users, k_dims))
        self.d = nn.Parameter(torch.zeros(n_items))
        self.phi = nn.Parameter(torch.randn(n_items, k_dims))

    def get_w(self, item_indices=None):
        if item_indices is not None:
            phi_subset = self.phi[item_indices]
            return F.softmax(phi_subset, dim=1)
        return F.softmax(self.phi, dim=1)

    def forward(self, user_indices, item_indices):
        batch_theta = self.theta[user_indices]    
        batch_d = self.d[item_indices]            
        batch_w = self.get_w(item_indices)        
        
        interaction = (batch_w * batch_theta).sum(dim=1)
        logits = interaction + batch_d
        return logits
    
    def compute_loss(self, logits, targets, weights, user_idx, item_idx, reg_strength):
        # 1. Weighted Binary Cross Entropy
        nll = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        nll_weighted = (weights * nll).mean()
        
        # 2. Theta Prior (L2)
        prior_theta = reg_strength * torch.mean(self.theta[user_idx]**2)
        
        # 3. Difficulty Prior (L2 around mu_d)
        d_val = self.d[item_idx]
        prior_d = reg_strength * torch.mean((d_val - self.mu_d)**2)
        
        # 4. Weight Prior (Dirichlet via Phi)
        phi_val = self.phi[item_idx]
        term1 = torch.sum(phi_val, dim=1)
        lse = torch.logsumexp(phi_val, dim=1)
        term2 = self.k_dims * lse
        prior_phi = -reg_strength * ((self.alpha - 1) * (term1 - term2).mean())
        
        total_loss = nll_weighted + prior_theta + prior_d + prior_phi
        return total_loss, nll_weighted.item()

# ------------------------------------------------------------------------------
# Optimized Fitting Function (Entire Dataset on GPU)
# ------------------------------------------------------------------------------

def fit_lada_model_joint(resmat_values, k_dims, n_epochs, seed=None, name="model"):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    print(f"\nFitting LADA model '{name}' (JOINT) with K={k_dims}...")
    
    # 1. Prepare Data Indices (CPU side first)
    n_total_persons, n_total_items = resmat_values.shape
    observed_pairs = np.argwhere(~np.isnan(resmat_values))
    np.random.shuffle(observed_pairs)
    
    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    test_pairs = observed_pairs[:n_test]
    train_pairs = observed_pairs[n_test:]
    
    # Calculate weights on CPU before moving to GPU
    item_counts = pd.Series(train_pairs[:, 1]).value_counts().reindex(range(n_total_items), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights_cpu = inv_freq_weights.iloc[train_pairs[:, 1]].values

    print(f"  Moving entire dataset ({len(train_pairs)} train, {len(test_pairs)} test) to {device}...")

    # 2. Move EVERYTHING to GPU (The Critical Optimization)
    train_rows = torch.from_numpy(train_pairs[:, 0]).long().to(device)
    train_cols = torch.from_numpy(train_pairs[:, 1]).long().to(device)
    train_ys = torch.from_numpy(resmat_values[train_pairs[:, 0], train_pairs[:, 1]]).float().to(device)
    train_weights = torch.from_numpy(train_weights_cpu).float().to(device)
    
    test_rows = torch.from_numpy(test_pairs[:, 0]).long().to(device)
    test_cols = torch.from_numpy(test_pairs[:, 1]).long().to(device)
    # Keep test_ys on CPU for sklearn metric calculation later, but we can also put a copy on GPU if we want custom metrics
    test_ys_cpu = resmat_values[test_pairs[:, 0], test_pairs[:, 1]]
    
    # Initialize Model
    model = LADAModel(n_total_persons, n_total_items, k_dims, 
                      alpha=ALPHA_DIRICHLET, mu_d=MU_D, sigma_d=SIGMA_D).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_JOINT)
    
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    training_history = []
    
    num_train = len(train_rows)
    
    # 3. Training Loop (Manual Batching)
    for epoch in range(n_epochs):
        model.train()
        epoch_nll = 0.0
        n_batches = 0
        
        # Shuffle indices on GPU directly
        permutation = torch.randperm(num_train, device=device)
        
        # Iterate with massive strides
        for i in range(0, num_train, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            
            # Slice directly from GPU tensors (Zero copy overhead)
            batch_rows = train_rows[indices]
            batch_cols = train_cols[indices]
            batch_ys = train_ys[indices]
            batch_wts = train_weights[indices]
            
            optimizer.zero_grad()
            
            # Forward & Loss
            logits = model(batch_rows, batch_cols)
            loss, nll_val = model.compute_loss(
                logits, batch_ys, batch_wts, 
                batch_rows, batch_cols, REG_STRENGTH
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_nll += nll_val
            n_batches += 1
        
        avg_nll = epoch_nll / n_batches
        
        # Validation (Every 5 epochs to save time, or every epoch if fast enough)
        # On A100, inference is instant, so we can do it every epoch.
        model.eval()
        with torch.no_grad():
            # Process test set (Full batch inference usually fits in A100 RAM easily)
            logits_test = model(test_rows, test_cols)
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
            
            try:
                val_auc = roc_auc_score(test_ys_cpu, probs_test)
            except ValueError:
                val_auc = 0.5
        
        training_history.append({'epoch': epoch+1, 'train_loss': avg_nll, 'val_auc': val_auc})
        
        # Simple logging
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} - NLL: {avg_nll:.4f}, Val AUC: {val_auc:.4f}")

        # Early Stopping
        if val_auc > best_auc + EARLY_STOPPING_THRESHOLD:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if val_auc > best_auc: best_auc = val_auc 
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}. Best AUC: {best_auc:.4f}")
                break
    
    # Restore best state
    if best_state:
        model.load_state_dict(best_state)
    
    # Final Metrics
    final_w = model.get_w().detach().cpu()
    
    # Calculate AIC/BIC (Need full train likelihood)
    with torch.no_grad():
        # Compute in chunks if necessary, but A100 usually handles it.
        # We will use a simplified loop for memory safety on the "full likelihood" check
        full_nll_sum = 0
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        
        # Split into chunks to avoid OOM on the *backward* pass (though this is no_grad)
        # just to be safe with 10M+ points.
        chunk_size = 1000000
        for i in range(0, num_train, chunk_size):
            end = min(i + chunk_size, num_train)
            chunk_rows = train_rows[i:end]
            chunk_cols = train_cols[i:end]
            chunk_ys = train_ys[i:end]
            full_nll_sum += loss_fn(model(chunk_rows, chunk_cols), chunk_ys).item()

    num_params = sum(p.numel() for p in model.parameters())
    aic = 2 * num_params + 2 * full_nll_sum
    bic = np.log(num_train) * num_params + 2 * full_nll_sum
    
    # Save History
    history_df = pd.DataFrame(training_history)
    history_path = os.path.join(RESULT_DIR, f"training_history_{name}_k{k_dims}.csv")
    history_df.to_csv(history_path, index=False)
    
    return {
        'theta': model.theta.detach().cpu(),
        'phi': model.phi.detach().cpu(),
        'w': final_w,
        'd': model.d.detach().cpu(),
        'test_auc': best_auc, 
        'aic': aic, 
        'bic': bic, 
        'best_epoch': best_epoch,
        'log_likelihood': -full_nll_sum,
        'num_params': num_params,
        'n_train': num_train,
        'n_test': len(test_ys_cpu)
    }

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

def main(k_values, n_epochs, target_scenario=None):
    if target_scenario:
        if target_scenario not in scenarios:
            print(f"Error: Scenario '{target_scenario}' not found.")
            sys.exit(1)
        scenarios_to_run = [target_scenario]
        run_full = False
        print(f"Targeting scenario: {target_scenario}")
    else:
        scenarios_to_run = sorted(scenarios)
        run_full = True
    
    all_results = {}
    
    for K in k_values:
        print(f"\n{'='*60}\nJOINT OPTIMIZATION K={K}\n{'='*60}")
        lada_results = defaultdict(dict)
        
        # 1. Full Dataset
        if run_full:
            res = fit_lada_model_joint(resmat_values, K, n_epochs, seed=FULL_SEED, name="combined_data")
            lada_results["combined_data"] = res
            
            torch.save({
                'theta': res['theta'],
                'd': res['d'], 
                'phi': res['phi'],
                'metrics': {k:v for k,v in res.items() if not torch.is_tensor(v)}
            }, os.path.join(RESULT_DIR, f"lada_joint_k{K}_combined_data.pt"))

        # 2. Per Scenario
        for idx, scenario in enumerate(scenarios_to_run):
            mask = (results.columns.get_level_values("scenario") == scenario)
            data = resmat_values[:, mask]
            if (~np.isnan(data)).sum() < 100: 
                print(f"Skipping {scenario} (insufficient data)")
                continue
            
            print(f"Processing {scenario}...")
            res = fit_lada_model_joint(data, K, n_epochs, seed=idx, name=scenario)
            lada_results[scenario] = res
            
            torch.save({
                'theta': res['theta'],
                'd': res['d'], 
                'phi': res['phi'],
                'metrics': {k:v for k,v in res.items() if not torch.is_tensor(v)}
            }, os.path.join(RESULT_DIR, f"lada_joint_k{K}_{scenario}.pt"))

        # 3. Save Summaries
        summary_data = []
        for name, res in lada_results.items():
            summary_data.append({
                'scenario': name, 'K': K, 
                'test_auc': res['test_auc'], 
                'aic': res['aic'], 
                'bic': res['bic'],
                'best_epoch': res['best_epoch'],
                'log_likelihood': res['log_likelihood']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data).sort_values('test_auc', ascending=False)
            print(f"\nK={K} Summary:\n{df.to_string(index=False)}")
            
            suffix = f"_{target_scenario}" if target_scenario else ""
            df.to_csv(os.path.join(RESULT_DIR, f"lada_joint_k{K}_summary{suffix}.csv"), index=False)
            
            # Save Complete JSON
            json_res = {}
            for name, res in lada_results.items():
                json_res[name] = {k: v for k, v in res.items() if not torch.is_tensor(v)}
                json_res[name]['theta_shape'] = list(res['theta'].shape)
            
            with open(os.path.join(RESULT_DIR, f"lada_joint_k{K}_complete_results{suffix}.json"), 'w') as f:
                json.dump(json_res, f, indent=2)
            
            if not df.empty:
                all_results[f"K_{K}"] = {'best_auc': df.iloc[0]['test_auc'], 'best_model': df.iloc[0]['scenario']}

    print("\nFinal Best Results per K:")
    overall_df = pd.DataFrame([{'K': k, 'AUC': v['best_auc'], 'Model': v['best_model']} for k,v in all_results.items()])
    print(overall_df)
    
    suffix = f"_{target_scenario}" if target_scenario else ""
    overall_df.to_csv(os.path.join(RESULT_DIR, f"lada_overall_summary{suffix}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-values', nargs='+', type=int, default=DEFAULT_K_VALUES,
                        help='List of K values to test')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--scenario', '-s', type=str, default=None, 
                        help="Specific scenario to fit")
    
    args = parser.parse_args()
    main(args.k_values, args.epochs, args.scenario)