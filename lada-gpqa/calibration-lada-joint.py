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
DEFAULT_K_VALUES = [2, 3, 4, 5, 10, 20, 30]  
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

RESULT_DIR = "../result/lada-fitting-joint-float"
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
    with open("../data-reeval-multi/gpqa_diamond/resmat.pkl", "rb") as f:
        results = pickle.load(f)
except FileNotFoundError:
    print("Error: Data file '../data-reeval-multi/gpqa_diamond/resmat.pkl' not found.")
    sys.exit(1)

resmat_values = results.values
n_persons, n_items = resmat_values.shape

# GPQA Diamond has simple column structure, not multi-level with scenarios
# We'll treat it as a single dataset
scenarios = []  # No scenarios for GPQA Diamond

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
        # 1. Mean Squared Error for continuous targets
        predictions = torch.sigmoid(logits)
        mse = F.mse_loss(predictions, targets, reduction='none')
        weighted_mse = (weights * mse).mean()
        
        # 2-4. Keep existing priors
        prior_theta = reg_strength * torch.mean(self.theta[user_idx]**2)
        d_val = self.d[item_idx]
        prior_d = reg_strength * torch.mean((d_val - self.mu_d)**2)
        phi_val = self.phi[item_idx]
        term1 = torch.sum(phi_val, dim=1)
        lse = torch.logsumexp(phi_val, dim=1)
        term2 = self.k_dims * lse
        prior_phi = -reg_strength * ((self.alpha - 1) * (term1 - term2).mean())
        
        total_loss = weighted_mse + prior_theta + prior_d + prior_phi
        return total_loss, weighted_mse.item()

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
    
    best_mse = np.inf  # Lower is better for MSE
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
        
        # Validation - using MSE and correlation for continuous targets
        model.eval()
        with torch.no_grad():
            # Process test set (Full batch inference usually fits in A100 RAM easily)
            logits_test = model(test_rows, test_cols)
            preds_test = torch.sigmoid(logits_test).cpu().numpy()
            
            # MSE for continuous targets
            val_mse = np.mean((preds_test - test_ys_cpu) ** 2)
            
            # Pearson correlation as additional metric
            val_corr = np.corrcoef(preds_test, test_ys_cpu)[0, 1]
            if np.isnan(val_corr):
                val_corr = 0.0
        
        training_history.append({
            'epoch': epoch+1, 
            'train_loss': avg_nll, 
            'val_mse': val_mse,
            'val_corr': val_corr
        })
        
        # Simple logging
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_nll:.4f}, Val MSE: {val_mse:.4f}, Val Corr: {val_corr:.4f}")

        # Early Stopping (lower MSE is better)
        if val_mse < best_mse - EARLY_STOPPING_THRESHOLD:
            best_mse = val_mse
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if val_mse < best_mse: best_mse = val_mse 
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}. Best MSE: {best_mse:.4f}")
                break
    
    # Restore best state
    if best_state:
        model.load_state_dict(best_state)
    
    # Final Metrics
    final_w = model.get_w().detach().cpu()
    
    # Calculate AIC/BIC using MSE for continuous targets
    with torch.no_grad():
        # Compute in chunks to avoid OOM
        full_mse_sum = 0
        
        chunk_size = 1000000
        for i in range(0, num_train, chunk_size):
            end = min(i + chunk_size, num_train)
            chunk_rows = train_rows[i:end]
            chunk_cols = train_cols[i:end]
            chunk_ys = train_ys[i:end]
            
            chunk_preds = torch.sigmoid(model(chunk_rows, chunk_cols))
            chunk_mse = F.mse_loss(chunk_preds, chunk_ys, reduction='sum')
            full_mse_sum += chunk_mse.item()

    num_params = sum(p.numel() for p in model.parameters())
    # For MSE, we use -2*log(L) where L is Gaussian likelihood
    # -2*log(L) ≈ n*log(MSE) for large n (simplified)
    neg_2_log_likelihood = num_train * np.log(full_mse_sum / num_train + 1e-10)
    aic = 2 * num_params + neg_2_log_likelihood
    bic = np.log(num_train) * num_params + neg_2_log_likelihood
    
    # Save History
    history_df = pd.DataFrame(training_history)
    history_path = os.path.join(RESULT_DIR, f"training_history_{name}_k{k_dims}.csv")
    history_df.to_csv(history_path, index=False)
    
    return {
        'theta': model.theta.detach().cpu(),
        'phi': model.phi.detach().cpu(),
        'w': final_w,
        'd': model.d.detach().cpu(),
        'test_mse': best_mse, 
        'aic': aic, 
        'bic': bic, 
        'best_epoch': best_epoch,
        'train_mse': full_mse_sum / num_train,
        'num_params': num_params,
        'n_train': num_train,
        'n_test': len(test_ys_cpu)
    }

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

def main(k_values, n_epochs, target_scenario=None):
    # GPQA Diamond doesn't have scenarios - just run on full dataset
    if target_scenario:
        print(f"Warning: GPQA Diamond doesn't have scenarios. Ignoring scenario parameter.")
    
    all_results = {}
    
    for K in k_values:
        print(f"\n{'='*60}\nJOINT OPTIMIZATION K={K}\n{'='*60}")
        lada_results = defaultdict(dict)
        
        # Single dataset (no scenarios in GPQA Diamond)
        res = fit_lada_model_joint(resmat_values, K, n_epochs, seed=FULL_SEED, name="gpqa_diamond")
        lada_results["gpqa_diamond"] = res
        
        torch.save({
            'theta': res['theta'],
            'd': res['d'], 
            'phi': res['phi'],
            'metrics': {k:v for k,v in res.items() if not torch.is_tensor(v)}
        }, os.path.join(RESULT_DIR, f"lada_joint_k{K}_gpqa_diamond.pt"))

        # 3. Save Summaries
        summary_data = []
        for name, res in lada_results.items():
            summary_data.append({
                'dataset': name, 'K': K, 
                'test_mse': res['test_mse'],
                'train_mse': res['train_mse'],
                'aic': res['aic'], 
                'bic': res['bic'],
                'best_epoch': res['best_epoch']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data).sort_values('test_mse', ascending=True)  # Lower MSE is better
            print(f"\nK={K} Summary:\n{df.to_string(index=False)}")
            
            df.to_csv(os.path.join(RESULT_DIR, f"lada_joint_k{K}_summary.csv"), index=False)
            
            # Save Complete JSON
            json_res = {}
            for name, res in lada_results.items():
                json_res[name] = {k: v for k, v in res.items() if not torch.is_tensor(v)}
                json_res[name]['theta_shape'] = list(res['theta'].shape)
            
            with open(os.path.join(RESULT_DIR, f"lada_joint_k{K}_complete_results.json"), 'w') as f:
                json.dump(json_res, f, indent=2)
            
            if not df.empty:
                all_results[f"K_{K}"] = {'best_mse': df.iloc[0]['test_mse'], 'best_model': df.iloc[0]['dataset']}

    print("\nFinal Best Results per K:")
    overall_df = pd.DataFrame([{'K': k, 'MSE': v['best_mse'], 'Model': v['best_model']} for k,v in all_results.items()])
    print(overall_df)
    
    overall_df.to_csv(os.path.join(RESULT_DIR, f"lada_overall_summary.csv"), index=False)

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