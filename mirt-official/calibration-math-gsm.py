#!/usr/bin/env python3
"""
MIRT Calibration Script - Math+GSM Composite Dataset

Multidimensional Item Response Theory (MIRT) fitting for combined math and gsm scenarios 
with different numbers of dimensions (K=1 to 10).

Usage:
    python calibration-math-gsm.py
    
The script will automatically run MIRT fitting for K=[1,2,3,4,5,6,7,8,9,10] dimensions.
"""

# Import and Setup
import sys
import argparse
import torch
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
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Different numbers of dimensions to test
N_EPOCHS = 1000
BATCH_SIZE = 50000
lr = 0.01

reg_strength = 0.01
RESULT_DIR = "../result/mirt-fitting"
os.makedirs(RESULT_DIR, exist_ok=True)

# For reproducibility with rep87 (seed 86)
COMPOSITE_SEED = 86

EARLY_STOPPING_THRESHOLD = 1e-4

print(f"MIRT with K values: {K_VALUES}")


# Data Loading
print("Loading data...")
with open("../data/resmat.pkl", "rb") as f:
    results = pickle.load(f)

print(f"Data shape: {results.shape}")
print(f"Number of scenarios: {results.columns.get_level_values('scenario').nunique()}")

# Extract and combine math and gsm scenarios
print("Extracting math and gsm scenarios...")
scenarios = results.columns.get_level_values("scenario").unique()
print(f"Available scenarios: {sorted(scenarios)}")

# Check if math and gsm scenarios exist
target_scenarios = ['math', 'gsm']
missing_scenarios = [s for s in target_scenarios if s not in scenarios]
if missing_scenarios:
    print(f"Warning: Missing scenarios: {missing_scenarios}")
    print(f"Available scenarios: {sorted(scenarios)}")
    # Use available scenarios that contain the target names
    available_math = [s for s in scenarios if 'math' in s.lower()]
    available_gsm = [s for s in scenarios if 'gsm' in s.lower()]
    if available_math:
        target_scenarios = available_math[:1]  # Take first match
    if available_gsm:
        target_scenarios.extend(available_gsm[:1])  # Take first match
    print(f"Using scenarios: {target_scenarios}")

# Extract data for math and gsm scenarios
math_gsm_mask = results.columns.get_level_values("scenario").isin(target_scenarios)
math_gsm_data = results.loc[:, math_gsm_mask]

print(f"Math+GSM composite data shape: {math_gsm_data.shape}")
print(f"Total observations in composite dataset: {(~math_gsm_data.isna()).sum().sum()}")

# Work directly with numpy values - much faster!
resmat_values = math_gsm_data.values  # Keep as numpy array
n_persons, n_items = resmat_values.shape

print(f"n_persons: {n_persons}, n_items: {n_items}")
print(f"Total observations: {(~np.isnan(resmat_values)).sum()}")

# MIRT Fitting Functions
def fit_mirt_model(resmat_values, k_dims, seed=None, name="model"):
    """
    Fit MIRT model with k dimensions - optimized for speed like original k-trials.
    
    Args:
        resmat_values: numpy array of shape (n_persons, n_items) with response data
        k_dims: number of latent dimensions
        seed: random seed for reproducibility
        name: name for logging
    
    Returns:
        Dictionary with fitted parameters and evaluation metrics
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    print(f"\nFitting MIRT model '{name}' with K={k_dims}...")
    
    # Work directly with numpy like the original - much faster!
    observed_pairs = np.argwhere(~np.isnan(resmat_values))
    np.random.shuffle(observed_pairs)
    
    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    test_pairs = observed_pairs[:n_test]
    train_pairs = observed_pairs[n_test:]
    
    # Prepare data efficiently
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat_values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat_values[test_rows, test_cols]
    
    # Convert to tensors once
    train_rows_t = torch.from_numpy(train_rows)
    train_cols_t = torch.from_numpy(train_cols)
    train_ys_t = torch.from_numpy(train_ys).float()
    
    print(f"  Train observations: {len(train_ys)}, Test observations: {len(test_ys)}")
    
    # Compute per-item weights for balancing (same as original)
    n_persons, n_items = resmat_values.shape
    item_counts = pd.Series(train_cols).value_counts().reindex(range(n_items), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights = inv_freq_weights.iloc[train_cols].values
    train_weights_t = torch.from_numpy(train_weights).float()
    
    # Initialize model parameters
    theta = torch.randn(n_persons, k_dims, device=device, requires_grad=True)
    a = torch.randn(n_items, k_dims, device=device, requires_grad=True)
    b = torch.randn(n_items, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([theta, a, b], lr=lr)
    
    # Create data loader with same settings as original
    train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t, train_weights_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=40,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Early stopping setup
    patience = 5
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    
    # Training history tracking
    training_history = []
    
    # Training loop
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{N_EPOCHS}", leave=False)
        
        for batch_rows, batch_cols, batch_ys, batch_wts in pbar:
            # Move batch to device
            batch_rows = batch_rows.to(device)
            batch_cols = batch_cols.to(device)
            batch_ys = batch_ys.to(device)
            batch_wts = batch_wts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            theta_r = theta[batch_rows]
            a_c = a[batch_cols]
            b_c = b[batch_cols]
            dot = torch.sum(theta_r * a_c, axis=1)
            logits = dot - b_c
            
            # Loss computation
            loss_per_obs = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            weighted_loss = (batch_wts * loss_per_obs).sum()
            l2_reg = reg_strength * (torch.sum(theta**2) + torch.sum(a**2) + torch.sum(b**2))
            loss = (weighted_loss + l2_reg) / len(batch_ys)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_rows)
            pbar.set_postfix({'loss': loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataset)
        
        # Validation
        with torch.no_grad():
            theta_test = theta[test_rows]
            a_test = a[test_cols]
            b_test = b[test_cols]
            logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
            val_auc = roc_auc_score(test_ys, probs_test)
        
        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_epoch_loss,
            'val_auc': val_auc
        })
        
        print(f"    Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_epoch_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping with threshold - corrected logic
        if val_auc > best_auc + EARLY_STOPPING_THRESHOLD:
            # Significant improvement found
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                'theta': theta.detach().clone(),
                'a': a.detach().clone(),
                'b': b.detach().clone()
            }
            epochs_no_improve = 0
        else:
            # No significant improvement
            epochs_no_improve += 1
            # Still update best_auc if this is better (even if not significant)
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                best_state = {
                    'theta': theta.detach().clone(),
                    'a': a.detach().clone(),
                    'b': b.detach().clone()
                }
            
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}. Best Val AUC: {best_auc:.4f} (epoch {best_epoch})")
                break
    
    # Use best model
    if best_state is None:
        print("    Warning: Model did not improve. Using last state.")
        best_state = {
            'theta': theta.detach().clone(),
            'a': a.detach().clone(),
            'b': b.detach().clone()
        }
        best_epoch = epoch + 1
    
    theta, a, b = best_state['theta'], best_state['a'], best_state['b']
    
    # Final evaluation with best model
    with torch.no_grad():
        theta_test = theta[test_rows]
        a_test = a[test_cols]
        b_test = b[test_cols]
        logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
    
    test_auc = roc_auc_score(test_ys, probs_test)
    print(f"  Final Test AUC: {test_auc:.4f} (using best model from epoch {best_epoch})")
    
    # Calculate log-likelihood and model fit metrics
    with torch.no_grad():
        train_rows_gpu = train_rows_t.to(device)
        train_cols_gpu = train_cols_t.to(device)
        train_ys_gpu = train_ys_t.to(device)
        
        final_logits = torch.sum(theta[train_rows_gpu] * a[train_cols_gpu], axis=1) - b[train_cols_gpu]
        final_log_likelihood = -F.binary_cross_entropy_with_logits(
            final_logits, train_ys_gpu, reduction='sum').item()
    
    num_params = theta.numel() + a.numel() + b.numel()
    aic = 2 * num_params - 2 * final_log_likelihood
    bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood
    
    # Save training history
    history_path = os.path.join(RESULT_DIR, f"training_history_{name}_k{k_dims}.csv")
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(history_path, index=False)
    print(f"  Training history saved to: {history_path}")
    
    return {
        'theta': theta.cpu(),
        'a': a.cpu(), 
        'b': b.cpu(),
        'test_auc': test_auc,
        'best_val_auc': best_auc,
        'best_epoch': best_epoch,
        'total_epochs': len(training_history),
        'log_likelihood': final_log_likelihood,
        'aic': aic,
        'bic': bic,
        'num_params': num_params,
        'n_train': len(train_ys),
        'n_test': len(test_ys),
        'training_history': training_history
    }


def main():
    """Main function to run MIRT fitting for all K values on math+gsm composite dataset."""
    print("Starting MIRT calibration for math+gsm composite dataset...")
    
    # Store all results across K values
    all_results = {}
    
    # Loop through different K values
    for K in K_VALUES:
        print(f"\n{'='*80}")
        print(f"PROCESSING K = {K} DIMENSIONS (Math+GSM Composite)")
        print(f"{'='*80}")
        
        # MIRT Fitting - Math+GSM Composite Dataset
        print(f"\nFitting MIRT on math+gsm composite dataset with K={K}...")
        composite_result = fit_mirt_model(
            resmat_values, 
            K, 
            seed=COMPOSITE_SEED,
            name="math_gsm_composite"
        )

        # Store results
        print(f"\nMath+GSM composite dataset MIRT fitting completed!")
        print(f"Test AUC: {composite_result['test_auc']:.4f}")
        print(f"AIC: {composite_result['aic']:.2f}")
        print(f"BIC: {composite_result['bic']:.2f}")

        # Save composite model
        composite_model_path = os.path.join(RESULT_DIR, f"mirt_model_k{K}_math_gsm_composite.pt")
        torch.save({
            'theta': composite_result['theta'],
            'a': composite_result['a'], 
            'b': composite_result['b'],
            'K': K,
            'seed': COMPOSITE_SEED,
            'scenarios': target_scenarios,
            'metrics': {
                'test_auc': composite_result['test_auc'],
                'log_likelihood': composite_result['log_likelihood'],
                'aic': composite_result['aic'],
                'bic': composite_result['bic'],
                'num_params': composite_result['num_params']
            }
        }, composite_model_path)
        print(f"Composite model saved to: {composite_model_path}")

        # Results Export and Summary for this K
        results_summary = [{
            'scenario': 'math_gsm_composite',
            'K': K,
            'test_auc': composite_result['test_auc'],
            'best_val_auc': composite_result['best_val_auc'],
            'best_epoch': composite_result['best_epoch'],
            'total_epochs': composite_result['total_epochs'],
            'log_likelihood': composite_result['log_likelihood'], 
            'aic': composite_result['aic'],
            'bic': composite_result['bic'],
            'num_params': composite_result['num_params'],
            'n_train': composite_result['n_train'],
            'n_test': composite_result['n_test']
        }]

        # Create summary DataFrame
        summary_df = pd.DataFrame(results_summary)

        print(f"\nMIRT Fitting Results Summary for K={K} (Math+GSM Composite):")
        print("=" * 100)
        print(summary_df.to_string(index=False, float_format='%.4f'))

        # Save summary to CSV
        summary_path = os.path.join(RESULT_DIR, f"mirt_k{K}_math_gsm_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

        # Save complete results as JSON
        results_path = os.path.join(RESULT_DIR, f"mirt_k{K}_math_gsm_complete_results.json") 
        # Convert tensors to lists for JSON serialization
        json_results = {
            'math_gsm_composite': {
                'test_auc': composite_result['test_auc'],
                'best_val_auc': composite_result['best_val_auc'],
                'best_epoch': composite_result['best_epoch'],
                'total_epochs': composite_result['total_epochs'],
                'log_likelihood': composite_result['log_likelihood'],
                'aic': composite_result['aic'], 
                'bic': composite_result['bic'],
                'num_params': composite_result['num_params'],
                'n_train': composite_result['n_train'],
                'n_test': composite_result['n_test'],
                'theta_shape': list(composite_result['theta'].shape),
                'a_shape': list(composite_result['a'].shape),
                'b_shape': list(composite_result['b'].shape),
                'scenarios_used': target_scenarios
            }
        }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Complete results saved to: {results_path}")

        print(f"\nMath+GSM MIRT K={K} model and results exported to: {RESULT_DIR}")
        print(f"Training history saved as: training_history_math_gsm_composite_k{K}.csv")
        
        # Store results for this K in overall results
        all_results[f"K_{K}"] = {
            'summary_df': summary_df,
            'composite_result': composite_result,
            'test_auc': composite_result['test_auc']
        }
    
    # Final summary across all K values
    print(f"\n{'='*80}")
    print("FINAL SUMMARY ACROSS ALL K VALUES (Math+GSM Composite)")
    print(f"{'='*80}")
    
    overall_summary = []
    for k_name, k_results in all_results.items():
        K_val = int(k_name.split('_')[1])
        overall_summary.append({
            'K': K_val,
            'test_auc': k_results['test_auc'],
            'best_val_auc': k_results['composite_result']['best_val_auc'],
            'aic': k_results['composite_result']['aic'],
            'bic': k_results['composite_result']['bic'],
            'num_params': k_results['composite_result']['num_params']
        })
    
    overall_df = pd.DataFrame(overall_summary)
    overall_df = overall_df.sort_values('test_auc', ascending=False)
    
    print("\nOverall Results by K (Math+GSM Composite):")
    print(overall_df.to_string(index=False, float_format='%.4f'))
    
    # Save overall summary
    overall_path = os.path.join(RESULT_DIR, "mirt_math_gsm_overall_summary.csv")
    overall_df.to_csv(overall_path, index=False)
    print(f"\nOverall summary saved to: {overall_path}")
    
    print(f"\nBest overall result: K={overall_df.iloc[0]['K']} with AUC={overall_df.iloc[0]['test_auc']:.4f}")
    print(f"Scenarios used: {target_scenarios}")
    print("Math+GSM MIRT calibration completed successfully!")


if __name__ == "__main__":
    # Optional: Add command line argument parsing for custom K values
    parser = argparse.ArgumentParser(description='MIRT Calibration for Math+GSM composite dataset')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='List of K values to test (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    
    args = parser.parse_args()
    
    # Update global variables based on arguments
    K_VALUES = args.k_values
    N_EPOCHS = args.epochs
    
    print(f"Starting Math+GSM MIRT calibration with K values: {K_VALUES}, Epochs: {N_EPOCHS}")
    main()
