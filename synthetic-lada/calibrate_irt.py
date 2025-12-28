"""
Calibrate IRT (Rasch) model on synthetic train data and evaluate on synthetic test data.
This provides a baseline to compare against LADA models.
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from torch.distributions import Bernoulli
from torch.optim import LBFGS
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
EPOCHS_Z = 100
EPOCHS_THETA = 100
SEED = 42

RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------------------

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter), desc="Training") if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": f"{grad_norm:.4f}", "d_param": f"{d_parameters:.4f}", "d_loss": f"{d_loss:.4f}"})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
    return parameters

# ------------------------------------------------------------------------------
# Rasch Model Fitting
# ------------------------------------------------------------------------------

def fit_rasch_model(train_values, test_values, seed=None):
    """
    Fit Rasch (1PL IRT) model on training data and evaluate on test data.
    
    Model: P(correct) = sigmoid(theta + z)
    where theta is person ability and z is item difficulty
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print("\nFitting Rasch (1PL IRT) model...")
    
    n_persons, n_items_train = train_values.shape
    _, n_items_test = test_values.shape
    
    # Prepare training data
    train_tensor = torch.from_numpy(train_values).float().to(device)
    train_mask = ~torch.isnan(train_tensor)
    train_tensor = torch.nan_to_num(train_tensor, nan=0.0)
    
    print(f"  Train data: {n_persons} persons × {n_items_train} items")
    print(f"  Test data: {n_persons} persons × {n_items_test} items")
    
    # Step 1: Fit item difficulties (z) using nuisance person abilities
    print("  Step 1: Fitting item difficulties...")
    thetas_nuisance = torch.randn(n_persons, device=device)
    z = torch.randn(n_items_train, requires_grad=True, device=device)
    
    optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    def closure_z():
        optim_z.zero_grad()
        probs = torch.sigmoid(thetas_nuisance[:, None] + z[None, :])
        loss = -(Bernoulli(probs=probs).log_prob(train_tensor) * train_mask).mean()
        loss.backward()
        return loss
    
    z = trainer([z], optim_z, closure_z, n_iter=EPOCHS_Z, verbose=True)[0]
    print(f"    z statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
    
    # Step 2: Fit person abilities (theta) with fixed item difficulties
    print("  Step 2: Fitting person abilities...")
    theta = torch.randn(n_persons, requires_grad=True, device=device)
    
    optim_theta = LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    def closure_theta():
        optim_theta.zero_grad()
        probs = torch.sigmoid(theta[:, None] + z[None, :])
        loss = -(Bernoulli(probs=probs).log_prob(train_tensor) * train_mask).mean()
        loss.backward()
        return loss
    
    theta = trainer([theta], optim_theta, closure_theta, n_iter=EPOCHS_THETA, verbose=True)[0]
    print(f"    theta statistics: mean={theta.mean().item():.4f}, std={theta.std().item():.4f}")
    
    # Evaluate on training data
    with torch.no_grad():
        train_probs = torch.sigmoid(theta[:, None] + z[None, :])
        train_probs_flat = train_probs[train_mask].cpu().numpy()
        train_labels_flat = train_tensor[train_mask].cpu().numpy()
        train_auc = roc_auc_score(train_labels_flat, train_probs_flat)
        train_preds = (train_probs_flat > 0.5).astype(int)
        train_acc = accuracy_score(train_labels_flat, train_preds)
    
    print(f"\n  Train AUC: {train_auc:.4f}")
    print(f"  Train Accuracy: {train_acc:.4f}")
    
    # Save model
    model_dict = {
        'theta': theta.detach().cpu().numpy(),
        'z': z.detach().cpu().numpy(),
        'train_auc': train_auc,
        'train_acc': train_acc
    }
    
    model_path = os.path.join(RESULT_DIR, "irt_rasch_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"  Saved model to {model_path}")
    
    return {
        'train_auc': train_auc,
        'train_acc': train_acc,
        'theta': theta.detach().cpu().numpy(),
        'z': z.detach().cpu().numpy()
    }

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("Rasch (1PL IRT) Calibration on Synthetic Data")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading synthetic data...")
    try:
        train_df = pd.read_pickle("./data/resmat_train.pkl")
        test_df = pd.read_pickle("./data/resmat_test.pkl")
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
    except FileNotFoundError:
        print("Error: Data files not found. Please run gen_syn.py first.")
        return
    
    train_values = train_df.values
    test_values = test_df.values
    
    # Fit model
    result = fit_rasch_model(train_values, test_values, seed=SEED)
    
    # Compare with ground truth
    try:
        ground_truth = np.load('./data/ground_truth.npy', allow_pickle=True).item()
        true_theta = ground_truth['theta'][:, 0]  # Use first dimension since Rasch is 1D
        estimated_theta = result['theta']
        
        print("\n" + "="*60)
        print("Comparison with Ground Truth:")
        print("="*60)
        
        # Compute correlation (accounting for potential sign flip)
        corr = np.corrcoef(true_theta, estimated_theta)[0, 1]
        corr_abs = abs(corr)
        print(f"Theta correlation: {corr:.4f} (absolute: {corr_abs:.4f})")
        
    except Exception as e:
        print(f"\nCould not load ground truth for comparison: {e}")
    
    print("\nCalibration complete!")

if __name__ == "__main__":
    main()
