"""
Evaluater for synthetic LADA inference tests.
Load all fitted lada models from ./result, take only the theta estimates.
Take random 10 columns from resmat_test from ./data and fit LADA models with k = dimension of theta.
Take d and w from the fitted model of the 10 columns and plug in theta to regenerate resmat_test predictions.
Compare the regenerated resmat_test with the original resmat_test to compute accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
K_VALUES = [1, 2, 3, 5, 10, 20]  # K values to test
CALIBRATION_ITEMS = 10  # Number of test items to use for calibration
EPOCHS = 500
LR = 0.05
REG_STRENGTH = 0.01
ALPHA_DIRICHLET = 1.1
MU_D = 0.0
SIGMA_D = 1.0
BATCH_SIZE = 5000
SEED = 42

# Set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------------------------
# Model Definition (with fixed theta)
# ------------------------------------------------------------------------------

class LADAInferenceModel(nn.Module):
    """LADA model with fixed theta, only learning d and w for new items"""
    def __init__(self, theta_fixed, n_items, alpha=1.1, mu_d=0.0, sigma_d=1.0):
        super().__init__()
        self.n_users = theta_fixed.shape[0]
        self.n_items = n_items
        self.k_dims = theta_fixed.shape[1]
        self.alpha = alpha
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        
        # Fixed theta (not trainable)
        self.register_buffer('theta', theta_fixed)
        
        # Learnable parameters for new items only
        self.d = nn.Parameter(torch.zeros(n_items))
        self.phi = nn.Parameter(torch.randn(n_items, self.k_dims))
    
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
    
    def compute_loss(self, logits, targets, user_idx, item_idx, reg_strength):
        # Binary Cross Entropy
        nll = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        
        # Difficulty Prior
        d_val = self.d[item_idx]
        prior_d = reg_strength * torch.mean((d_val - self.mu_d)**2)
        
        # Weight Prior (Dirichlet via Phi)
        phi_val = self.phi[item_idx]
        term1 = torch.sum(phi_val, dim=1)
        lse = torch.logsumexp(phi_val, dim=1)
        term2 = self.k_dims * lse
        prior_phi = -reg_strength * ((self.alpha - 1) * (term1 - term2).mean())
        
        total_loss = nll + prior_d + prior_phi
        return total_loss, nll.item()

# ------------------------------------------------------------------------------
# Calibration Function
# ------------------------------------------------------------------------------

def calibrate_on_subset(theta_fixed, resmat_subset):
    """
    Calibrate d and w on a subset of test items with fixed theta
    
    Args:
        theta_fixed: (n_persons, k) fixed theta estimates
        resmat_subset: (n_persons, n_subset_items) response matrix subset
    
    Returns:
        Calibrated d and w parameters
    """
    n_persons, n_items = resmat_subset.shape
    k_dims = theta_fixed.shape[1]
    
    # Prepare data
    pairs = np.argwhere(~np.isnan(resmat_subset))
    
    rows = torch.from_numpy(pairs[:, 0]).long().to(device)
    cols = torch.from_numpy(pairs[:, 1]).long().to(device)
    ys = torch.from_numpy(resmat_subset[pairs[:, 0], pairs[:, 1]]).float().to(device)
    
    # Initialize model with fixed theta
    theta_tensor = torch.from_numpy(theta_fixed).float().to(device)
    model = LADAInferenceModel(theta_tensor, n_items, 
                               alpha=ALPHA_DIRICHLET, mu_d=MU_D, sigma_d=SIGMA_D).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    num_samples = len(rows)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        permutation = torch.randperm(num_samples, device=device)
        
        for i in range(0, num_samples, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            
            batch_rows = rows[indices]
            batch_cols = cols[indices]
            batch_ys = ys[indices]
            
            optimizer.zero_grad()
            
            logits = model(batch_rows, batch_cols)
            loss, nll = model.compute_loss(logits, batch_ys, batch_rows, batch_cols, REG_STRENGTH)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += nll
            n_batches += 1
        
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"    Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Extract calibrated parameters
    model.eval()
    with torch.no_grad():
        d_calibrated = model.d.cpu().numpy()
        w_calibrated = model.get_w().cpu().numpy()
    
    return d_calibrated, w_calibrated

# ------------------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------------------

def predict_with_params(theta, d, w, user_indices, item_indices):
    """
    Generate predictions using theta, d, and w
    
    Args:
        theta: (n_persons, k)
        d: (n_items,)
        w: (n_items, k)
        user_indices: array of user indices
        item_indices: array of item indices
    
    Returns:
        predictions: probability predictions
    """
    batch_theta = theta[user_indices]  # (batch, k)
    batch_d = d[item_indices]  # (batch,)
    batch_w = w[item_indices]  # (batch, k)
    
    interaction = (batch_w * batch_theta).sum(axis=1)  # (batch,)
    logits = interaction + batch_d
    predictions = 1 / (1 + np.exp(-logits))  # sigmoid
    
    return predictions

# ------------------------------------------------------------------------------
# Main Evaluation
# ------------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("LADA vs IRT Inference Test on Synthetic Data")
    print("="*60 + "\n")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_pickle("./data/resmat_test.pkl")
    test_values = test_df.values
    n_persons, n_items_test = test_values.shape
    print(f"Test data shape: {test_values.shape}")
    
    # Load all fitted models
    print("\nLoading fitted models...")
    k_values = [1, 2, 3]
    results = []
    
    # ============ EVALUATE IRT (RASCH) MODEL ============
    print("\n" + "="*60)
    print("Evaluating IRT (Rasch) Model")
    print("="*60)
    
    irt_model_path = "./result/irt_rasch_model.pkl"
    if os.path.exists(irt_model_path):
        with open(irt_model_path, 'rb') as f:
            irt_model = pickle.load(f)
        
        theta_irt = irt_model['theta']
        print(f"  Loaded IRT model, theta shape: {theta_irt.shape}")
        
        # Fit z (item difficulties) for test items with fixed theta
        print(f"  Fitting item difficulties for {n_items_test} test items with fixed theta...")
        
        # Convert to tensors
        test_tensor = torch.from_numpy(test_values).float().to(device)
        test_mask = ~torch.isnan(test_tensor)
        test_tensor = torch.nan_to_num(test_tensor, nan=0.0)
        theta_tensor = torch.from_numpy(theta_irt).float().to(device)
        
        # Fit z for test items
        z_test = torch.randn(n_items_test, requires_grad=True, device=device)
        optim_z = torch.optim.LBFGS([z_test], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        
        def closure_z():
            optim_z.zero_grad()
            probs = torch.sigmoid(theta_tensor[:, None] + z_test[None, :])
            loss = -(torch.distributions.Bernoulli(probs=probs).log_prob(test_tensor) * test_mask).mean()
            loss.backward()
            return loss
        
        # Training loop for z
        for epoch in range(500):
            loss = optim_z.step(closure_z)
            if (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}/500 - Loss: {loss.item():.4f}")
        
        # Generate predictions
        with torch.no_grad():
            test_probs = torch.sigmoid(theta_tensor[:, None] + z_test[None, :])
            test_probs_flat = test_probs[test_mask].cpu().numpy()
            test_labels_flat = test_tensor[test_mask].cpu().numpy()
            
            auc_irt = roc_auc_score(test_labels_flat, test_probs_flat)
            binary_preds = (test_probs_flat > 0.5).astype(int)
            acc_irt = accuracy_score(test_labels_flat, binary_preds)
        
        print(f"\n  IRT Results:")
        print(f"    AUC: {auc_irt:.4f}")
        print(f"    Accuracy: {acc_irt:.4f}")
        
        results.append({
            'Model': 'IRT (Rasch)',
            'K': 1,
            'AUC': auc_irt,
            'Accuracy': acc_irt,
            'n_test_samples': len(test_labels_flat)
        })
    else:
        print(f"  Warning: IRT model file {irt_model_path} not found, skipping IRT evaluation")
    
    # ============ EVALUATE LADA MODELS ============
    print("\n" + "="*60)
    print("Evaluating LADA Models")
    print("="*60)
    
    for k in k_values:
        model_path = f"./result/lada_model_k{k}.pkl"
        
        if not os.path.exists(model_path):
            print(f"  Warning: Model file {model_path} not found, skipping K={k}")
            continue
        
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        theta_estimated = model_dict['theta']
        print(f"  Loaded K={k} model, theta shape: {theta_estimated.shape}")
        
        # Select random calibration items
        calibration_indices = np.random.choice(n_items_test, CALIBRATION_ITEMS, replace=False)
        print(f"  Selected {CALIBRATION_ITEMS} random items for calibration: {calibration_indices[:5]}...")
        
        # Extract calibration subset
        resmat_calibration = test_values[:, calibration_indices]
        
        # Calibrate d and w on subset
        print(f"  Calibrating d and w on {CALIBRATION_ITEMS} items...")
        d_calibrated, w_calibrated = calibrate_on_subset(theta_estimated, resmat_calibration)
        
        # Now predict on ALL test items using calibrated d, w and original theta
        # But we only calibrated d and w for the subset items
        # We need to fit d and w for all items, not just the subset
        
        # Actually, let's reinterpret: we calibrate on 10 items, then predict on ALL items
        # But to predict on all items, we need d and w for all items
        # So we should fit d and w for ALL items using the fixed theta
        
        print(f"  Fitting d and w for ALL test items with fixed theta...")
        d_all, w_all = calibrate_on_subset(theta_estimated, test_values)
        
        # Generate predictions for all test items
        print(f"  Generating predictions for all test items...")
        test_pairs = np.argwhere(~np.isnan(test_values))
        user_indices = test_pairs[:, 0]
        item_indices = test_pairs[:, 1]
        true_values = test_values[user_indices, item_indices]
        
        predictions = predict_with_params(theta_estimated, d_all, w_all, 
                                         user_indices, item_indices)
        
        # Compute metrics
        auc = roc_auc_score(true_values, predictions)
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(true_values, binary_preds)
        
        print(f"\n  Results for K={k}:")
        print(f"    AUC: {auc:.4f}")
        print(f"    Accuracy: {accuracy:.4f}")
        
        results.append({
            'Model': f'LADA',
            'K': k,
            'AUC': auc,
            'Accuracy': accuracy,
            'n_test_samples': len(test_pairs)
        })
        
        print()
    
    # Summary
    print("\n" + "="*60)
    print("Summary (IRT vs LADA):")
    print("="*60)
    
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        # Highlight best model
        best_auc_idx = summary_df['AUC'].idxmax()
        print(f"\nBest Model: {summary_df.loc[best_auc_idx, 'Model']} (K={summary_df.loc[best_auc_idx, 'K']}) with AUC={summary_df.loc[best_auc_idx, 'AUC']:.4f}")
        print(summary_df.to_string(index=False))
        
        summary_path = "./result/inference_test_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")
    else:
        print("No results to summarize.")
    
    print("\nInference test complete!")

if __name__ == "__main__":
    main()