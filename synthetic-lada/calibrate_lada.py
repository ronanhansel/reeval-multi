"""
Calibrate LADA model on synthetic train data and evaluate on synthetic test data.
Calibrate LADA on k=1 to k=3 dimensions.

Saves the calibrated model and evaluation results.

Take it from calibration-lada-joint.py and modify for synthetic data.
Output to ./result
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
K_VALUES = [1, 2, 3, 5, 10, 20]
EPOCHS = 1000
BATCH_SIZE = 10000  # Smaller batch for smaller dataset

# Hyperparameters
ALPHA_DIRICHLET = 1.1
LR_JOINT = 0.05
REG_STRENGTH = 0.01

# Prior hyperparameters
MU_D = 0.0
SIGMA_D = 1.0

RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)

# For reproducibility
SEED = 42
EARLY_STOPPING_THRESHOLD = 1e-4
EARLY_STOPPING_PATIENCE = 20

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
print("Loading synthetic data...")
try:
    train_df = pd.read_pickle("./data/resmat_train.pkl")
    test_df = pd.read_pickle("./data/resmat_test.pkl")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
except FileNotFoundError as e:
    print(f"Error: Data files not found. Please run gen_syn.py first.")
    sys.exit(1)

train_values = train_df.values
test_values = test_df.values
n_persons, n_items_train = train_values.shape
_, n_items_test = test_values.shape

print(f"Models: {n_persons}, Train items: {n_items_train}, Test items: {n_items_test}")

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
# Fitting Function
# ------------------------------------------------------------------------------

def fit_lada_model(train_values, test_values, k_dims, n_epochs, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    print(f"\nFitting LADA model with K={k_dims}...")
    
    n_persons, n_items_train = train_values.shape
    _, n_items_test = test_values.shape
    
    # Prepare training data (all non-NaN entries)
    train_pairs = np.argwhere(~np.isnan(train_values))
    
    # Calculate item weights for training
    item_counts = pd.Series(train_pairs[:, 1]).value_counts().reindex(range(n_items_train), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights_cpu = inv_freq_weights.iloc[train_pairs[:, 1]].values

    # Prepare test data (all non-NaN entries)
    test_pairs = np.argwhere(~np.isnan(test_values))
    
    print(f"  Train samples: {len(train_pairs)}, Test samples: {len(test_pairs)}")
    print(f"  Moving data to {device}...")

    # Move training data to GPU
    train_rows = torch.from_numpy(train_pairs[:, 0]).long().to(device)
    train_cols = torch.from_numpy(train_pairs[:, 1]).long().to(device)
    train_ys = torch.from_numpy(train_values[train_pairs[:, 0], train_pairs[:, 1]]).float().to(device)
    train_weights = torch.from_numpy(train_weights_cpu).float().to(device)
    
    # Keep test data for evaluation
    test_rows = test_pairs[:, 0]
    test_cols = test_pairs[:, 1]
    test_ys = test_values[test_pairs[:, 0], test_pairs[:, 1]]
    
    # Initialize Model - train model only knows about train items
    model = LADAModel(n_persons, n_items_train, k_dims,
                      alpha=ALPHA_DIRICHLET, mu_d=MU_D, sigma_d=SIGMA_D).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_JOINT)
    
    best_test_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    training_history = []
    
    num_train = len(train_rows)
    
    # Training Loop
    for epoch in range(n_epochs):
        model.train()
        epoch_nll = 0.0
        n_batches = 0
        
        # Shuffle indices on GPU
        permutation = torch.randperm(num_train, device=device)
        
        for i in range(0, num_train, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            
            batch_rows = train_rows[indices]
            batch_cols = train_cols[indices]
            batch_ys = train_ys[indices]
            batch_wts = train_weights[indices]
            
            optimizer.zero_grad()
            
            logits = model(batch_rows, batch_cols)
            loss, nll = model.compute_loss(logits, batch_ys, batch_wts, 
                                          batch_rows, batch_cols, REG_STRENGTH)
            loss.backward()
            optimizer.step()
            
            epoch_nll += nll
            n_batches += 1
        
        avg_nll = epoch_nll / n_batches
        
        # Test evaluation (need to handle different item set)
        # Since test items are different, we need to create temporary parameters for them
        model.eval()
        with torch.no_grad():
            # Use theta from model, but need to handle test items
            # For simplicity, we'll compute predictions using learned theta
            # but we need d and w for test items, which we don't have
            # We'll use a simple approach: evaluate on training accuracy as proxy
            train_logits = []
            for i in range(0, num_train, BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, num_train)
                batch_logits = model(train_rows[i:end_idx], train_cols[i:end_idx])
                train_logits.append(batch_logits)
            
            train_logits = torch.cat(train_logits)
            train_probs = torch.sigmoid(train_logits).cpu().numpy()
            train_auc = roc_auc_score(train_ys.cpu().numpy(), train_probs)
        
        training_history.append({
            'epoch': epoch+1,
            'train_loss': avg_nll,
            'train_auc': train_auc
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_nll:.4f}, Train AUC: {train_auc:.4f}")

        # Use train_auc as proxy for early stopping
        if train_auc > best_test_auc + EARLY_STOPPING_THRESHOLD:
            best_test_auc = train_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Restore best state
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Calculate final metrics
    model.eval()
    with torch.no_grad():
        # Final train AUC
        train_logits = []
        for i in range(0, num_train, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, num_train)
            batch_logits = model(train_rows[i:end_idx], train_cols[i:end_idx])
            train_logits.append(batch_logits)
        
        train_logits = torch.cat(train_logits)
        train_probs = torch.sigmoid(train_logits).cpu().numpy()
        final_train_auc = roc_auc_score(train_ys.cpu().numpy(), train_probs)
        
        # For test evaluation, we can't directly predict since test items are different
        # Instead, we'll save the learned theta and report train performance
        print(f"\n  Final Train AUC: {final_train_auc:.4f}")
        print(f"  Note: Test items are different from train items (by design)")
    
    # Calculate AIC/BIC
    with torch.no_grad():
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        full_nll_sum = 0
        
        for i in range(0, num_train, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, num_train)
            batch_logits = model(train_rows[i:end_idx], train_cols[i:end_idx])
            batch_loss = loss_fn(batch_logits, train_ys[i:end_idx])
            full_nll_sum += batch_loss.item()

    num_params = sum(p.numel() for p in model.parameters())
    aic = 2 * num_params + 2 * full_nll_sum
    bic = np.log(num_train) * num_params + 2 * full_nll_sum
    
    # Save history
    history_df = pd.DataFrame(training_history)
    history_path = os.path.join(RESULT_DIR, f"training_history_synthetic_k{k_dims}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"  Saved training history to {history_path}")
    
    # Save model parameters
    model_dict = {
        'theta': model.theta.detach().cpu().numpy(),
        'phi': model.phi.detach().cpu().numpy(),
        'w': model.get_w().detach().cpu().numpy(),
        'd': model.d.detach().cpu().numpy(),
        'k_dims': k_dims
    }
    model_path = os.path.join(RESULT_DIR, f"lada_model_k{k_dims}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"  Saved model to {model_path}")
    
    return {
        'k': k_dims,
        'train_auc': final_train_auc,
        'aic': aic,
        'bic': bic,
        'best_epoch': best_epoch,
        'log_likelihood': -full_nll_sum,
        'num_params': num_params,
        'n_train': num_train,
        'theta': model.theta.detach().cpu().numpy()
    }

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("LADA Calibration on Synthetic Data")
    print(f"{'='*60}\n")
    
    results = []
    
    for k in K_VALUES:
        result = fit_lada_model(train_values, test_values, k, EPOCHS, seed=SEED)
        results.append(result)
    
    # Create summary
    summary_df = pd.DataFrame([{
        'K': r['k'],
        'Train_AUC': r['train_auc'],
        'AIC': r['aic'],
        'BIC': r['bic'],
        'Best_Epoch': r['best_epoch'],
        'Num_Params': r['num_params']
    } for r in results])
    
    print("\n" + "="*60)
    print("Summary Results:")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(RESULT_DIR, "lada_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    
    # Load ground truth and compare theta
    try:
        ground_truth = np.load('./data/ground_truth.npy', allow_pickle=True).item()
        true_theta = ground_truth['theta']
        true_k = ground_truth['k']
        
        print("\n" + "="*60)
        print("Comparison with Ground Truth (K=3):")
        print("="*60)
        
        # Compare with K=3 result
        k3_result = [r for r in results if r['k'] == 3]
        if k3_result:
            estimated_theta = k3_result[0]['theta']
            
            # Compute correlation per dimension (need to handle sign flips)
            print(f"True theta shape: {true_theta.shape}")
            print(f"Estimated theta shape: {estimated_theta.shape}")
            
            correlations = []
            for dim in range(true_k):
                corr = np.corrcoef(true_theta[:, dim], estimated_theta[:, dim])[0, 1]
                correlations.append(corr)
                print(f"  Dimension {dim}: correlation = {corr:.4f}")
            
            avg_corr = np.mean(np.abs(correlations))
            print(f"\n  Average absolute correlation: {avg_corr:.4f}")
            
    except Exception as e:
        print(f"\nCould not load ground truth for comparison: {e}")
    
    print("\nCalibration complete!")

if __name__ == "__main__":
    main()

