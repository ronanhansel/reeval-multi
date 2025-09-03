import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# ===================================================================
# A) Configuration & Device Setup
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
RESULT_DIR = "./output"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================================================================
# B) Load and Prepare Data
# ===================================================================
print("\nLoading and preparing data...")
resmat = pd.read_pickle("../data/resmat.pkl")
observed_pairs = np.argwhere(~resmat.isna().values)
np.random.seed(42)
np.random.shuffle(observed_pairs)
test_frac = 0.20
n_test = int(len(observed_pairs) * test_frac)
test_pairs = observed_pairs[:n_test]
train_pairs = observed_pairs[n_test:]
train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
train_ys = resmat.values[train_rows, train_cols]
test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
test_ys = resmat.values[test_rows, test_cols]
n_persons, n_items = resmat.shape

train_rows_t = torch.from_numpy(train_rows).to(device)
train_cols_t = torch.from_numpy(train_cols).to(device)
train_ys_t = torch.from_numpy(train_ys).float().to(device)

print(f"n_persons: {n_persons}, n_items: {n_items}, n_train_obs: {len(train_ys)}")

# ===================================================================
# B2) NEW: Compute per-item weights (factor-aware balancing)
# ===================================================================
print("\nComputing item weights to correct imbalance...")
item_counts = pd.Series(train_cols).value_counts().reindex(range(n_items), fill_value=0)
inv_freq_weights = 1.0 / (item_counts + 1e-6)
inv_freq_weights /= inv_freq_weights.mean()

train_weights = inv_freq_weights.iloc[train_cols].values
train_weights_t = torch.from_numpy(train_weights).float().to(device)

# ===================================================================
# C) Experiment Loop with Verbose Logging + Early Stopping
# ===================================================================
k_values_to_test = [19]
results_path = os.path.join(RESULT_DIR, "mirt_comparison.csv")

try:
    results_df = pd.read_csv(results_path)
    print(f"\nLoaded previous results from {results_path}")
except FileNotFoundError:
    results_df = pd.DataFrame()
    print("\nNo previous results file found. Starting fresh.")

for k in k_values_to_test:
    print(f"\n{'='*20} PROCESSING K = {k} {'='*20}")
    
    if not results_df.empty and k in results_df['K'].values:
        print(f"Results for K={k} already exist. Skipping.")
        continue

    # --- Initialize Model Parameters ---
    theta = torch.randn(n_persons, k, device=device, requires_grad=True)
    a = torch.randn(n_items, k, device=device, requires_grad=True)
    b = torch.randn(n_items, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([theta, a, b], lr=0.01)
    N_EPOCHS = 20
    BATCH_SIZE = 65536
    reg_strength = 0.01

    train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t, train_weights_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Early Stopping Setup ---
    patience = 5
    best_auc = -np.inf
    best_state = None
    epochs_no_improve = 0
    
    print(f"Starting optimization for K={k}...")
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for batch_rows, batch_cols, batch_ys, batch_wts in pbar:
            optimizer.zero_grad()
            theta_r = theta[batch_rows]
            a_c = a[batch_cols]
            b_c = b[batch_cols]
            dot = torch.sum(theta_r * a_c, axis=1)
            logits = dot - b_c

            loss_per_obs = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            weighted_loss = (batch_wts * loss_per_obs).sum()
            l2_reg = reg_strength * (torch.sum(theta**2) + torch.sum(a**2) + torch.sum(b**2))
            loss = (weighted_loss + l2_reg) / len(batch_ys)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_rows)
            pbar.set_postfix({'loss': loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataset)

        # --- Validation AUC ---
        with torch.no_grad():
            theta_test = theta[test_rows]
            a_test = a[test_cols]
            b_test = b[test_cols]
            logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
            val_auc = roc_auc_score(test_ys, probs_test)

        print(f"  Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_epoch_loss:.4f}, Val AUC: {val_auc:.4f}")

        # --- Early Stopping Check ---
        if val_auc > best_auc + 1e-4:  # small tolerance
            best_auc = val_auc
            best_state = {
                'theta': theta.detach().clone(),
                'a': a.detach().clone(),
                'b': b.detach().clone()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val AUC: {best_auc:.4f}")
                break

    print("Optimization complete.")

    # --- Use Best Model ---
    theta, a, b = best_state['theta'], best_state['a'], best_state['b']

    # --- Final Evaluation ---
    with torch.no_grad():
        theta_test = theta[test_rows]
        a_test = a[test_cols]
        b_test = b[test_cols]
        logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
    test_auc = roc_auc_score(test_ys, probs_test)
    print(f"Final Test AUC for K={k}: {test_auc:.4f}")

    # --- Calculate Final Log-Likelihood, AIC, and BIC ---
    with torch.no_grad():
        final_logits = torch.sum(theta[train_rows_t] * a[train_cols_t], axis=1) - b[train_cols_t]
        final_log_likelihood = -F.binary_cross_entropy_with_logits(
            final_logits, train_ys_t, reduction='sum').item()
    
    num_params = theta.numel() + a.numel() + b.numel()
    aic = 2 * num_params - 2 * final_log_likelihood
    bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood

    model_path = os.path.join(RESULT_DIR, f"mirt_model_k{k}.pt")
    torch.save({'theta': theta, 'a': a, 'b': b}, model_path)
    print(f"Saved best model parameters to {model_path}")

    current_result = pd.DataFrame([{
        'K': k,
        'Test AUC': test_auc,
        'AIC': aic,
        'BIC': bic,
        'Num Params': num_params,
        'LogLikelihood': final_log_likelihood
    }])
    results_df = pd.concat([results_df, current_result], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"Updated results saved to {results_path}")


print("\n\n--- FINAL MODEL COMPARISON RESULTS ---")
print(results_df.sort_values(by='K').set_index('K'))
