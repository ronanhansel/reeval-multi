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
RESULT_DIR = "../result"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================================================================
# B) Load and Prepare Data (This section is unchanged)
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
# C) Experiment Loop with Verbose Logging
# ===================================================================
# --- Define experiments and load previous results to allow resuming ---
k_values_to_test = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
results_path = os.path.join(RESULT_DIR, "mirt_comparison.csv")

try:
    results_df = pd.read_csv(results_path)
    print(f"\nLoaded previous results from {results_path}")
except FileNotFoundError:
    results_df = pd.DataFrame()
    print("\nNo previous results file found. Starting fresh.")

for k in k_values_to_test:
    print(f"\n{'='*20} PROCESSING K = {k} {'='*20}")
    
    # --- Check if this experiment was already completed ---
    if not results_df.empty and k in results_df['K'].values:
        print(f"Results for K={k} already exist. Skipping.")
        continue

    # --- Initialize Model Parameters ---
    theta = torch.randn(n_persons, k, device=device, requires_grad=True)
    a = torch.randn(n_items, k, device=device, requires_grad=True)
    b = torch.randn(n_items, device=device, requires_grad=True)

    # --- Set up Optimizer and Training Loop ---
    optimizer = torch.optim.Adam([theta, a, b], lr=0.01)
    N_EPOCHS = 20
    BATCH_SIZE = 65536
    reg_strength = 0.01

    train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Starting optimization for K={k}...")
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        
        # --- NEW: Create a tqdm progress bar for the batches ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for batch_rows, batch_cols, batch_ys in pbar:
            optimizer.zero_grad()
            theta_r = theta[batch_rows]
            a_c = a[batch_cols]
            b_c = b[batch_cols]
            dot = torch.sum(theta_r * a_c, axis=1)
            logits = dot - b_c
            log_likelihood = -F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='sum')
            l2_reg = -reg_strength * (torch.sum(theta**2) + torch.sum(a**2) + torch.sum(b**2))
            loss = -(log_likelihood + l2_reg) / len(batch_ys)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_rows)
            
            # --- NEW: Update the progress bar with the current batch loss ---
            pbar.set_postfix({'loss': loss.item()})
        
        # --- NEW: Print the average loss for the completed epoch ---
        avg_epoch_loss = epoch_loss / len(train_dataset)
        print(f"  Epoch {epoch+1}/{N_EPOCHS} - Average Loss: {avg_epoch_loss:.4f}")

    print("Optimization complete.")

    # --- Evaluate on Test Set ---
    with torch.no_grad():
        theta_test = theta[test_rows]
        a_test = a[test_cols]
        b_test = b[test_cols]
        logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
    test_auc = roc_auc_score(test_ys, probs_test)
    print(f"Test AUC for K={k}: {test_auc:.4f}")

    # --- Calculate Final Log-Likelihood, AIC, and BIC ---
    with torch.no_grad():
        final_logits = torch.sum(theta[train_rows_t] * a[train_cols_t], axis=1) - b[train_cols_t]
        final_log_likelihood = -F.binary_cross_entropy_with_logits(final_logits, train_ys_t, reduction='sum').item()
    
    num_params = theta.numel() + a.numel() + b.numel()
    aic = 2 * num_params - 2 * final_log_likelihood
    bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood

    # --- Save the trained model parameters ---
    model_path = os.path.join(RESULT_DIR, f"mirt_model_k{k}.pt")
    torch.save({
        'theta': theta.detach(),
        'a': a.detach(),
        'b': b.detach(),
    }, model_path)
    print(f"Saved model parameters to {model_path}")

    # --- Append results to the DataFrame and save to CSV ---
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