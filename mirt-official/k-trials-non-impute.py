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
RESULT_DIR = "../data"
N_EPOCHS = 20
BATCH_SIZE = 65536
reg_strength = 0.01
os.makedirs(RESULT_DIR, exist_ok=True)
k_values_to_test = [1, 2, 3]
N_REPETITIONS = 200 # Set the number of times to repeat the experiment

# --- NEW: Define the number of repetitions ---
print(f"Performing {N_REPETITIONS} repetitions to assess variance.")

# ===================================================================
# B) Load Base Data (outside the loop)
# ===================================================================
print("\nLoading base data...")
resmat = pd.read_pickle("../data/resmat.pkl")
observed_pairs_base = np.argwhere(~resmat.isna().values)
n_persons, n_items = resmat.shape

# ===================================================================
# C) Experiment Loop
# ===================================================================
results_path = os.path.join(RESULT_DIR, "mirt_comparison_repeated.csv")

try:
    results_df = pd.read_csv(results_path)
    print(f"\nLoaded previous results from {results_path}")
except FileNotFoundError:
    results_df = pd.DataFrame()
    print("\nNo previous results file found. Starting fresh.")

# --- NEW: Outer loop for repetitions ---
for repetition in range(N_REPETITIONS):
    
    # ==================== OPTIMIZATION ====================
    # Check if this ENTIRE repetition can be skipped BEFORE doing any work.
    # We do this if all K values for this repetition are already in the results file.
    if not results_df.empty:
        # Find which K values have already been completed for this repetition
        completed_k_for_rep = results_df[results_df['Repetition'] == repetition]['K'].unique()
        
        # If the set of K's to test is a subset of the completed ones, skip
        if set(k_values_to_test).issubset(set(completed_k_for_rep)):
            print(f"All K values for Repetition {repetition + 1} are complete. Skipping.")
            continue
    # ================= END OPTIMIZATION ===================

    # --- NEW: Set seed for this specific repetition for reproducibility ---
    # This ensures each repetition is different from the others, but the same
    # repetition will always produce the same result if re-run.
    np.random.seed(repetition)
    torch.manual_seed(repetition)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(repetition)

    # --- NEW: Data splitting is now INSIDE the repetition loop ---
    print("Preparing data with new random split...")
    observed_pairs = observed_pairs_base.copy()
    np.random.shuffle(observed_pairs)
    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    test_pairs = observed_pairs[:n_test]
    train_pairs = observed_pairs[n_test:]
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat.values[test_rows, test_cols]

    train_rows_t = torch.from_numpy(train_rows)
    train_cols_t = torch.from_numpy(train_cols)
    train_ys_t = torch.from_numpy(train_ys).float()

    print(f"  n_persons: {n_persons}, n_items: {n_items}, n_train_obs: {len(train_ys)}")
    
    # Compute per-item weights (factor-aware balancing)
    item_counts = pd.Series(train_cols).value_counts().reindex(range(n_items), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights = inv_freq_weights.iloc[train_cols].values
    train_weights_t = torch.from_numpy(train_weights).float()

    # --- Inner loop for K values ---
    for k in k_values_to_test:
        # --- NEW: Check if this specific (K, Repetition) combination has been run ---
        if not results_df.empty and \
           ((results_df['K'] == k) & (results_df['Repetition'] == repetition)).any():
            print(f"Results for K={k}, Repetition={repetition} already exist. Skipping.")
            continue

        # --- Initialize Model Parameters ---

        theta = torch.randn(n_persons, k, device=device, requires_grad=True)
        a = torch.randn(n_items, k, device=device, requires_grad=True)
        b = torch.randn(n_items, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([theta, a, b], lr=0.01)
        train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t, train_weights_t)

        # Use multiple CPU cores to prepare data in the background.
        # A good starting point for num_workers is 4, 8, or the number of CPU cores you have.
        # pin_memory=True speeds up the CPU-to-GPU memory transfer.
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=24,
            pin_memory=True,
            persistent_workers=True
        )
        
        # --- Early Stopping Setup ---
        patience = 5
        best_auc = -np.inf
        best_state = None
        epochs_no_improve = 0
        
        print(f"Starting optimization for K={k}...")
        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False)
            
            for batch_rows, batch_cols, batch_ys, batch_wts in pbar:
                # Move the batch of data to the GPU
                batch_rows = batch_rows.to(device)
                batch_cols = batch_cols.to(device)
                batch_ys = batch_ys.to(device)
                batch_wts = batch_wts.to(device)

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
            if val_auc > best_auc + 1e-4:
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
        if best_state is None:
             print("Warning: Model did not improve. Using last state.")
             best_state = {
                'theta': theta.detach().clone(),
                'a': a.detach().clone(),
                'b': b.detach().clone()
            }
        theta, a, b = best_state['theta'], best_state['a'], best_state['b']

        # --- Final Evaluation ---
        with torch.no_grad():
            theta_test = theta[test_rows]
            a_test = a[test_cols]
            b_test = b[test_cols]
            logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
        test_auc = roc_auc_score(test_ys, probs_test)
        print(f"Final Test AUC for K={k}, Rep {repetition + 1}: {test_auc:.4f}")
        
        # --- Calculate Final Log-Likelihood, AIC, and BIC ---
        with torch.no_grad():
            # Move the tensors needed for this calculation to the GPU
            train_rows_gpu = train_rows_t.to(device)
            train_cols_gpu = train_cols_t.to(device)
            train_ys_gpu = train_ys_t.to(device)

            final_logits = torch.sum(theta[train_rows_gpu] * a[train_cols_gpu], axis=1) - b[train_cols_gpu]
            final_log_likelihood = -F.binary_cross_entropy_with_logits(
                final_logits, train_ys_gpu, reduction='sum').item()
        
        num_params = theta.numel() + a.numel() + b.numel()
        aic = 2 * num_params - 2 * final_log_likelihood
        bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood

        model_path = os.path.join(RESULT_DIR, f"mirt_model_k{k}_rep{repetition}.pt")
        torch.save({'theta': theta, 'a': a, 'b': b}, model_path)
        print(f"Saved best model parameters to {model_path}")

        # --- NEW: Add repetition number to the results ---
        current_result = pd.DataFrame([{
            'Repetition': repetition,
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


# ===================================================================
# D) Final Aggregated Results
# ===================================================================
print("\n\n--- INDIVIDUAL RUN RESULTS ---")
print(results_df.sort_values(by=['K', 'Repetition']).set_index(['K', 'Repetition']))

print("\n\n--- AGGREGATED RESULTS (MEAN & STD DEV) ---")
# --- NEW: Group by K and calculate mean and standard deviation ---
summary_stats = results_df.groupby('K')['Test AUC'].agg(['mean', 'std', 'count'])
print(summary_stats)