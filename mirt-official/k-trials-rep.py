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

# MODIFICATION: Fill all NaN values with 0
print("Filling all NaN values with 0...")
resmat.fillna(0, inplace=True)

# This line now selects ALL (row, col) pairs because there are no NaNs.
observed_pairs = np.argwhere(~resmat.isna().values)
n_persons, n_items = resmat.shape

print(f"n_persons: {n_persons}, n_items: {n_items}")
print(f"Total observations to be split for train/test: {len(observed_pairs)}")

# ===================================================================
# C) Experiment Loop with Correct Re-initialization
# ===================================================================
k_values_to_test = [19]
N_REPETITIONS = 4 # Set the number of times to repeat the experiment
results_path = os.path.join(RESULT_DIR, "mirt_comparison.csv")

try:
    results_df = pd.read_csv(results_path)
    print(f"\nLoaded previous results from {results_path}")
except FileNotFoundError:
    results_df = pd.DataFrame()
    print("\nNo previous results file found. Starting fresh.")

# Outer loop for repetitions
for repetition in range(N_REPETITIONS):
    print(f"\n{'#'*25} STARTING REPETITION {repetition + 1}/{N_REPETITIONS} {'#'*25}")

    # --- 1. SET SEEDS FOR REPRODUCIBILITY ---
    # Using the repetition number as the seed makes each run different but reproducible
    seed = repetition
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 2. RESHUFFLE AND SPLIT DATA FOR THIS REPETITION ---
    print("Shuffling and creating new train/test split...")
    np.random.shuffle(observed_pairs) # This is the key change!

    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    test_pairs = observed_pairs[:n_test]
    train_pairs = observed_pairs[n_test:]
    
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat.values[test_rows, test_cols]

    train_rows_t = torch.from_numpy(train_rows).to(device)
    train_cols_t = torch.from_numpy(train_cols).to(device)
    train_ys_t = torch.from_numpy(train_ys).float().to(device)

    print(f"  n_train_obs: {len(train_ys)}, n_test_obs: {len(test_ys)}")

    # --- Baseline calculation moved inside the loop as test set changes ---
    item_mean = np.mean(resmat.values, axis=0)
    preds_item_mean = item_mean[test_cols]
    auc_item = roc_auc_score(test_ys, preds_item_mean)

    person_mean = np.mean(resmat.values, axis=1)
    preds_person_mean = person_mean[test_rows]
    auc_person = roc_auc_score(test_ys, preds_person_mean)
    print(f"Rep {repetition+1} Baseline AUCs: item_mean = {auc_item:.4f}, person_mean = {auc_person:.4f}")

    for k in k_values_to_test:
        print(f"\n{'='*20} PROCESSING K = {k} {'='*20}")
        
        # --- 3. RE-INITIALIZE MODEL PARAMETERS (was already correct) ---
        theta = torch.randn(n_persons, k, device=device, requires_grad=True)
        a = torch.randn(n_items, k, device=device, requires_grad=True)
        b = torch.randn(n_items, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([theta, a, b], lr=0.01)
        N_EPOCHS = 20
        BATCH_SIZE = 65536
        reg_strength = 0.01

        train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # --- Early Stopping Setup ---
        patience = 5
        best_auc = -np.inf
        best_state = None
        epochs_no_improve = 0
        
        print(f"Starting optimization for K={k}, Repetition {repetition+1}...")
        for epoch in range(N_EPOCHS):
            epoch_loss = 0
            # Disable the progress bar if not running interactively by setting disable=None
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Rep {repetition+1}]")
            
            for batch_rows, batch_cols, batch_ys in pbar:
                optimizer.zero_grad()
                theta_r = theta[batch_rows]
                a_c = a[batch_cols]
                b_c = b[batch_cols]
                dot = torch.sum(theta_r * a_c, axis=1)
                logits = dot - b_c

                bce_loss = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='sum')
                l2_reg = reg_strength * (torch.sum(theta**2) + torch.sum(a**2) + torch.sum(b**2))
                loss = (bce_loss + l2_reg) / len(batch_ys)

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
        print(f"Final Test AUC for K={k}, Rep {repetition+1}: {test_auc:.4f}")

        # --- Calculate Final Log-Likelihood, AIC, and BIC ---
        with torch.no_grad():
            final_logits = torch.sum(theta[train_rows_t] * a[train_cols_t], axis=1) - b[train_cols_t]
            final_log_likelihood = -F.binary_cross_entropy_with_logits(
                final_logits, train_ys_t, reduction='sum').item()
        
        num_params = theta.numel() + a.numel() + b.numel()
        aic = 2 * num_params - 2 * final_log_likelihood
        bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood

        # Filename now includes repetition number
        model_path = os.path.join(RESULT_DIR, f"mirt_model_k{k}_rep{repetition+1}.pt")
        torch.save({'theta': theta, 'a': a, 'b': b}, model_path)
        print(f"Saved best model parameters to {model_path}")

        # Added Repetition column to results
        current_result = pd.DataFrame([{
            'Repetition': repetition + 1,
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
print("Full results from all repetitions:")
print(results_df)

print("\nSummary statistics across repetitions:")
# Better summary using groupby
summary_stats = results_df.groupby('K')['Test AUC'].agg(['mean', 'std', 'min', 'max', 'count'])
print(summary_stats)