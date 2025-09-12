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
# User: choose how many group skills (must match your NMF CSV)
# ===================================================================
N_GROUPS = 10  # <-- change this to the K you used in NMF (e.g., 10, 19, etc.)
item_skill_csv = f"item_skill_assignments_K{N_GROUPS}.csv"

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
# Load item->group assignments (from your NMF output)
# ===================================================================
if os.path.exists(item_skill_csv):
    df_assign = pd.read_csv(item_skill_csv)
    # Expect columns: item, assigned_skill
    item2group = df_assign['assigned_skill'].values
    if len(item2group) != n_items:
        raise ValueError(f"Item assignment length ({len(item2group)}) != n_items ({n_items})")
    print(f"Loaded item->group mapping from {item_skill_csv}")
else:
    # fallback: random assignment (NOT ideal) — warn user
    print(f"Warning: {item_skill_csv} not found. Falling back to random assignment.")
    rng = np.random.RandomState(42)
    item2group = rng.randint(0, N_GROUPS, size=n_items)

# Build mask matrix (n_items x N_GROUPS) where mask[i,g]=1 if item i assigned to g
mask = np.zeros((n_items, N_GROUPS), dtype=np.float32)
for i, g in enumerate(item2group):
    mask[i, int(g)] = 1.0
mask_torch = torch.from_numpy(mask).to(device)

# ===================================================================
# C) Experiment Loop with Verbose Logging + Early Stopping (bifactor)
# ===================================================================
# We'll test a single bifactor configuration: 1 general + N_GROUPS group factors.
k_values_to_test = [1 + N_GROUPS]   # keep interface similar; K = 1 + #groups
results_path = os.path.join(RESULT_DIR, "mirt_bifactor_comparison.csv")

try:
    results_df = pd.read_csv(results_path)
    print(f"\nLoaded previous results from {results_path}")
except FileNotFoundError:
    results_df = pd.DataFrame()
    print("\nNo previous results file found. Starting fresh.")

for k in k_values_to_test:
    print(f"\n{'='*20} PROCESSING BIFACTOR (K_total = {k}) {'='*20}")
    
    if not results_df.empty and k in results_df['K'].values:
        print(f"Results for K={k} already exist. Skipping.")
        continue

    # --- Initialize Bi-Factor Model Parameters ---
    # theta_general: (n_persons, 1)
    theta_general = torch.randn(n_persons, 1, device=device, requires_grad=True)
    # theta_group: (n_persons, N_GROUPS)
    theta_group = torch.randn(n_persons, N_GROUPS, device=device, requires_grad=True)

    # a_general: (n_items, 1)
    a_general = torch.randn(n_items, 1, device=device, requires_grad=True)
    # a_group: (n_items, N_GROUPS) but non-assigned entries will be masked to zero via mask_torch
    # we keep them as parameters but they will be multiplied by mask in forward pass
    a_group = torch.randn(n_items, N_GROUPS, device=device, requires_grad=True) * 0.1

    b = torch.randn(n_items, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([theta_general, theta_group, a_general, a_group, b], lr=0.01)
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
    
    print(f"Starting bifactor optimization (1 general + {N_GROUPS} groups)...")
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for batch_rows, batch_cols, batch_ys, batch_wts in pbar:
            optimizer.zero_grad()
            # Gather parameters for this batch
            tg_g = theta_general[batch_rows]            # (batch, 1)
            tg_s = theta_group[batch_rows]              # (batch, N_GROUPS)

            a_g = a_general[batch_cols]                 # (batch, 1)
            a_s = a_group[batch_cols] * mask_torch[batch_cols]  # (batch, N_GROUPS) masked

            b_c = b[batch_cols]                         # (batch,)

            # Compute logits: general term + group term - b
            term_general = (tg_g.squeeze(1) * a_g.squeeze(1))   # (batch,)
            term_group = torch.sum(tg_s * a_s, dim=1)           # (batch,)
            logits = term_general + term_group - b_c

            loss_per_obs = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            weighted_loss = (batch_wts * loss_per_obs).sum()
            # Regularization: penalize magnitude of all params
            l2_reg = reg_strength * (
                torch.sum(theta_general**2) + torch.sum(theta_group**2)
                + torch.sum(a_general**2) + torch.sum(a_group**2) + torch.sum(b**2)
            )
            loss = (weighted_loss + l2_reg) / len(batch_ys)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_rows)
            pbar.set_postfix({'loss': loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataset)

        # --- Validation AUC ---
        with torch.no_grad():
            tg_g_test = theta_general[test_rows]
            tg_s_test = theta_group[test_rows]
            a_g_test = a_general[test_cols]
            a_s_test = a_group[test_cols] * mask_torch[test_cols]
            b_test = b[test_cols]

            term_general_test = (tg_g_test.squeeze(1) * a_g_test.squeeze(1))
            term_group_test = torch.sum(tg_s_test * a_s_test, dim=1)
            logits_test = term_general_test + term_group_test - b_test
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
            val_auc = roc_auc_score(test_ys, probs_test)

        print(f"  Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_epoch_loss:.4f}, Val AUC: {val_auc:.4f}")

        # --- Early Stopping Check ---
        if val_auc > best_auc + 1e-4:  # small tolerance
            best_auc = val_auc
            best_state = {
                'theta_general': theta_general.detach().clone(),
                'theta_group': theta_group.detach().clone(),
                'a_general': a_general.detach().clone(),
                'a_group': a_group.detach().clone(),
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
    theta_general, theta_group = best_state['theta_general'], best_state['theta_group']
    a_general, a_group, b = best_state['a_general'], best_state['a_group'], best_state['b']

    # --- Final Evaluation ---
    with torch.no_grad():
        tg_g_test = theta_general[test_rows]
        tg_s_test = theta_group[test_rows]
        a_g_test = a_general[test_cols]
        a_s_test = a_group[test_cols] * mask_torch[test_cols]
        b_test = b[test_cols]

        term_general_test = (tg_g_test.squeeze(1) * a_g_test.squeeze(1))
        term_group_test = torch.sum(tg_s_test * a_s_test, dim=1)
        logits_test = term_general_test + term_group_test - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
    test_auc = roc_auc_score(test_ys, probs_test)
    print(f"Final Test AUC (bifactor) : {test_auc:.4f}")

    # --- Calculate Final Log-Likelihood, AIC, and BIC ---
    with torch.no_grad():
        final_tg_g = theta_general[train_rows_t]
        final_tg_s = theta_group[train_rows_t]
        final_a_g = a_general[train_cols_t]
        final_a_s = a_group[train_cols_t] * mask_torch[train_cols_t]
        final_b = b[train_cols_t]

        final_logits = (final_tg_g.squeeze(1) * final_a_g.squeeze(1)) + torch.sum(final_tg_s * final_a_s, dim=1) - final_b
        final_log_likelihood = -F.binary_cross_entropy_with_logits(
            final_logits, train_ys_t, reduction='sum').item()
    
    num_params = (
        theta_general.numel() + theta_group.numel()
        + a_general.numel() + a_group.numel() + b.numel()
    )
    aic = 2 * num_params - 2 * final_log_likelihood
    bic = np.log(len(train_ys)) * num_params - 2 * final_log_likelihood

    model_path = os.path.join(RESULT_DIR, f"mirt_bifactor_groups{N_GROUPS}.pt")
    torch.save({
        'theta_general': theta_general,
        'theta_group': theta_group,
        'a_general': a_general,
        'a_group': a_group,
        'b': b,
        'item2group': item2group
    }, model_path)
    print(f"Saved best bi-factor model parameters to {model_path}")

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
