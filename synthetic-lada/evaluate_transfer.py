"""
Evaluate Transfer Learning with LIMITED BUDGET (10 Items).

Scenario:
- We can only afford to run 10 items per model.
- We want to predict the True Score (average of all 100 items).

Comparison:
1. Baseline: Just use the raw average of the 10 items.
2. IRT Transfer: Use 10 items to calibrate, then predict using 1D Theta.
3. LADA Transfer: Use 10 items to calibrate, then predict using 3D Theta.
"""

import numpy as np
import pandas as pd
import pickle
import os
import glob
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR = "./data"
RESULT_DIR = "./result"
BUDGET_ITEMS = 10  # We only see 10 items!
SEED = 42

# ------------------------------------------------------------------------------
# Transfer Logic
# ------------------------------------------------------------------------------
def evaluate_limited_transfer(theta, observed_scores, true_gt_scores):
    """
    Train a linear probe using ONLY the 'observed_scores' (noisy 10-item avg).
    Evaluate prediction accuracy against 'true_gt_scores' (clean 100-item avg).
    """
    loo = LeaveOneOut()
    preds = []
    
    # We want to predict the TRUE score, but we can only learn from the OBSERVED score.
    X = theta
    y_noisy = observed_scores 
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_noisy = y_noisy[train_idx]
        
        # Standardize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-9
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        # Fit Probe on NOISY data
        # "Based on these 10 items, how does Theta map to Score?"
        probe = Ridge(alpha=1.0)
        probe.fit(X_train_norm, y_train_noisy)
        
        # Predict
        pred = probe.predict(X_test_norm)[0]
        preds.append(np.clip(pred, 0.0, 1.0))
        
    preds = np.array(preds)
    
    # Evaluate against GROUND TRUTH (The 100-item average)
    r2 = r2_score(true_gt_scores, preds)
    rho, _ = spearmanr(true_gt_scores, preds)
    
    return r2, rho

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    print("="*80)
    print(f"LIMITED BUDGET EVALUATION (Only {BUDGET_ITEMS} Items Observed)")
    print("Can LADA predict the full leaderboard from a tiny sample?")
    print("="*80)

    # 1. Load Embeddings
    try:
        with open(os.path.join(RESULT_DIR, "irt_rasch_model.pkl"), 'rb') as f:
            theta_irt = pickle.load(f)['theta'].reshape(-1, 1)
        with open(os.path.join(RESULT_DIR, "lada_model_k3.pkl"), 'rb') as f:
            theta_lada = pickle.load(f)['theta']
    except:
        print("Error loading models.")
        return

    # 2. Find Benchmarks
    files = sorted(glob.glob(os.path.join(DATA_DIR, "resmat_*.pkl")))
    files = [f for f in files if "train" not in f and "test" not in f] # Only specialized ones
    
    print(f"\nEvaluating on {len(files)} Specialized Benchmarks...")
    print("-" * 100)
    print(f"{'Benchmark':<15} | {'Baseline Rho':<12} || {'IRT Rho':<10} | {'LADA Rho':<10} || {'LADA vs Base'}")
    print("-" * 100)
    
    results = []
    
    np.random.seed(SEED) # Ensure same 10 items for fair comparison

    for f_path in files:
        name = os.path.basename(f_path).replace("resmat_", "").replace(".pkl", "")
        
        # Load Full Data
        df = pd.read_pickle(f_path)
        
        # A. Establish Ground Truth (100 items)
        true_gt_scores = df.mean(axis=1).values
        
        # B. Simulate Budget (Random 10 items)
        # We pick 10 items randomly. These are the ONLY items the models "saw".
        chosen_items = np.random.choice(df.columns, BUDGET_ITEMS, replace=False)
        observed_scores = df[chosen_items].mean(axis=1).values
        
        # 1. Baseline: Raw Ranking (Just trust the 10 items)
        rho_base, _ = spearmanr(true_gt_scores, observed_scores)
        
        # 2. IRT Transfer
        _, rho_irt = evaluate_limited_transfer(theta_irt, observed_scores, true_gt_scores)
        
        # 3. LADA Transfer
        _, rho_lada = evaluate_limited_transfer(theta_lada, observed_scores, true_gt_scores)
        
        improvement = rho_lada - rho_base
        
        print(f"{name:<15} | {rho_base:>12.4f} || {rho_irt:>10.4f} | {rho_lada:>10.4f} || {improvement:+10.4f}")
        
        results.append({
            'Benchmark': name,
            'Baseline': rho_base,
            'IRT': rho_irt,
            'LADA': rho_lada
        })

    # Summary Stats
    print("-" * 100)
    avg_base = np.mean([r['Baseline'] for r in results])
    avg_lada = np.mean([r['LADA'] for r in results])
    print(f"AVERAGE         | {avg_base:>12.4f} || {'-':>10} | {avg_lada:>10.4f} || {avg_lada - avg_base:+10.4f}")
    
    # Visualization
    df_res = pd.DataFrame(results)
    df_res.set_index('Benchmark', inplace=True)
    
    ax = df_res.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title(f"Ranking Accuracy with only {BUDGET_ITEMS} Items (Ground Truth = 100 Items)")
    plt.ylabel("Spearman Correlation with Truth")
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "limited_budget_ranking.png"))
    print(f"\nSaved plot to {os.path.join(RESULT_DIR, 'limited_budget_ranking.png')}")

if __name__ == "__main__":
    main()