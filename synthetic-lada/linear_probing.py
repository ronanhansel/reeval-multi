"""
Real-World Workflow: "Bayesian Smoothing" for Expensive Tests.

Scenario:
1. We have robust Theta from a cheap Anchor (Training Data).
2. We run a tiny, noisy Pilot (5 items) of an Expensive Benchmark.
3. We use Theta to 'smooth' the noisy scores and recover the true ranking.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# In real life, K is decided by your Anchor model (e.g., K=3 or 5)
ANCHOR_MODEL_PATH = "./result/lada_model_k3.pkl" 

# ------------------------------------------------------------------------------
# Workflow
# ------------------------------------------------------------------------------
def run_real_world_smoothing():
    print("="*60)
    print("REAL-WORLD workflow: Small Budget (5 Items)")
    print("="*60)
    
    # 1. Load Anchor Profiles (Theta)
    # This represents your knowledge from MMLU/HELM
    try:
        with open(ANCHOR_MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            theta = model_data['theta'] # (20, 3)
            print(f"1. Loaded Anchor Profiles for {theta.shape[0]} models.")
    except:
        print("Error: Anchor model not found.")
        return

    # 2. Simulate the "Expensive Test" (Real Ground Truth)
    # Let's say the expensive test is a "Coding" Benchmark
    # We load the FULL 100-item ground truth just to check our accuracy later
    try:
        df_full = pd.read_pickle("./data/resmat_coding.pkl")
        true_scores = df_full.mean(axis=1).values # The "God View" ranking
        print(f"2. Target Benchmark: 'Coding' (100 items hidden cost)")
    except:
        print("Error: Coding benchmark data not found.")
        return

    # 3. Run the "Budget" Experiment (Only 5 Items)
    # We randomly pick 5 items to represent our limited budget
    np.random.seed(42)
    budget_items = np.random.choice(df_full.columns, 5, replace=False)
    print(f"3. Running Budget Pilot: {list(budget_items)}")
    
    # Get the Noisy Scores (Raw Average of 5 items)
    # This is what you would get if you didn't use LADA
    noisy_scores = df_full[budget_items].mean(axis=1).values
    
    # 4. Apply LADA Smoothing (The Fix)
    # We fit a Ridge Regressor: Noisy_Score ~ w * Theta
    # Alpha=1.0 prevents overfitting to the noise
    probe = Ridge(alpha=1.0)
    probe.fit(theta, noisy_scores)
    
    # Predict "Smoothed" Scores
    smoothed_scores = probe.predict(theta)
    
    # 5. Validation: Did we get the "Right Result"?
    # We compare both rankings to the TRUE 100-item ranking
    rho_raw, _ = spearmanr(true_scores, noisy_scores)
    rho_smooth, _ = spearmanr(true_scores, smoothed_scores)
    r2_check = r2_score(noisy_scores, smoothed_scores) # Proxy for "Signal Strength"

    print("\n" + "-"*60)
    print("RESULTS: Accuracy vs Ground Truth (100 Items)")
    print("-" * 60)
    print(f"Raw Score (5 items) Correlation:      {rho_raw:.4f}")
    print(f"LADA Smoothed Score Correlation:      {rho_smooth:.4f}")
    print(f"Improvement:                          {rho_smooth - rho_raw:+.4f}")
    print("-" * 60)
    
    # 6. The "Confidence" Check (For the User)
    print("\nHow to know if you can trust this result?")
    print(f"Probe R² (Signal Strength): {r2_check:.2f}")
    
    if r2_check > 0.5:
        print(">> VERDICT: TRUST. The 5 items align well with your Anchor.")
        print("   The smoothed score is likely much closer to the truth.")
    else:
        print(">> VERDICT: CAUTION. The 5 items are too noisy or novel.")
        print("   Your Anchor Theta doesn't explain these results well.")

if __name__ == "__main__":
    run_real_world_smoothing()