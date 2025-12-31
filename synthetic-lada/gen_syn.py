"""
Generate synthetic binary response matrices for LADA experiments.

Outputs:
1. General Data (Mixed Skills):
   - resmat_train.pkl (1000 items)
   - resmat_test.pkl  (100 items)

2. Specialized Data (Biased Skills):
   - resmat_math.pkl        (High Dim 0)
   - resmat_coding.pkl      (High Dim 1)
   - resmat_creative.pkl    (High Dim 2)
   - resmat_datascience.pkl (High Dim 0 + Dim 1)

Model: y_obs = d + w*theta
"""

import numpy as np
import pandas as pd
import os
from scipy.special import expit  # sigmoid function

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_models = 20           # Number of users/models (rows)
n_items_train = 1000    # General training items
n_items_test = 100      # General & Specialized test items
k = 3                   # Latent dimensions
alpha_general = 1.1     # Dirichlet parameter for general datasets

# Create data directory
os.makedirs('./data', exist_ok=True)

# ==============================================================================
# 1. Generate Shared User Profiles (Theta)
# ==============================================================================
# This is the "Ground Truth" ability of the 20 models.
# It stays constant across ALL datasets.
theta = np.random.randn(n_models, k)
print(f"Generated global Theta: {theta.shape}")

# ==============================================================================
# 2. Helper Function for Generation
# ==============================================================================
def generate_dataset(name, n_items, alpha_vec):
    """
    Generates a response matrix based on specific Dirichlet alphas.
    name: filename suffix
    alpha_vec: list of concentration parameters for Dirichlet (controls bias)
    """
    print(f"\nGenerating {name}...")
    
    # 1. Generate Difficulty (d)
    d = np.random.randn(n_items)
    
    # 2. Generate Discrimination/Weights (w) based on bias
    # If alpha_vec is scalar, replicate it k times (General case)
    if isinstance(alpha_vec, (float, int)):
        alphas = [alpha_vec] * k
    else:
        alphas = alpha_vec
        
    w = np.random.dirichlet(alphas, size=n_items)
    
    # 3. Calculate Responses
    # Logits = d + (Theta * W)
    linear_pred = d[np.newaxis, :] + theta @ w.T
    prob = expit(linear_pred)
    resmat = (prob > 0.5).astype(int)
    
    # 4. Save to DataFrame
    df = pd.DataFrame(
        resmat,
        index=[f"model_{i}" for i in range(n_models)],
        columns=[f"item_{i}" for i in range(n_items)]
    )
    
    path = f"./data/resmat_{name}.pkl"
    df.to_pickle(path)
    
    # Stats
    print(f"  > Saved to {path}")
    print(f"  > Shape: {resmat.shape}")
    print(f"  > Avg Score: {resmat.mean():.3f}")
    print(f"  > Avg Weights (Dim Loadings): {w.mean(axis=0).round(2)}")
    
    return d, w

# ==============================================================================
# 3. Generate Datasets
# ==============================================================================

ground_truth = {
    'theta': theta,
    'k': k,
    'datasets': {}
}

# --- A. General Datasets (Training & Test) ---
# Used to train the LADA model and establish the baseline
d_tr, w_tr = generate_dataset("train", n_items_train, alpha_general)
ground_truth['datasets']['train'] = {'d': d_tr, 'w': w_tr}

d_te, w_te = generate_dataset("test", n_items_test, alpha_general)
ground_truth['datasets']['test'] = {'d': d_te, 'w': w_te}


# --- B. Specialized Datasets (Downstream Benchmarks) ---
# Used to test Transfer Learning. 
# We bias the Dirichlet alphas to create "specialist" items.

# 1. Math Benchmark (Heavily Dim 0)
d_math, w_math = generate_dataset("math", n_items_test, [6.0, 0.5, 0.5])
ground_truth['datasets']['math'] = {'d': d_math, 'w': w_math}

# 2. Coding Benchmark (Heavily Dim 1)
d_code, w_code = generate_dataset("coding", n_items_test, [0.5, 6.0, 0.5])
ground_truth['datasets']['coding'] = {'d': d_code, 'w': w_code}

# 3. Creative Writing Benchmark (Heavily Dim 2)
d_creat, w_creat = generate_dataset("creative", n_items_test, [0.5, 0.5, 6.0])
ground_truth['datasets']['creative'] = {'d': d_creat, 'w': w_creat}

# 4. Data Science Benchmark (Mix of Dim 0 and Dim 1)
d_ds, w_ds = generate_dataset("datascience", n_items_test, [4.0, 4.0, 0.2])
ground_truth['datasets']['datascience'] = {'d': d_ds, 'w': w_ds}


# ==============================================================================
# 4. Save Ground Truth
# ==============================================================================
np.save('./data/ground_truth.npy', ground_truth)
print("\nSaved ground_truth.npy")
print("Generation Complete.")