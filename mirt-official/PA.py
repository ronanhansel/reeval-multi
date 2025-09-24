import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import os

# --- 1. Define the function for a SINGLE simulation ---
def run_single_pca_simulation(matrix_shape):
    """
    Generates a random NORMALIZED matrix, runs PCA, and returns the eigenvalues.
    """
    # Create random data
    random_matrix = np.random.normal(0, 1, matrix_shape)
    # The random data is already 'standardized' by definition (mean=0, std=1)
    
    n_components = matrix_shape[0] - 1
    # Note: No need for 'randomized' solver on this size of random data, 
    # the standard 'auto' is efficient.
    pca_random = PCA(n_components=n_components)
    pca_random.fit(random_matrix)
    
    # Return the raw eigenvalues (explained_variance_)
    return pca_random.explained_variance_

# ===================================================================
# --- Main Script ---
# ===================================================================

# --- 2. Load and Prepare Data ---
print("Loading and preparing data...")
shapeMatrix = pd.read_pickle("../data/resmat.pkl")
shapeMatrix.fillna(0, inplace=True)

# --- 3. NORMALIZE THE DATA ---
# This is the critical step. We standardize the data to have a mean of 0 and a variance of 1.
# This is equivalent to running PCA on the correlation matrix.
print("Normalizing data (this may take a moment)...")
scaler = StandardScaler()
normalized_shapeMatrix = scaler.fit_transform(shapeMatrix)
print("Data normalized successfully.")

# --- 4. Run PCA on your ACTUAL NORMALIZED data ---
print("Running PCA on actual data...")
n_components = normalized_shapeMatrix.shape[0] - 1
pca = PCA(n_components=n_components)
pca.fit(normalized_shapeMatrix)
# Get the raw eigenvalues for the real data
real_eigenvalues = pca.explained_variance_

# --- 5. Run simulations IN PARALLEL on all CPU cores ---
n_simulations = 10000
print(f"Running {n_simulations} simulations in parallel...")
results = Parallel(n_jobs=-1)(
    delayed(run_single_pca_simulation)(normalized_shapeMatrix.shape) for i in tqdm(range(n_simulations))
)

# --- 6. Aggregate results and find the number of factors ---
all_random_eigenvalues = np.array(results)
percentile_eigenvalues = np.percentile(all_random_eigenvalues, 95, axis=0)

# Find the number of factors from Parallel Analysis
n_factors_parallel = sum(real_eigenvalues > percentile_eigenvalues)

# Find the number of factors from the (now valid) Kaiser Criterion
n_factors_kaiser = sum(real_eigenvalues > 1)

print("\n--- Analysis Complete ---")
print(f"✅ Parallel Analysis (95th percentile rule) recommends: {n_factors_parallel} factors.")
print(f"✅ Kaiser Criterion (eigenvalue > 1 rule) recommends: {n_factors_kaiser} factors.")
np.save("../result/real_eigenvalues.npy", real_eigenvalues)
np.save("../result/percentile_eigenvalues.npy", percentile_eigenvalues)
np.save("../result/n_factors_parallel.npy", np.array([n_factors_parallel]))