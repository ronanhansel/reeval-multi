import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.metrics import roc_auc_score
import jax
import pytensor

# --------------------------------------------------------------------------
# A) Configuration: Set PyMC to use JAX and float32 for performance
# --------------------------------------------------------------------------
pytensor.config.floatX = "float32" # <-- OPTIMIZATION 1
pytensor.config.mode = "JAX"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

# --------------------------------------------------------------------------
# B) Step 1: Load and prepare data (no changes needed)
# --------------------------------------------------------------------------
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
train_ys = resmat.values[train_rows, train_cols].astype(np.int32)
test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
test_ys = resmat.values[test_rows, test_cols].astype(np.int32)
n_persons, n_items = resmat.shape
K = 7
print(f"n_persons: {n_persons}, n_items: {n_items}, n_train_obs: {len(train_ys)}")
print(f"Using {K} latent dimensions.")

# --------------------------------------------------------------------------
# C) Step 2: Build the MIRT model with a Non-Centered Parameterization
# --------------------------------------------------------------------------
with pm.Model() as mirt_model_optimized:
    # <-- OPTIMIZATION 2: Non-centered priors
    theta_offset = pm.Normal("theta_offset", mu=0.0, sigma=1.0, shape=(n_persons, K))
    a_offset = pm.Normal("a_offset", mu=0.0, sigma=1.0, shape=(n_items, K))
    b_offset = pm.Normal("b_offset", mu=0.0, sigma=1.0, shape=(n_items,))

    # Reconstruct actual parameters
    theta = pm.Deterministic("theta", theta_offset)
    a = pm.Deterministic("a", a_offset)
    b = pm.Deterministic("b", b_offset)

    theta_r = theta[train_rows]
    a_c = a[train_cols]
    dot = pm.math.sum(theta_r * a_c, axis=1)
    logits = dot - b[train_cols]
    
    y = pm.Bernoulli("y_obs", p=pm.math.sigmoid(logits), observed=train_ys)

# --------------------------------------------------------------------------
# D) Step 3: Fit the model using fewer MCMC steps
# --------------------------------------------------------------------------
print(f"\nStarting MCMC sampling on {jax.default_backend()}...")
with mirt_model_optimized:
    idata = pm.sample(
        draws=1000, # <-- OPTIMIZATION 3
        tune=1000,  # <-- OPTIMIZATION 3
        chains=4,
        nuts_sampler="numpyro",
        chain_method="vectorized",
        cores=1,
        progressbar=True,
    )

# --- SAVE THE RESULTS ---
# This is the crucial step. Save the idata object to a file.
print("Saving InferenceData to file...")
idata.to_netcdf("../result/mirt_mcmc_converged.nc") 
print("Save complete.")

# --------------------------------------------------------------------------
# E) Step 4: Evaluate on the test set using vectorized operations
# --------------------------------------------------------------------------
print("Evaluating model on the test set...")

# Extract posterior samples from the InferenceData object
# `stack` combines the chains and draws into a single dimension for easier processing
posterior = idata.posterior.stack(sample=("chain", "draw"))

theta_s = posterior["theta"].values  # shape: (n_persons, K, n_samples)
a_s = posterior["a"].values          # shape: (n_items, K, n_samples)
b_s = posterior["b"].values          # shape: (n_items, n_samples)

# --- Vectorized Posterior Predictive Calculation ---
# This avoids a slow Python loop and uses efficient NumPy broadcasting.
# 1. Select the parameters for the specific test pairs
theta_test = theta_s[test_rows, :, :]  # shape: (n_test, K, n_samples)
a_test = a_s[test_cols, :, :]          # shape: (n_test, K, n_samples)
b_test = b_s[test_cols, :]             # shape: (n_test, n_samples)

# 2. Calculate logits for all test pairs across all posterior samples
# The sum is over the K latent dimensions
logits_test = np.sum(theta_test * a_test, axis=1) - b_test

# 3. Apply sigmoid to get probabilities
probs = 1.0 / (1.0 + np.exp(-logits_test))  # shape: (n_test, n_samples)

# 4. Average probabilities across all samples to get the final prediction
pred_mean = probs.mean(axis=1)

auc = roc_auc_score(test_ys, pred_mean)
print("\nMIRT posterior mean AUC on test set:", auc)

# Optional: dump ArviZ summary
print("\nArviZ summary for item parameters (first 5 items):")
print(az.summary(idata, var_names=["a", "b"], hdi_prob=0.95, coords={'b_dim_0': range(5), 'a_dim_0': range(5)}))