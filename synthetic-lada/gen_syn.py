"""
Generate synthetic binary response matrix, resmat_train.pkl of 20 rows and 1000 columns
y_obs = d + w*theta
where theta and w have the same number of dimensions k=3
d, theta follows a standard normal distribution prior
w follows a Dirichlet distribution prior with concentration parameter alpha=1.1

also resmat_test.pkl of 20 rows and 100 columns using the same theta but with different d and w (same prior)
Save it to ./data/
"""

import numpy as np
import pandas as pd
import os
from scipy.special import expit  # sigmoid function

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_models = 20  # number of models (rows)
n_items_train = 1000  # number of items for training (columns)
n_items_test = 100  # number of items for testing (columns)
k = 3  # number of dimensions
alpha = 1.1  # Dirichlet concentration parameter

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Generate theta (model abilities) - shared between train and test
# theta: (n_models, k)
theta = np.random.randn(n_models, k)

print(f"Generated theta with shape: {theta.shape}")

# ============= TRAINING DATA =============
# Generate d (item difficulties) for training
d_train = np.random.randn(n_items_train)

# Generate w (item discrimination vectors) for training
# w: (n_items_train, k) - each row is a k-dimensional vector from Dirichlet
w_train = np.random.dirichlet([alpha] * k, size=n_items_train)

# Calculate linear predictor: y_obs = d + w*theta
# For each item i and model m: y_obs[m, i] = d[i] + sum(w[i, :] * theta[m, :])
linear_pred_train = d_train[np.newaxis, :] + theta @ w_train.T  # (n_models, n_items_train)

# Convert to binary responses using sigmoid and threshold at 0.5
prob_train = expit(linear_pred_train)
resmat_train = (prob_train > 0.5).astype(int)

print(f"Generated training response matrix with shape: {resmat_train.shape}")
print(f"Training response rate: {resmat_train.mean():.3f}")

# Convert to DataFrame
resmat_train_df = pd.DataFrame(
    resmat_train,
    index=[f"model_{i}" for i in range(n_models)],
    columns=[f"item_{i}" for i in range(n_items_train)]
)

# Save training data
resmat_train_df.to_pickle('./data/resmat_train.pkl')
print("Saved resmat_train.pkl")

# ============= TEST DATA =============
# Generate new d and w for test data (using same priors)
d_test = np.random.randn(n_items_test)
w_test = np.random.dirichlet([alpha] * k, size=n_items_test)

# Calculate linear predictor using the SAME theta
linear_pred_test = d_test[np.newaxis, :] + theta @ w_test.T  # (n_models, n_items_test)

# Convert to binary responses
prob_test = expit(linear_pred_test)
resmat_test = (prob_test > 0.5).astype(int)

print(f"Generated test response matrix with shape: {resmat_test.shape}")
print(f"Test response rate: {resmat_test.mean():.3f}")

# Convert to DataFrame
resmat_test_df = pd.DataFrame(
    resmat_test,
    index=[f"model_{i}" for i in range(n_models)],
    columns=[f"item_{i}" for i in range(n_items_test)]
)

# Save test data
resmat_test_df.to_pickle('./data/resmat_test.pkl')
print("Saved resmat_test.pkl")

# Also save ground truth parameters for evaluation
ground_truth = {
    'theta': theta,
    'd_train': d_train,
    'w_train': w_train,
    'd_test': d_test,
    'w_test': w_test,
    'k': k,
    'alpha': alpha
}

np.save('./data/ground_truth.npy', ground_truth)
print("Saved ground_truth.npy")

print("\nGeneration complete!")
print(f"Training data: {n_models} models × {n_items_train} items")
print(f"Test data: {n_models} models × {n_items_test} items")
print(f"Dimensions: k={k}")
