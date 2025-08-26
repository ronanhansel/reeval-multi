import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================================
# 1. LOAD & STANDARDIZE SVD DATA
# =============================================================================
print("Step 1: Loading and standardizing SVD factors...")

# --- Load your SVD factors (from CSV) ---
item_factors_raw = pd.read_csv("../data/all_item_factors.csv").to_numpy()
true_thetas_observed_raw = pd.read_csv("../data/subject_scores.csv").to_numpy()

# Clean any potential NaN/Inf values
item_factors_raw = np.nan_to_num(item_factors_raw, nan=0.0)
true_thetas_observed_raw = np.nan_to_num(true_thetas_observed_raw, nan=0.0)

# --- CRITICAL FIX: Standardize the SVD Factors for Numerical Stability ---
scaler_items = StandardScaler()
scaler_thetas = StandardScaler()

item_factors = scaler_items.fit_transform(item_factors_raw)
true_thetas_observed = scaler_thetas.fit_transform(true_thetas_observed_raw)
# -------------------------------------------------------------------------

# --- Dynamically configure parameters ---
N_ITEMS, N_FACTORS = item_factors.shape
N_SIM_PEOPLE = 5
np.random.seed(42)

print(f"  - Loaded and standardized {N_ITEMS} items with {N_FACTORS} factors.")

# --- Generate simulated users based on the STANDARDIZED distribution ---
theta_mean = np.mean(true_thetas_observed, axis=0)
theta_cov = np.cov(true_thetas_observed, rowvar=False)
true_thetas_simulated = np.random.multivariate_normal(mean=theta_mean, cov=theta_cov, size=N_SIM_PEOPLE)

print(f"Step 2: Generated {N_SIM_PEOPLE} new simulated users.")

# --- CAT Configuration ---
MAX_TEST_LENGTH = 100
SEM_THRESHOLD = 0.5
ITEM_SAMPLE_SIZE = 1500

# =============================================================================
# 2. CUSTOM SVD-BASED CAT FUNCTIONS
# =============================================================================

def svd_prob(theta, item_factor):
    """Calculates probability of correct response using the SVD model."""
    logit = np.dot(theta, item_factor)
    return expit(logit)

def simulate_response(true_theta, item_factor):
    """Simulates a 0/1 response."""
    prob = svd_prob(true_theta, item_factor)
    return np.random.binomial(1, prob)

def calculate_fisher_matrix(item_factors_subset, theta):
    """Calculates the Fisher Information Matrix for a set of items."""
    info_matrix = np.zeros((N_FACTORS, N_FACTORS))
    if item_factors_subset.shape[0] == 0:
        return info_matrix
    
    for factor in item_factors_subset:
        prob = svd_prob(theta, factor)
        weight = prob * (1.0 - prob)
        info_matrix += weight * np.outer(factor, factor)
    return info_matrix

def estimate_theta_map(responses, factors, initial_theta, prior_mean, prior_cov_inv):
    """Estimates theta using MAP with a numerical optimizer."""
    def neg_log_posterior(theta, responses, factors, prior_mean, prior_cov_inv):
        logits = factors @ theta
        # Numerically stable log-likelihood for Bernoulli
        log_likelihood = np.sum(responses * logits - np.log(1 + np.exp(logits)))
        
        diff = theta - prior_mean
        log_prior = -0.5 * diff.T @ prior_cov_inv @ diff
        
        return -(log_likelihood + log_prior)

    result = minimize(
        fun=neg_log_posterior,
        x0=initial_theta,
        args=(responses, factors, prior_mean, prior_cov_inv),
        method='L-BFGS-B'
    )
    return result.x

def select_next_item_mdet(current_theta, answered_factors, candidate_factors, candidate_indices):
    """Selects the next item using the Maximum Determinant (MDET) method."""
    current_info_matrix = calculate_fisher_matrix(answered_factors, current_theta)
    identity_matrix = np.eye(N_FACTORS) * 1e-5
    
    determinants = []
    for factor in candidate_factors:
        prob = svd_prob(current_theta, factor)
        weight = prob * (1.0 - prob)
        candidate_info = weight * np.outer(factor, factor)
        potential_total_info = current_info_matrix + candidate_info
        determinants.append(np.linalg.det(potential_total_info + identity_matrix))
        
    best_local_index = np.argmax(determinants)
    return candidate_indices[best_local_index]

# =============================================================================
# 3. MAIN SIMULATION LOOP
# =============================================================================
print("Step 3: Running M-CAT simulation...")

estimated_thetas = np.zeros_like(true_thetas_simulated)
final_test_lengths = np.zeros(N_SIM_PEOPLE)
prior_cov_inverse = np.linalg.inv(theta_cov)

for p in tqdm(range(N_SIM_PEOPLE)):
    person_true_theta = true_thetas_simulated[p]
    current_theta_estimate = np.copy(theta_mean)
    
    answered_indices = []
    answered_responses = []
    
    sem = np.inf
    while np.any(sem > SEM_THRESHOLD) and len(answered_indices) < MAX_TEST_LENGTH:
        remaining_indices = np.setdiff1d(np.arange(N_ITEMS), answered_indices)
        candidate_indices = np.random.choice(
            remaining_indices, 
            size=min(ITEM_SAMPLE_SIZE, len(remaining_indices)), 
            replace=False
        )
        
        answered_factors_matrix = item_factors[answered_indices]
        
        item_to_administer_idx = select_next_item_mdet(
            current_theta_estimate, answered_factors_matrix, item_factors[candidate_indices], candidate_indices
        )
        
        response = simulate_response(person_true_theta, item_factors[item_to_administer_idx])
        
        answered_indices.append(item_to_administer_idx)
        answered_responses.append(response)
        
        current_theta_estimate = estimate_theta_map(
            np.array(answered_responses), 
            item_factors[answered_indices], 
            current_theta_estimate, 
            theta_mean, 
            prior_cov_inverse
        )
        
        total_info_matrix = calculate_fisher_matrix(item_factors[answered_indices], current_theta_estimate)
        try:
            var_cov = np.linalg.inv(total_info_matrix)
            if np.any(np.diag(var_cov) < 0):
                sem = np.inf
            else:
                sem = np.sqrt(np.diag(var_cov))
        except np.linalg.LinAlgError:
            sem = np.inf

    estimated_thetas[p] = current_theta_estimate
    final_test_lengths[p] = len(answered_indices)

print("Simulation complete!")

# =============================================================================
# 4. EVALUATION
# =============================================================================
print("\nStep 4: Evaluating the results...")

rmse = np.sqrt(np.mean((true_thetas_simulated - estimated_thetas)**2))
print(f"\nOverall RMSE: {rmse:.4f}")

print("\nCorrelation between True and Estimated Thetas (by dimension):")
for d in range(N_FACTORS):
    correlation = np.corrcoef(true_thetas_simulated[:, d], estimated_thetas[:, d])[0, 1]
    print(f"  - Dimension {d+1}: {correlation:.4f}")

print("\nTest Length Statistics:")
print(f"  - Average Test Length: {np.mean(final_test_lengths):.2f} items")
print(f"  - Minimum Test Length: {np.min(final_test_lengths):.0f} items")
print(f"  - Maximum Test Length: {np.max(final_test_lengths):.0f} items")