import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================
print("Step 1: Loading and standardizing SVD factors...")

item_factors_raw = pd.read_csv("../data/all_item_factors.csv").to_numpy()
true_thetas_observed_raw = pd.read_csv("../data/subject_scores.csv").to_numpy()

item_factors_raw = np.nan_to_num(item_factors_raw, nan=0.0)
true_thetas_observed_raw = np.nan_to_num(true_thetas_observed_raw, nan=0.0)

scaler_items = StandardScaler()
scaler_thetas = StandardScaler()
item_factors = scaler_items.fit_transform(item_factors_raw)
true_thetas_observed = scaler_thetas.fit_transform(true_thetas_observed_raw)

N_ITEMS, N_FACTORS = item_factors.shape
N_SIM_PEOPLE = 1
np.random.seed(42)

print(f"  - Loaded and standardized {N_ITEMS} items with {N_FACTORS} factors.")

theta_mean = np.mean(true_thetas_observed, axis=0)
theta_cov = np.cov(true_thetas_observed, rowvar=False)
true_theta_simulated = np.random.multivariate_normal(mean=theta_mean, cov=theta_cov, size=N_SIM_PEOPLE)[0]

print(f"\nStep 2: Generated 1 new simulated user to test the CAT.")

# --- CAT Configuration ---
MAX_TEST_LENGTH = 500
ITEM_SAMPLE_SIZE = 1500

# =============================================================================
# 2. CUSTOM SVD-BASED CAT FUNCTIONS
# =============================================================================

def svd_prob(theta, item_factor):
    logit = np.dot(theta, item_factor)
    return expit(logit)

def simulate_response(true_theta, item_factor):
    prob = svd_prob(true_theta, item_factor)
    return np.random.binomial(1, prob)

def calculate_fisher_matrix(item_factors_subset, theta):
    info_matrix = np.zeros((N_FACTORS, N_FACTORS))
    if item_factors_subset.shape[0] == 0: return info_matrix
    for factor in item_factors_subset:
        prob = svd_prob(theta, factor)
        weight = prob * (1.0 - prob)
        info_matrix += weight * np.outer(factor, factor)
    return info_matrix

def estimate_theta_map(responses, factors, initial_theta, prior_mean, prior_cov_inv):
    def neg_log_posterior(theta, responses, factors, prior_mean, prior_cov_inv):
        logits = factors @ theta
        log_likelihood = np.sum(responses * logits - np.log(1 + np.exp(logits)))
        diff = theta - prior_mean
        log_prior = -0.5 * diff.T @ prior_cov_inv @ diff
        return -(log_likelihood + log_prior)

    result = minimize(fun=neg_log_posterior, x0=initial_theta, args=(responses, factors, prior_mean, prior_cov_inv), method='L-BFGS-B')
    return result.x

def select_next_item_mdet(current_theta, answered_factors, candidate_factors, candidate_indices):
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

# --- NEW: Function to calculate reliability from SEM ---
def calculate_reliability_from_sem(sem_vector, theta_population_variance):
    """Calculates reliability for each dimension."""
    # Since the data is standardized, the variance for each dimension is approx 1.
    reliability = 1 - (sem_vector**2 / theta_population_variance)
    # Clamp values to be between 0 and 1
    return np.clip(reliability, 0, 1)

# =============================================================================
# 3. RUN SIMULATION FOR ONE PERSON
# =============================================================================
print("\nStep 3: Running M-CAT simulation for one person...")

current_theta_estimate = np.copy(theta_mean)
answered_indices = []
answered_responses = []
prior_cov_inverse = np.linalg.inv(theta_cov)
theta_pop_variance = np.diag(theta_cov) # Get the variance for each dimension

results_history = []

for step in tqdm(range(MAX_TEST_LENGTH)):
    remaining_indices = np.setdiff1d(np.arange(N_ITEMS), answered_indices)
    candidate_indices = np.random.choice(
        remaining_indices, size=min(ITEM_SAMPLE_SIZE, len(remaining_indices)), replace=False
    )
    
    answered_factors_matrix = item_factors[answered_indices]
    
    item_to_administer_idx = select_next_item_mdet(
        current_theta_estimate, answered_factors_matrix, item_factors[candidate_indices], candidate_indices
    )
    
    response = simulate_response(true_theta_simulated, item_factors[item_to_administer_idx])
    
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
            sem = np.full(N_FACTORS, np.inf)
        else:
            sem = np.sqrt(np.diag(var_cov))
    except np.linalg.LinAlgError:
        sem = np.full(N_FACTORS, np.inf)
        
    # --- NEW: Calculate reliability at this step ---
    reliability = calculate_reliability_from_sem(sem, theta_pop_variance)
        
    # Store the results for this step, including reliability
    step_results = {'step': step + 1}
    for d in range(N_FACTORS):
        step_results[f'theta_{d+1}'] = current_theta_estimate[d]
        step_results[f'sem_{d+1}'] = sem[d] if np.isfinite(sem[d]) else None
        step_results[f'reliability_{d+1}'] = reliability[d] if np.isfinite(reliability[d]) else None
    results_history.append(step_results)

print("Simulation complete!")

# =============================================================================
# 4. EXPORT RESULTS
# =============================================================================
output_dir = "../result"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "single_person_cat_history_with_reliability.csv")

results_df = pd.DataFrame(results_history)
results_df.to_csv(output_path, index=False)

print(f"\nStep 4: Results exported successfully to {output_path}")

print("\n--- Final Estimate vs. True Ability ---")
final_results = pd.DataFrame({
    'Dimension': [f'Dim {d+1}' for d in range(N_FACTORS)],
    'True_Ability': true_theta_simulated,
    'Estimated_Ability': current_theta_estimate,
    'Final_SEM': sem,
    'Final_Reliability': reliability
})
print(final_results.round(4))