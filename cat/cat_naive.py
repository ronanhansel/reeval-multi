import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from tqdm import tqdm
import os
import sys
sys.path.append('../mirt-official') 
from load_params import load_and_rotate

# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

resmat = pd.read_pickle("../data/resmat.pkl")

theta, a, b = load_and_rotate("../mirt-official/output/mirt_model_k19.pt")

# Step 1: Get the final theta tensor into a NumPy array
theta_abilities = theta
# Step 2: Create a labeled pandas DataFrame
# Use the model names from your original resmat for the index
model_names = resmat.index
factor_names = [f'F{i+1}' for i in range(theta.shape[1])]
ability_df = pd.DataFrame(theta_abilities, index=model_names, columns=factor_names)

item_factors_raw = a
item_difficulty_raw = b
true_thetas_observed_raw = theta_abilities

item_factors_raw = np.nan_to_num(item_factors_raw, nan=0.0)
true_thetas_observed_raw = np.nan_to_num(true_thetas_observed_raw, nan=0.0)

item_factors = item_factors_raw
item_difficulties = item_difficulty_raw
true_thetas_observed = true_thetas_observed_raw

N_ITEMS, N_FACTORS = item_factors.shape
N_SIM_PEOPLE = 1  # Run 200 simulations
np.random.seed(42)

print(f"  - Loaded and standardized {N_ITEMS} items with {N_FACTORS} factors.")

theta_mean = np.mean(true_thetas_observed, axis=0)
theta_cov = np.cov(true_thetas_observed, rowvar=False)

print(f"\nStep 2: Will simulate {N_SIM_PEOPLE} users to test the naive CAT.")

# --- CAT Configuration ---
MAX_TEST_LENGTH = 1500
ITEM_SAMPLE_SIZE = 78712

# =============================================================================
# 2. CUSTOM SVD-BASED CAT FUNCTIONS
# =============================================================================

def correct_prob(theta, item_factor, item_difficulty):
    logit = np.dot(theta, item_factor) - item_difficulty
    return expit(logit)

def simulate_response(true_theta, item_factor, item_difficulty):
    prob = correct_prob(true_theta, item_factor, item_difficulty)
    return np.random.binomial(1, prob)

def calculate_fisher_matrix(item_factors_subset, theta, item_difficulties_subset):
    info_matrix = np.zeros((N_FACTORS, N_FACTORS))
    if item_factors_subset.shape[0] == 0: return info_matrix
    for i, factor in enumerate(item_factors_subset):
        prob = correct_prob(theta, factor, item_difficulties_subset[i])
        weight = prob * (1.0 - prob)
        info_matrix += weight * np.outer(factor, factor)
    return info_matrix

def estimate_theta_map(responses, factors, item_difficulties_subset, initial_theta, prior_mean, prior_cov_inv):
    def neg_log_posterior(theta, responses, factors, item_difficulties_subset, prior_mean, prior_cov_inv):
        logits = factors @ theta - item_difficulties_subset
        log_likelihood = np.sum(responses * logits - np.log(1 + np.exp(logits)))
        diff = theta - prior_mean
        log_prior = -0.5 * diff.T @ prior_cov_inv @ diff
        return -(log_likelihood + log_prior)

    result = minimize(fun=neg_log_posterior, x0=initial_theta, args=(responses, factors, item_difficulties_subset, prior_mean, prior_cov_inv), method='L-BFGS-B')
    return result.x

def select_next_item_naive(candidate_indices):
    """Naive item selection: simply pick randomly from candidates."""
    return np.random.choice(candidate_indices)

# --- NEW: Function to calculate reliability from SEM ---
def calculate_reliability_from_sem(sem_vector, theta_population_variance):
    """Calculates reliability for each dimension."""
    # Since the data is standardized, the variance for each dimension is approx 1.
    reliability = 1 - (sem_vector**2 / theta_population_variance)
    # Clamp values to be between 0 and 1
    return np.clip(reliability, 0, 1)

# =============================================================================
# 3. RUN SIMULATION FOR MULTIPLE PEOPLE
# =============================================================================
print(f"\nStep 3: Running naive CAT simulation for {N_SIM_PEOPLE} people...")

prior_cov_inverse = np.linalg.inv(theta_cov)
theta_pop_variance = np.diag(theta_cov)  # Get the variance for each dimension

all_results_history = []

for person_idx in tqdm(range(N_SIM_PEOPLE), desc="Simulating people"):
    # Generate a new person for each simulation
    true_theta_simulated = np.random.multivariate_normal(mean=theta_mean, cov=theta_cov, size=1)[0]
    
    current_theta_estimate = np.copy(theta_mean)
    answered_indices = []
    answered_responses = []
    
    person_results = []
    
    for step in range(MAX_TEST_LENGTH):
        remaining_indices = np.setdiff1d(np.arange(N_ITEMS), answered_indices)
        
        if len(remaining_indices) == 0:
            break
            
        candidate_indices = np.random.choice(
            remaining_indices, size=min(ITEM_SAMPLE_SIZE, len(remaining_indices)), replace=False
        )
        
        # Use naive (random) selection instead of intelligent selection
        item_to_administer_idx = select_next_item_naive(candidate_indices)
        
        response = simulate_response(true_theta_simulated, item_factors[item_to_administer_idx], item_difficulties[item_to_administer_idx])
        
        answered_indices.append(item_to_administer_idx)
        answered_responses.append(response)
        
        current_theta_estimate = estimate_theta_map(
            np.array(answered_responses), 
            item_factors[answered_indices], 
            item_difficulties[answered_indices],
            current_theta_estimate, 
            theta_mean, 
            prior_cov_inverse
        )
        
        total_info_matrix = calculate_fisher_matrix(item_factors[answered_indices], current_theta_estimate, item_difficulties[answered_indices])
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
            
        # Store the results for this step, including reliability and person ID
        step_results = {'person_id': person_idx + 1, 'step': step + 1}
        for d in range(N_FACTORS):
            step_results[f'theta_{d+1}'] = current_theta_estimate[d]
            step_results[f'sem_{d+1}'] = sem[d] if np.isfinite(sem[d]) else None
            step_results[f'reliability_{d+1}'] = reliability[d] if np.isfinite(reliability[d]) else None
        person_results.append(step_results)
    
    all_results_history.extend(person_results)

print("Simulation complete!")

# =============================================================================
# 4. EXPORT RESULTS
# =============================================================================
output_dir = "../result"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "naive_cat_history_with_reliability.csv")

results_df = pd.DataFrame(all_results_history)
results_df.to_csv(output_path, index=False)

print(f"\nStep 4: Results exported successfully to {output_path}")

# Calculate and show average final results across all people
print("\n--- Average Final Results Across All Simulations ---")
final_step_results = results_df.groupby('person_id').last()
avg_final_results = {
    'N_Simulations': N_SIM_PEOPLE,
    'Avg_Final_SEM': final_step_results[[f'sem_{d+1}' for d in range(N_FACTORS)]].mean().mean(),
    'Avg_Final_Reliability': final_step_results[[f'reliability_{d+1}' for d in range(N_FACTORS)]].mean().mean()
}
print(pd.Series(avg_final_results).round(4))
