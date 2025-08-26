import numpy as np
import torch
from tqdm import tqdm

# ===================================================================
# == 1. Setup PyTorch and GPU Device
# ===================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found. Using CUDA.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

# ===================================================================
# == 2. Load Data and Move to GPU
# ===================================================================
print("\nStep 1: Loading data and moving to GPU...")
item_factors = torch.from_numpy(np.nan_to_num(np.load("../data/all_item_factors.npy"), nan=0)).float().to(device)
true_thetas_observed = torch.from_numpy(np.nan_to_num(np.load("../data/subject_scores.npy"), nan=0)).float().to(device)

N_ITEMS, N_FACTORS = item_factors.shape
print(f"  - Loaded {N_ITEMS} items with {N_FACTORS} factors.")

# ===================================================================
# == 3. Generate New Simulated Users
# ===================================================================
N_SIM_PEOPLE = 5
theta_mean = torch.mean(true_thetas_observed, axis=0)
theta_cov = torch.from_numpy(np.cov(true_thetas_observed.cpu().numpy(), rowvar=False)).float().to(device)
prior_dist = torch.distributions.MultivariateNormal(theta_mean, theta_cov)

torch.manual_seed(42)
true_thetas_simulated = prior_dist.sample((N_SIM_PEOPLE,))
print(f"\nStep 2: Generated {N_SIM_PEOPLE} new simulated users.")

# ===================================================================
# == 4. CAT Configuration
# ===================================================================
SEM_THRESHOLD = 0.45
MAX_TEST_LENGTH = 100
ITEM_SAMPLE_SIZE = 2000

# ===================================================================
# == 5. Core CAT Functions (These are correct)
# ===================================================================

def simulate_response_torch(true_theta, item_factor):
    logit = torch.dot(true_theta, item_factor)
    prob_correct = torch.sigmoid(logit)
    return torch.bernoulli(prob_correct).item()

def select_next_item_fisher_trace(current_theta, candidate_factors, candidate_indices):
    logits = torch.matmul(candidate_factors, current_theta)
    probs = torch.sigmoid(logits)
    info_weights = probs * (1 - probs)
    item_infos = info_weights * torch.sum(candidate_factors**2, dim=1)
    best_local_index = torch.argmax(item_infos)
    return candidate_indices[best_local_index]

def estimate_theta_map(answered_responses, answered_factors, initial_theta, prior):
    theta = initial_theta.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([theta], lr=0.75, max_iter=30)
    
    def closure():
        optimizer.zero_grad()
        logits = torch.mv(answered_factors, theta)
        log_likelihood = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, answered_responses, reduction='sum'
        )
        log_prior = prior.log_prob(theta)
        loss = log_likelihood - log_prior
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach()

def calculate_sem_torch(answered_item_factors, current_theta):
    if answered_item_factors.shape[0] < N_FACTORS:
        return torch.tensor(float('inf'), device=device)

    logits = torch.mv(answered_item_factors, current_theta)
    probs = torch.sigmoid(logits)
    weights = probs * (1 - probs)
    information_matrices = weights[:, None, None] * torch.bmm(answered_item_factors.unsqueeze(2), answered_item_factors.unsqueeze(1))
    information_matrix = torch.sum(information_matrices, dim=0)

    try:
        var_cov_matrix = torch.linalg.inv(information_matrix)
        variances = torch.diag(var_cov_matrix)
        if torch.any(variances < 0): return torch.tensor(float('inf'), device=device)
        return torch.sqrt(variances)
    except torch.linalg.LinAlgError:
        return torch.tensor(float('inf'), device=device)

# ===================================================================
# == 6. Main Simulation Loop (With Critical Bug Fix)
# ===================================================================
print(f"\nStep 3: Running stable CAT simulation...")
estimated_thetas = torch.zeros_like(true_thetas_simulated)
final_test_lengths = np.zeros(N_SIM_PEOPLE)

for i in tqdm(range(N_SIM_PEOPLE)):
    person_true_theta = true_thetas_simulated[i]
    current_theta_estimate = theta_mean.clone()
    
    # --- CRITICAL FIX: Initialize a stable list for responses ---
    answered_item_indices = []
    answered_responses_list = [] # Store responses here
    
    sem = torch.tensor(float('inf'), device=device)
    while torch.any(sem > SEM_THRESHOLD) and len(answered_item_indices) < MAX_TEST_LENGTH:
        candidate_indices_np = np.random.choice(
            np.setdiff1d(np.arange(N_ITEMS), answered_item_indices), 
            ITEM_SAMPLE_SIZE, 
            replace=False
        )
        candidate_indices = torch.from_numpy(candidate_indices_np).long().to(device)
        
        item_to_administer_idx = select_next_item_fisher_trace(
            current_theta_estimate, item_factors[candidate_indices], candidate_indices
        )
        
        # --- CRITICAL FIX: Simulate response ONCE and store it ---
        response = simulate_response_torch(person_true_theta, item_factors[item_to_administer_idx])
        answered_item_indices.append(item_to_administer_idx.item())
        answered_responses_list.append(response) # Add the stable response to our list
        
        # Convert the STABLE, CUMULATIVE history to tensors for estimation
        answered_factors_tensor = item_factors[answered_item_indices]
        answered_responses_tensor = torch.tensor(answered_responses_list, dtype=torch.float, device=device)

        current_theta_estimate = estimate_theta_map(
             answered_responses_tensor, answered_factors_tensor, current_theta_estimate, prior_dist
        )
        
        sem = calculate_sem_torch(answered_factors_tensor, current_theta_estimate)

    estimated_thetas[i] = current_theta_estimate
    final_test_lengths[i] = len(answered_item_indices)

print("\nSimulation complete!")

# ===================================================================
# == 7. Evaluation
# ===================================================================
print("\nStep 4: Evaluating the results...")
true_thetas_simulated_np = true_thetas_simulated.cpu().numpy()
estimated_thetas_np = estimated_thetas.cpu().numpy()

rmse = np.sqrt(np.mean((true_thetas_simulated_np - estimated_thetas_np)**2))
print(f"\nOverall RMSE: {rmse:.4f}")

print("\nCorrelation between True and Estimated Thetas (by dimension):")
for d in range(N_FACTORS):
    correlation = np.corrcoef(true_thetas_simulated_np[:, d], estimated_thetas_np[:, d])[0, 1]
    print(f"  - Dimension {d+1}: {correlation:.4f}")

print("\nTest Length Statistics:")
print(f"  - Average Test Length: {np.mean(final_test_lengths):.2f} items")
print(f"  - Minimum Test Length: {np.min(final_test_lengths):.0f} items")
print(f"  - Maximum Test Length: {np.max(final_test_lengths):.0f} items")