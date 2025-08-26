# =============================================================================
# SVD-based Multidimensional Computerized Adaptive Test (M-CAT) Simulation
# =============================================================================
# This script uses custom functions based on the SVD dot-product model to
# simulate an M-CAT, comparing an adaptive strategy (MDET) vs. random.
# =============================================================================
# Install packages if needed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("Metrics")) install.packages("Metrics")
if (!require("MASS")) install.packages("MASS")
if (!require("readr")) install.packages("readr")


# Load required libraries
library(tidyverse)
library(Metrics)
library(MASS)
library(readr)
library(ggplot2)

# =============================================================================
# 1. LOAD SVD DATA & CONFIGURE
# =============================================================================
cat("Step 1: Loading SVD factors...\n")

# --- Load your SVD factors (saved as CSV) ---
item_factors_raw <- as.matrix(read_csv("../data/all_item_factors.csv"))
true_thetas_observed_raw <- as.matrix(read_csv("../data/subject_scores.csv"))

# Clean any potential NaN/Inf values
item_factors_raw[is.na(item_factors_raw) | !is.finite(item_factors_raw)] <- 0
true_thetas_observed_raw[is.na(true_thetas_observed_raw) | !is.finite(true_thetas_observed_raw)] <- 0

# --- NEW: Standardize the SVD Factors for Numerical Stability ---
item_factors <- scale(item_factors_raw)
true_thetas_observed <- scale(true_thetas_observed_raw)
# ----------------------------------------------------------------

# --- Dynamically configure parameters ---
N_ITEMS <- nrow(item_factors)
N_FACTORS <- ncol(item_factors)
N_SIM_PEOPLE <- 5
set.seed(42)

cat(sprintf("  - Loaded and standardized %d items with %d factors.\n", N_ITEMS, N_FACTORS))

# --- Generate simulated users based on the STANDARDIZED distribution ---
theta_mean <- colMeans(true_thetas_observed) # Will be approx. 0
theta_cov <- cov(true_thetas_observed)     # Will be approx. an identity matrix
true_thetas_simulated <- mvrnorm(n = N_SIM_PEOPLE, mu = theta_mean, Sigma = theta_cov)

cat(sprintf("Step 2: Generated %d new simulated users.\n", N_SIM_PEOPLE))

# --- CAT Configuration ---
MAX_TEST_LENGTH <- 100
SEM_THRESHOLD <- 0.5
ITEM_SAMPLE_SIZE <- 1500

# =============================================================================
# 2. CUSTOM SVD-BASED CAT FUNCTIONS (No changes in this section)
# =============================================================================

# SVD-based probability of a correct response
svd_prob <- function(theta, item_factor) {
  logit <- sum(theta * item_factor)
  return(1 / (1 + exp(-logit)))
}

# Simulate a response based on the SVD model
simulate_response <- function(true_theta, item_factor) {
  prob <- svd_prob(true_theta, item_factor)
  return(rbinom(1, 1, prob))
}

# Calculate the Fisher Information Matrix for a set of items
calculate_fisher_matrix <- function(item_factors_subset, theta) {
  info_matrix <- matrix(0, nrow = N_FACTORS, ncol = N_FACTORS)
  if (nrow(item_factors_subset) == 0) return(info_matrix)
  
  for (i in 1:nrow(item_factors_subset)) {
    factor <- item_factors_subset[i, ]
    prob <- svd_prob(theta, factor)
    weight <- prob * (1 - prob)
    info_matrix <- info_matrix + weight * outer(factor, factor)
  }
  return(info_matrix)
}

# The MAP estimator for theta
estimate_theta_map <- function(responses, factors, initial_theta, prior_mean, prior_cov_inv) {
  neg_log_posterior <- function(theta) {
    logits <- factors %*% theta
    log_likelihood <- sum(responses * logits - log(1 + exp(logits)))
    log_prior <- -0.5 * t(theta - prior_mean) %*% prior_cov_inv %*% (theta - prior_mean)
    return(-(log_likelihood + log_prior))
  }
  opt_result <- optim(par = initial_theta, fn = neg_log_posterior, method = "L-BFGS-B")
  return(opt_result$par)
}

# The Maximum Determinant (MDET) item selection strategy
select_next_item_mdet <- function(current_theta, answered_factors, candidate_factors, candidate_indices) {
  current_info_matrix <- calculate_fisher_matrix(answered_factors, current_theta)
  identity_matrix <- diag(N_FACTORS) * 1e-5
  
  determinants <- sapply(1:nrow(candidate_factors), function(i) {
    factor <- candidate_factors[i, ]
    prob <- svd_prob(current_theta, factor)
    weight <- prob * (1 - prob)
    candidate_info <- weight * outer(factor, factor)
    potential_total_info <- current_info_matrix + candidate_info
    det(potential_total_info + identity_matrix)
  })
  
  best_local_index <- which.max(determinants)
  return(candidate_indices[best_local_index])
}

# =============================================================================
# 3. MAIN SIMULATION LOOP (No changes in this section)
# =============================================================================
cat("Step 3: Running M-CAT simulation...\n")

estimated_thetas <- matrix(NA, nrow = N_SIM_PEOPLE, ncol = N_FACTORS)
final_test_lengths <- numeric(N_SIM_PEOPLE)
prior_cov_inverse <- solve(theta_cov)

for (p in 1:N_SIM_PEOPLE) {
  if (p %% 25 == 0) cat(sprintf("  ...simulating person %d of %d\n", p, N_SIM_PEOPLE))
  
  person_true_theta <- true_thetas_simulated[p, ]
  current_theta_estimate <- theta_mean
  
  answered_indices <- c()
  answered_responses <- c()
  
  sem <- Inf
  while (any(sem > SEM_THRESHOLD) && length(answered_indices) < MAX_TEST_LENGTH) {
    remaining_indices <- setdiff(1:N_ITEMS, answered_indices)
    candidate_indices <- sample(remaining_indices, size = min(ITEM_SAMPLE_SIZE, length(remaining_indices)))
    
    answered_factors_matrix <- if (length(answered_indices) > 0) item_factors[answered_indices, , drop = FALSE] else matrix(nrow = 0, ncol = N_FACTORS)
    
    item_to_administer_idx <- select_next_item_mdet(
      current_theta_estimate, answered_factors_matrix, item_factors[candidate_indices, ], candidate_indices
    )
    
    response <- simulate_response(person_true_theta, item_factors[item_to_administer_idx, ])
    
    answered_indices <- c(answered_indices, item_to_administer_idx)
    answered_responses <- c(answered_responses, response)
    
    answered_factors_matrix <- item_factors[answered_indices, , drop = FALSE]
    
    current_theta_estimate <- estimate_theta_map(
      answered_responses, answered_factors_matrix, current_theta_estimate, theta_mean, prior_cov_inverse
    )
    
    total_info_matrix <- calculate_fisher_matrix(answered_factors_matrix, current_theta_estimate)
    var_cov <- try(solve(total_info_matrix), silent = TRUE)
    if (inherits(var_cov, "try-error") || any(diag(var_cov) < 0)) {
      sem <- Inf
    } else {
      sem <- sqrt(diag(var_cov))
    }
  }
  
  estimated_thetas[p, ] <- current_theta_estimate
  final_test_lengths[p] <- length(answered_indices)
}

cat("Simulation complete!\n")

# =============================================================================
# 4. EVALUATION (No changes in this section)
# =============================================================================
cat("\nStep 4: Evaluating the results...\n")

rmse_val <- rmse(true_thetas_simulated, estimated_thetas)
cat(sprintf("\nOverall RMSE: %.4f\n", rmse_val))

cat("\nCorrelation between True and Estimated Thetas (by dimension):\n")
for (d in 1:N_FACTORS) {
  correlation <- cor(true_thetas_simulated[, d], estimated_thetas[, d])
  cat(sprintf("  - Dimension %d: %.4f\n", d, correlation))
}

cat("\nTest Length Statistics:\n")
cat(sprintf("  - Average Test Length: %.2f items\n", mean(final_test_lengths)))
cat(sprintf("  - Minimum Test Length: %d items\n", min(final_test_lengths)))
cat(sprintf("  - Maximum Test Length: %d items\n", max(final_test_lengths)))