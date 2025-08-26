# Load necessary libraries
install.packages(c("catR", "tidyverse", "Metrics", "MASS"))
library(catR)
library(tidyverse)
library(Metrics)
library(MASS)

## ----------------------------------------------------------------
## Step 1: Load Your Data and Define Simulation Parameters
## ----------------------------------------------------------------
set.seed(42)

# Using the same simulated data as before
N_ITEMS <- 1000
N_DIM <- 2
N_PEOPLE <- 200
TEST_LENGTH <- 40

a_params <- matrix(rlnorm(N_ITEMS * N_DIM, 0, .3), ncol = N_DIM)
d_params <- matrix(rnorm(N_ITEMS))
item_bank <- as.matrix(cbind(a_params, d_params))
colnames(item_bank) <- c("a1", "a2", "d")

ability_correlation <- matrix(c(1, 0.5, 0.5, 1), nrow = N_DIM)
true_thetas <- mvrnorm(N_PEOPLE, mu = rep(0, N_DIM), Sigma = ability_correlation)

## ----------------------------------------------------------------
## Step 2: Run the CAT Simulation using a Loop
## ----------------------------------------------------------------
print("Running CAT simulation...")

# --- A) ADAPTIVE (MFI) SIMULATION ---
estimated_thetas_adaptive <- matrix(NA, nrow = N_PEOPLE, ncol = N_DIM)

print("Starting Adaptive Simulation...")
for (p in 1:N_PEOPLE) {
  if (p %% 10 == 0) {
    print(paste("...simulating person", p, "of", N_PEOPLE))
  }
  
  person_cat <- list(items = numeric(), responses = numeric(), thetas = matrix(0, nrow = 1, ncol = N_DIM))
  
  for (i in 1:TEST_LENGTH) {
    # FINAL FIX: The correct argument is 'method'
    next_item <- nextItem(itemBank = item_bank, theta = person_cat$thetas[i, ], method = "MFI", out = person_cat$items)
    person_cat$items <- c(person_cat$items, next_item$item)
    
    response <- genPattern(theta = true_thetas[p, ], itemBank = item_bank[next_item$item, , drop = FALSE], model = "M2PL")
    person_cat$responses <- c(person_cat$responses, response)
    
    updated_theta <- eapEst(it = item_bank[person_cat$items, , drop = FALSE], x = person_cat$responses, model = "M2PL")
    person_cat$thetas <- rbind(person_cat$thetas, updated_theta)
  }
  estimated_thetas_adaptive[p, ] <- person_cat$thetas[TEST_LENGTH + 1, ]
}

# --- B) RANDOM SIMULATION ---
estimated_thetas_random <- matrix(NA, nrow = N_PEOPLE, ncol = N_DIM)

print("Starting Random Simulation...")
for (p in 1:N_PEOPLE) {
  if (p %% 10 == 0) {
    print(paste("...simulating person", p, "of", N_PEOPLE))
  }
  
  person_cat <- list(items = numeric(), responses = numeric(), thetas = matrix(0, nrow = 1, ncol = N_DIM))
  
  for (i in 1:TEST_LENGTH) {
    # FINAL FIX: The correct argument is 'method'
    next_item <- nextItem(itemBank = item_bank, method = "random", out = person_cat$items)
    person_cat$items <- c(person_cat$items, next_item$item)
    
    response <- genPattern(theta = true_thetas[p, ], itemBank = item_bank[next_item$item, , drop = FALSE], model = "M2PL")
    person_cat$responses <- c(person_cat$responses, response)
    
    updated_theta <- eapEst(it = item_bank[person_cat$items, , drop = FALSE], x = person_cat$responses, model = "M2PL")
    person_cat$thetas <- rbind(person_cat$thetas, updated_theta)
  }
  estimated_thetas_random[p, ] <- person_cat$thetas[TEST_LENGTH + 1, ]
}

print("Simulation complete.")

## ----------------------------------------------------------------
## Step 3: Evaluate and Visualize the Results
## ----------------------------------------------------------------
print("Evaluating results...")
results <- tibble()
for (d in 1:N_DIM) {
  rmse_adaptive <- rmse(true_thetas[, d], estimated_thetas_adaptive[, d])
  corr_adaptive <- cor(true_thetas[, d], estimated_thetas_adaptive[, d])
  
  rmse_random <- rmse(true_thetas[, d], estimated_thetas_random[, d])
  corr_random <- cor(true_thetas[, d], estimated_thetas_random[, d])
  
  results <- results %>%
    add_row(Dimension = d, Strategy = "Adaptive (MFI)", RMSE = rmse_adaptive, Correlation = corr_adaptive) %>%
    add_row(Dimension = d, Strategy = "Random", RMSE = rmse_random, Correlation = corr_random)
}

print("--- CAT Simulation Performance ---")
print(results)

# Visualization
plot_data <- tibble(True_Theta = true_thetas[,1], Adaptive_Estimate = estimated_thetas_adaptive[,1], Random_Estimate = estimated_thetas_random[,1])
ggplot(plot_data, aes(x = True_Theta)) +
  geom_point(aes(y = Adaptive_Estimate, color = "Adaptive (MFI)"), alpha = 0.7) +
  geom_point(aes(y = Random_Estimate, color = "Random"), alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(title = "CAT Performance: True vs. Estimated Ability (Dimension 1)", x = "True Ability (Theta)", y = "Estimated Ability", color = "Selection Strategy") +
  scale_color_manual(values = c("Adaptive (MFI)" = "blue", "Random" = "red")) +
  theme_minimal() + coord_fixed()