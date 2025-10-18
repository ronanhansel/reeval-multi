library(mirt)

# 1. Load your saved model from the .rds file
# (You can skip all the data loading and model fitting steps)
fit_loaded <- readRDS("../data-reeval-multi/final_mirt_model.rds")

# 2. Print estimations and results (answering your second question)
print("--- Model Summary (Loadings, etc.) ---")
summary(fit_loaded)

print("--- Item Parameters (a, d) ---")
coefs <- coef(fit_loaded, simplify = TRUE)
print(coefs$items)

print("--- Factor Correlations ---")
print(summary(fit_loaded)$fcor)

# 3. Print test-takers' ability predictions (answering your first question)
# This calculates the abilities for the original data used to train the model
abilities <- fscores(fit_loaded)

# Print the first 10 rows
print("--- Test-Taker Abilities (Top 10) ---")
print(abilities)

# --- 1. Load your saved model ---
library(mirt)
library.dynam('mirt', 'mirt', .libPaths())
fit_loaded <- readRDS("../data-reeval-multi/final_mirt_model.rds")

# --- 2. Calculate Item Fit ---
# This will take a very long time.
# S_X2 is the 'Orlando & Thissen's S-X2 statistic'
print("Calculating Item Fit (S-X2)... This will be very slow.")
ifit <- itemfit(fit_loaded, fit_stats = 'S_X2')

# --- 3. View Results ---
print("--- Item Fit Statistics (S-X2) ---")
print(ifit)

# To see which items are a 'bad fit' (p-value < 0.05)
print("--- Items that DO NOT fit the model (p < 0.05) ---")
print(ifit[ifit$p.S_X2 < 0.05, ])