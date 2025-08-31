
# --- 1. Install and Load Packages ---
# You only need to run install.packages once
install.packages("RcppCNPy")
install.packages("logisticPCA") # If you don't have it from before

library(RcppCNPy)
library(logisticPCA)

# --- 2. Load Your .npy Data Directly ---
# Use the npyLoad() function to load the numpy array into an R matrix
resmat <- npyLoad("/home/azureuser/cloudfiles/code/Users/manhductranvu/reeval-multi/data/resmat.npy")


# --- 6. Install & Load MIRT for Model Fitting ---
install.packages("mirt")
library(mirt)

# --- 7. Convert Your Matrix into an Item Response Data Frame ---
# Ensure it's integer/binary {0,1}, not numeric {0.0,1.0}
resmat <- as.data.frame(resmat)
resmat[] <- lapply(resmat, as.integer)

# --- 8. Define Candidate Dimensions ---
K_candidates <- 12

# Storage for results
fit_results <- data.frame(
  K = integer(),
  AIC = numeric(),
  BIC = numeric(),
  logLik = numeric(),
  RMSEA = numeric(),
  CFI = numeric(),
  SRMSR = numeric(),
  stringsAsFactors = FALSE
)

# --- 9. Loop over K and Fit MIRT ---
for (k in K_candidates) {
  cat("Fitting MIRT model with", k, "dimensions...\n")
  
  # Fit multidimensional 2PL model
  model_fit <- mirt(resmat, model = k, itemtype = "2PL", verbose = FALSE)
  
  # Extract model fit stats
  aic_val <- AIC(model_fit)
  bic_val <- BIC(model_fit)
  ll_val <- extract.mirt(model_fit, "logLik")
  
  # Limited-information fit: M2 test (may be slow for 78k items!)
  m2_stats <- tryCatch({
    mirt::M2(model_fit, type = "C2") # C2 is more stable than plain M2
  }, error = function(e) {
    cat("M2 failed for k =", k, "\n")
    return(NULL)
  })
  
  # Parse indices
  if (!is.null(m2_stats)) {
    rmsea_val <- m2_stats$RMSEA
    cfi_val   <- m2_stats$CFI
    srmsr_val <- m2_stats$SRMSR
  } else {
    rmsea_val <- NA
    cfi_val   <- NA
    srmsr_val <- NA
  }
  
  # Save results
  fit_results <- rbind(fit_results, data.frame(
    K = k,
    AIC = aic_val,
    BIC = bic_val,
    logLik = ll_val,
    RMSEA = rmsea_val,
    CFI = cfi_val,
    SRMSR = srmsr_val
  ))
}

# --- 10. Inspect Results ---
print(fit_results)

# Save to CSV
write.csv(fit_results, "./model_fit_results.csv", row.names = FALSE)

# --- 11. Plot BIC/CFI/RMSEA Across Dimensions ---
png("./model_fit_plot.png")

par(mfrow = c(1,3))
plot(fit_results$K, fit_results$BIC, type="b", main="BIC by K", xlab="K", ylab="BIC")
plot(fit_results$K, fit_results$CFI, type="b", main="CFI by K", xlab="K", ylab="CFI")
plot(fit_results$K, fit_results$RMSEA, type="b", main="RMSEA by K", xlab="K", ylab="RMSEA")

dev.off()
