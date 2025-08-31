# --- 1. Install required packages (run once only) ---
# install.packages("RcppCNPy")
# install.packages("mirt")

# --- 2. Load libraries ---
library(RcppCNPy)
library(mirt)

# --- 3. Load response matrix (from Python .npy) ---
resmat <- npyLoad("/home/azureuser/cloudfiles/code/Users/manhductranvu/reeval-multi/data/resmat.npy")

# --- 4. Define dimensions to test ---
dims_to_test <- 12:15   # change to 1:12 if you want more

# --- 5. Storage for fit results ---
fit_results <- data.frame(
  K = integer(),
  AIC = numeric(),
  BIC = numeric(),
  logLik = numeric(),
  RMSEA = numeric(),
  CFI = numeric(),
  SRMSR = numeric()
)

# --- 5b. Assign unique names to each column (item) ---
colnames(resmat) <- paste0("Item", seq_len(ncol(resmat)))

# Quick check
head(colnames(resmat), 5)
tail(colnames(resmat), 5)
sum(duplicated(colnames(resmat)))

# --- 6. Add a progress bar ---
pb <- txtProgressBar(min = 1, max = length(dims_to_test), style = 3)

for (i in seq_along(dims_to_test)) {
  k <- dims_to_test[i]
  cat("\n\n===============================\n")
  cat(">>> Fitting MIRT model with", k, "dimensions...\n")
  cat("===============================\n")
  
  # Fit model (2PL standard; verbose=TRUE shows optimizer progress)
  model <- mirt(resmat, model = k, itemtype = "2PL", verbose = TRUE, technical = list(NCYCLES = 500, parallel = TRUE))
  
  # Extract standard fit stats
  aic_val  <- AIC(model)
  bic_val  <- BIC(model)
  ll_val   <- logLik(model)
  
  # Extract M2-based indices
  m2_fit <- M2(model, type = "C2")  # more stable for large datasets
  rmsea_val <- m2_fit$RMSEA
  cfi_val   <- m2_fit$CFI
  srmsr_val <- m2_fit$SRMSR
  
  # Store in dataframe
  fit_results <- rbind(
    fit_results,
    data.frame(K = k, AIC = aic_val, BIC = bic_val, logLik = ll_val,
               RMSEA = rmsea_val, CFI = cfi_val, SRMSR = srmsr_val)
  )
  
  # Update progress bar
  setTxtProgressBar(pb, i)
}
close(pb)

# --- 7. Save results to CSV ---
write.csv(fit_results, "./model_fit_results.csv", row.names = FALSE)

# --- 8. Plot selection criteria ---
png("./model_fit_plot.png", width = 1200, height = 400)
par(mfrow = c(1,3))
plot(fit_results$K, fit_results$BIC, type="b", main="BIC by K", xlab="K", ylab="BIC")
plot(fit_results$K, fit_results$CFI, type="b", main="CFI by K", xlab="K", ylab="CFI")
plot(fit_results$K, fit_results$RMSEA, type="b", main="RMSEA by K", xlab="K", ylab="RMSEA")
dev.off()

print(fit_results)
