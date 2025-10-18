# --- 1. Load Libraries & Model ---
print("Loading libraries and saved model...")
library(mirt)
library(parallel)
# This line helps ensure mirt's compiled code is properly linked, especially after parallel tasks
library.dynam('mirt', 'mirt', .libPaths()) 

# Load the model you saved earlier
fit_loaded <- readRDS("../data-reeval-multi/final_mirt_model.rds")
print("Model loaded.")

# --- 2. Setup Parallel Cluster FOR itemfit ---
# We create a new cluster specifically for the itemfit calculation.
# This helps ensure it's recognized correctly.
n_cores <- parallel::detectCores() - 1 
cl <- NULL # Make sure no old cluster definition interferes
print(paste("Setting up parallel cluster with", n_cores, "cores for itemfit test..."))
cl <- mirtCluster(n_cores) # Start the cluster
print("Cluster ready.")

# --- 3. Run itemfit Test & Time It ---
print("Starting itemfit test on the first 10 items (using parallel cores)...")
start_time <- Sys.time() # Record the start time

# Calculate S_X2 for items 1 through 10 ONLY.
# itemfit will automatically use the active 'cl' cluster.
ifit_test_results <- itemfit(fit_loaded, 
                             items = 1:10,       # <-- Only process these items
                             fit_stats = 'S_X2') # <-- Specify the statistic

end_time <- Sys.time() # Record the end time
time_for_10_items <- end_time - start_time # Calculate the duration

print("Itemfit test for 10 items finished.")

# --- 4. IMPORTANT: Stop the Cluster ---
# Always stop the cluster when you're done to free up resources.
print("Shutting down the parallel cluster...")
stopCluster(cl)
print("Cluster shut down.")

# --- 5. Show Results & Estimate Total Time ---
print("--- Results for the first 10 items ---")
print(ifit_test_results) # Show the S_X2 results for the first 10 items

# Convert the recorded time into minutes for calculation
time_minutes_for_10 <- as.numeric(time_for_10_items, units = "mins") 
print(paste("Time taken for 10 items:", round(time_minutes_for_10, 2), "minutes"))

# Estimate the total time needed for all items
total_items <- ncol(extract.mirt(fit_loaded, 'data')) # Get exact number of items from model
estimated_total_minutes <- (time_minutes_for_10 / 10) * total_items
estimated_total_hours <- estimated_total_minutes / 60

print("--- Estimated Total Time ---")
print(paste("Estimated time for all", total_items, "items:", 
            round(estimated_total_minutes, 2), "minutes (approx.",
            round(estimated_total_hours, 2), "hours)."))