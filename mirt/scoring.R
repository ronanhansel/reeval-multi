# Load necessary libraries
library(data.table)
library(mirt)
library(parallel)

## --- 🚀 SETUP PARALLEL PROCESSING --- ##
num_cores <- detectCores() - 1 
if(num_cores > 1) {
  print(paste("Initializing parallel processing with", num_cores, "cores."))
  mirtCluster(num_cores)
}

## --- 📂 LOAD DATA --- ##
print("Reading item parameters (a and b)...")
a_params <- as.matrix(read.csv("../result/a.csv", header = FALSE))
difficulty_params <- as.matrix(read.csv("../result/b.csv", header = FALSE))

print("Reading response data...")
response_data <- as.matrix(fread("../result/response_data.csv", header = FALSE))

# Previous diagnostic checks can be removed now as data is confirmed to be clean.

## --- 🛠️ BUILD MIRT MODEL --- ##
print("Setting up model specifications...")
num_dimensions <- ncol(a_params)
num_items <- nrow(a_params)
model_spec_string <- paste0('F = 1-', num_items)
model_spec <- mirt.model(model_spec_string)

print("Creating parameter template...")
param_df <- mirt(data = response_data, 
                 model = model_spec, 
                 itemtype = '2PL', 
                 pars = 'values')

print("Populating template with custom parameters...")
param_df$value[param_df$name == 'd'] <- -difficulty_params
for(i in 1:num_dimensions) {
  param_df$value[param_df$name == paste0('a', i)] <- a_params[, i]
}

## --- ✨ NEW CRUCIAL STEP: Fix the parameters to prevent estimation --- ##
print("Fixing all parameters to their starting values...")
param_df$est <- FALSE

print("Rebuilding final model object with fixed parameters...")
# The "converged immediately" warning should now disappear
final_mirt_model <- mirt(data = response_data, 
                         model = model_spec, 
                         itemtype = '2PL', 
                         pars = param_df,
                         verbose = TRUE)

## --- 📈 CALCULATE SCORES --- ##
print("Calculating factor scores...")
theta_r <- fscores(final_mirt_model, response.pattern = response_data, method = 'EAP')

## --- 💾 SAVE RESULTS --- ##
print("Saving the final theta scores to CSV...")
write.csv(theta_r, file = "../result/theta_scores_from_r.csv", row.names = FALSE)

print("Saving the final model object...")
saveRDS(final_mirt_model, file = "../result/final_mirt_model.rds")

## --- ✅ DONE --- ##
print("All tasks complete. Results have been saved.")