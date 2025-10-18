library(mirt)
library(parallel)
library.dynam('mirt', 'mirt', .libPaths()) 

# Load the model that stopped at 20000 iterations
fit_stopped <- readRDS("../data-reeval-multi/final_mirt_model.rds") 
print("Loaded model stopped at 20k iterations.")

# Get the parameter values from the loaded model
start_values <- mod2values(fit_stopped) 
# print(head(start_values)) # Optional: view the extracted values

# --- Setup Parallel Cluster AGAIN ---
n_cores <- parallel::detectCores() - 1 
cl <- NULL 
print(paste("Setting up parallel cluster with", n_cores, "cores..."))
cl <- mirtCluster(n_cores)
print("Cluster ready.")

# --- Load original data and model string (needed again) ---
print("Loading original data...")
data <- read.csv("../data-reeval-multi/lsat_qa/lsat_resmat.csv", 
                 header = FALSE, row.names = 1)
colnames(data) <- paste0("Q", 1:ncol(data))

# --- 1. Define Your Model Structure ---

n_items <- ncol(data)
anchors_F1 <- c(1, 7, 12, 19, 27)
anchors_F2 <- c(1, 3, 7, 11, 23)
anchors_F3 <- c(5, 17, 31, 53, 81)

# Find all items that are NOT anchors
all_anchor_items <- unique(c(anchors_F1, anchors_F2, anchors_F3))
free_items <- setdiff(1:n_items, all_anchor_items)

# --- 2. Helper function to create item ranges (e.g., "2,4,6-10") ---
numbers_to_ranges <- function(nums) {
  if (length(nums) == 0) return("")
  nums <- sort(unique(nums))
  diffs <- diff(nums)
  start_indices <- c(1, which(diffs != 1) + 1)
  end_indices <- c(which(diffs != 1), length(nums))
  
  ranges <- mapply(function(start, end) {
    if (nums[start] == nums[end]) {
      return(as.character(nums[start]))
    } else if (nums[start] + 1 == nums[end]) {
       return(paste(nums[start], nums[end], sep = ",")) # e.g., 4,5
    } else {
      return(paste(nums[start], nums[end], sep = "-")) # e.g., 6-10
    }
  }, start_indices, end_indices)
  
  return(paste(ranges, collapse = ","))
}

# --- 3. Build the Final, Corrected Model String ---

# Convert the "free" items into a compact string
free_items_str <- numbers_to_ranges(free_items)

# Create the full model string
model_string_corrected <- paste("
  F1 = ", paste(anchors_F1, collapse = ","), ",", free_items_str, "
  F2 = ", paste(anchors_F2, collapse = ","), ",", free_items_str, "
  F3 = ", paste(anchors_F3, collapse = ","), ",", free_items_str, "
  COV = F1*F2, F1*F3, F2*F3
", sep = "")

# --- Run MIRT again with warm start ---
print("Restarting MIRT estimation with previous parameters as starting values...")
fit_continued <- mirt(data,
                      model_string_corrected,
                      itemtype = "2PL",
                      method = "MHRM",
                      verbose = TRUE,
                      pars = start_values,  # <-- Use extracted parameters as start values
                      technical = list(NCYCLES = 40000)) # <-- Set a NEW, higher limit

print("Continued MIRT run finished!")

# --- Save the NEW, hopefully converged model ---
saveRDS(fit_continued, file = "../data-reeval-multi/final_mirt_model_resumed.rds") # Use a new file name
print("Newly converged model saved.")

# --- Stop Cluster ---
print("Shutting down cluster...")
stopCluster(cl)
print("Cluster shut down.")

# --- Check Results ---
summary(fit_continued)