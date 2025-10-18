library(mirt)
library(pROC)

# --- Load model & data (your code) ---
fit_loaded <- readRDS("../data-reeval-multi/final_mirt_model_100.rds")
data <- read.csv("../data-reeval-multi/lsat_qa/lsat_resmat.csv", header = FALSE, row.names = 1)
colnames(data) <- paste0("Q", 1:ncol(data))

# --- Get abilities (as you did) ---
abilities <- fscores(fit_loaded, verbose = FALSE)  # returns matrix with one column (theta) for 1D
# if fscores returns a data.frame/matrix, make sure it's a plain numeric matrix:
Theta <- as.matrix(abilities)

# --- Get model-implied probabilities of the 'correct' category (P=1) ---
# Option A: expected.item() -> returns expected value (for dichotomous this is P(correct))
# For an estimated model you can call expected.item(extract.item(fit_loaded, i), Theta)
# But mirt provides a convenience expected.test() as well. We'll use expected.item for clarity.

n_items <- ncol(data)
n_persons <- nrow(data)

# Pre-allocate matrix of probabilities: rows = persons, cols = items
pred_probs <- matrix(NA_real_, nrow = n_persons, ncol = n_items)
colnames(pred_probs) <- colnames(data)
rownames(pred_probs) <- rownames(data)

for (i in seq_len(n_items)) {
  extr_i <- extract.item(fit_loaded, i)            # extract internal item object
  # expected.item returns a vector of expected scores (for dichotomous item this is P(1))
  pred_probs[, i] <- as.numeric(expected.item(extr_i, Theta, min = 0))
}

# --- Optional check if probtrace would produce same P(1) ---
# You can inspect probtrace output shape if you want to confirm:
# tmp <- probtrace(fit_loaded, Theta)
# str(tmp)  # check how categories are laid out; for dichotomous need the "category 1" slice

# --- Ensure observed responses are numeric 0/1 ---
# Convert factor/character to numeric 0/1 safely:
clean_response <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  # attempt numeric conversion
  xnum <- suppressWarnings(as.numeric(x))
  if (any(is.na(xnum) & !is.na(x))) {
    stop("Observed responses contain non-numeric values that cannot be coerced to 0/1.")
  }
  # If values are 1/2 etc, convert according to your coding (here we assume 1 = correct)
  unique_vals <- sort(unique(xnum[!is.na(xnum)]))
  if (!all(unique_vals %in% c(0,1))) {
    stop("Observed responses are not coded as 0/1. Please recode (e.g., 1 = correct, 0 = incorrect).")
  }
  return(xnum)
}

# --- Compute AUC per item using pROC robustly ---
item_aucs <- setNames(rep(NA_real_, n_items), colnames(data))

for (i in seq_len(n_items)) {
  actual_responses <- clean_response(data[, i])
  predicted_probabilities <- pred_probs[, i]

  nas <- is.na(actual_responses) | is.na(predicted_probabilities)
  actual <- actual_responses[!nas]
  pred <- predicted_probabilities[!nas]

  if (length(unique(actual)) <= 1) {
    item_aucs[i] <- NA  # cannot compute AUC when all outcomes identical
    next
  }

  # ensure response is coded 0/1; pROC accepts numeric(0/1) or factor(2 levels)
  roc_obj <- roc(actual, pred, quiet = TRUE)
  item_aucs[i] <- auc(roc_obj)
}

# --- Summaries ---
mean_auc <- mean(item_aucs, na.rm = TRUE)
cat("--- Mean AUC across all items ---\n"); print(mean_auc)
cat("--- AUC for first 20 items ---\n"); print(head(item_aucs, 20))
