library(mirt)
library(parallel)

# Read the CSV
data <- read.csv("/Users/ronan/Developer/reeval-multi/data-reeval-multi/lsat_qa/lsat_resmat.csv", header = FALSE, row.names = 1)

# Label the columns (since there’s no header)
colnames(data) <- paste0("Q", 1:ncol(data))

# Check
head(data)

model <- "
F1 = 1,7,12,19,27
F2 = 1,3,7,11,23
F3 = 5,17,31,53,81
COV = F1*F2, F1*F3, F2*F3
"

fit <- mirt(data, model, itemtype = "2PL", method = "EM")
summary(fit)

# View item parameters (discrimination & difficulty)
coef(fit, simplify = TRUE)$items

# Estimated skill (theta) for each model
abilities <- fscores(fit)
print(abilities)

# Model fit
M2(fit)