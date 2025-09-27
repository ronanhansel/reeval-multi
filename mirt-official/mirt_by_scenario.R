# %% Install packages if needed
if(!require(mirt)) install.packages("mirt")
if(!require(psych)) install.packages("psych")
if(!require(GPArotation)) install.packages("GPArotation")

# %% Load libraries
library(mirt)
library(psych)
library(GPArotation)
library(parallel)

# %% Enable parallelism inside mirt
mirtCluster(4)   # uses 4 workers internally where possible

# %% Load data
data <- read.csv("./data/resmat_by_scenario/gsm.csv", 
                 row.names = 1, na.strings = "NA")

# %% Parallel model fitting across dimensions 1..10
ks <- 1:10

if(.Platform$OS.type == "windows"){
  # Windows: use parLapply
  cl <- makeCluster(4)   # create cluster with 4 cores
  clusterEvalQ(cl, { library(mirt) })   # load mirt on workers
  fits <- parLapply(cl, ks, function(k){
    mirt(data, k, itemtype = "2PL", method = "MHRM", 
         rotate = "oblimin", verbose = FALSE)
  })
  stopCluster(cl)
} else {
  # Linux/macOS: can use mclapply
  fits <- mclapply(ks, function(k){
    mirt(data, k, itemtype = "2PL", method = "MHRM", 
         rotate = "oblimin", verbose = FALSE)
  }, mc.cores = 4)
}

names(fits) <- paste0("dim", ks)

# %% Compare model fit indices
fit_stats <- sapply(fits, function(x) 
  c(logLik = logLik(x), AIC = AIC(x), BIC = BIC(x))
)
print(fit_stats)

# %% Model fit statistic (M2) for 2D solution
M2(fits[["dim2"]])
