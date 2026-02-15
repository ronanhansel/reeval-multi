# Fix for capsule-4252248 (corebench_hard)

## Problem Diagnosis

This task requires running R scripts (`main-ctrpv.R`, `main-nci.R`, `main-network-generation.R`) that depend on specific R packages not available in the standard HAL agent runner environment.

### Required R Packages

The R scripts in this capsule require:
- **CRAN packages**: PRROC, ROCR, SNFtool, apcluster, fingerprint, proxy, reshape2, Hmisc, rcdk
- **Bioconductor packages**: PharmacoGx, survcomp, annotate, org.Hs.eg.db, netbiov
- **System dependency**: default-jdk (required by rcdk for Java-based chemoinformatics)

These packages are specified in the original Code Ocean capsule's Dockerfile (`environment/Dockerfile`), but the HAL harness:
1. Filters out the `environment/` directory for `corebench_hard` difficulty
2. Does not use the capsule's Dockerfile to build the execution environment

Without these packages, the R scripts fail to load their dependencies and cannot execute, making the task impossible regardless of agent capability.

## Fix Applied

**Fix Type**: Environment Fix (env_override.json)

The fix pre-installs the required R packages via conda channels (conda-forge and bioconda). This is appropriate because:

1. **The original capsule provided these packages** - The Dockerfile in the capsule explicitly installs all these packages. The packages SHOULD be available for the task to be solvable.

2. **This preserves task difficulty** - The agent still needs to:
   - Read and understand the README to know what scripts to run
   - Create the correct symbolic links (`../results` and `../data`)
   - Run the three R scripts using Rscript
   - Parse the output to extract the AUC value
   - Format the answer as a Python dictionary

3. **This does NOT nerf the task** - No hints about the answer are provided. No results are pre-computed. The agent must still do all the computational work to obtain the AUC value.

## What Was NOT Fixed

- No modification to the task prompt/instructions
- No pre-computed results added
- No hints about expected output values

## Verification

With this fix applied, the agent should be able to:
1. Find the R scripts in `code/` directory
2. Run `Rscript code/main-ctrpv.R` (and other scripts)
3. The scripts will generate PR curves and output AUC values
4. Agent extracts and reports the requested AUC value
