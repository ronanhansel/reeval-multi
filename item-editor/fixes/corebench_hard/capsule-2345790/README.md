# Fix for capsule-2345790 (corebench_hard)

## Problem Diagnosis

This task requires agents to test the computational reproducibility of an R-based scientific research project about attentional fluctuations and memory (Jayakumar, Balusu & Aly, 2023). The task requires agents to:
1. Create subfolders in `./results`: `intermediates`, `figures`, `stats_figures_markdowns`
2. Run all `.Rmd` files using Rscript and render them as HTML
3. Store output files in `./results/stats_figures_markdowns`
4. Extract the mean response rate across all participants from Study 1 and Study 2

**Root Cause**: The original Code Ocean capsule was designed to run in a Docker container (`registry.codeocean.com/published/aefe6507-baa3-470e-94cc-bddf88d069aa:v1`) based on `registry.codeocean.com/codeocean/r-studio:2022.07.0-548-r4.2.1-ubuntu18.04` with R 4.2.2, pandoc, and all required R packages pre-installed. The HAL benchmark execution environment does not have R installed by default.

**Evidence from Model Logs**:
- All agents encountered: `Exit Code: 127`, `Rscript: not found` when attempting to render R Markdown files
- When gpt-4.1-04-14 managed to get R installed, it failed with: `Error: pandoc version 1.12.3 or higher is required and was not found`
- The HAL harness toolchain preflight explicitly reported: `Missing baseline R/pandoc/TeX toolchain in the container image; this causes widespread CoreBench 'environmental barrier' failures`
- All 4 models (gpt-4.1-04-14, o3-04-16, o4-mini-04-16) failed with the same infrastructure barriers

**Permission Issue**: The prompt says to create subfolders in `../results`, but the parent directory is not writable. The prompt already clarifies this: "Do not write to `../results` (parent directory is not writable). Use `./results` (or `HAL_OUTPUT_DIR`) instead." However, this is an instruction issue, not an environment issue, so we don't fix it here.

## Fix Applied

**Type**: Environment Fix (env_override.json)

**Solution**: Pre-install R and the required R packages via conda-forge. The packages match those specified in the original Dockerfile:

- `r-base=4.2` - R runtime (matching R 4.2.2 in original Dockerfile)
- `r-rmarkdown` - For rendering RMarkdown files
- `r-knitr` - For knitting documents
- `r-dplyr`, `r-tidyr`, `r-purrr`, `r-readr` - Data manipulation (tidyverse)
- `r-ggplot2`, `r-ggpubr`, `r-gridextra`, `r-cowplot` - Visualization
- `r-kableextra` - Table formatting
- `r-lme4`, `r-lmertest`, `r-emmeans`, `r-ez`, `r-afex` - Statistical modeling
- `r-bayesfactor` - Bayesian statistics
- `r-rmisc`, `r-car`, `r-effects` - Additional statistical utilities
- `r-smoother`, `r-nloptr` - Smoothing and optimization
- `r-stringr`, `r-zoo` - Additional utilities mentioned in Rmd files
- `pandoc` - Document conversion (required for HTML rendering)

## Why This Fix is Appropriate

1. **This is an infrastructure requirement, not part of the challenge**: The original capsule was designed to run in an environment where R and all packages were pre-installed. The task is about testing computational reproducibility by running existing R analysis code, not about setting up an R environment from scratch.

2. **The fix does NOT nerf the question**: Agents still need to:
   - Understand the task requirements from the README
   - Create the required directory structure (`./results/intermediates`, `./results/figures`, `./results/stats_figures_markdowns`)
   - Run all 10 R Markdown files using Rscript to render HTML
   - Wait for the R analyses to complete
   - Parse the generated HTML outputs to extract:
     - Mean response rate for Study 1 across all participants
     - Mean response rate for Study 2 across all participants
   - Return the correct dictionary format with the exact question keys

3. **This matches the original execution environment**: The Code Ocean Dockerfile shows R 4.2.2 with pandoc 1.19.2 and all packages should be pre-installed. We are simply restoring the intended environment.

## What This Fix Does NOT Do

- Does NOT simplify the computational task
- Does NOT give hints about the answer values
- Does NOT pre-compute any results
- Does NOT change the questions being asked
- Does NOT reduce the number of R Markdown files to process
- Does NOT reduce the number of steps agents need to perform

The core challenge remains: understanding the codebase structure, executing the full R analysis pipeline (10 Rmd files), and extracting specific statistical values from the generated outputs.

## Original Dockerfile Reference

The original environment uses:
```dockerfile
FROM registry.codeocean.com/codeocean/r-studio:2022.07.0-548-r4.2.1-ubuntu18.04
# R packages installed: BayesFactor, Rmisc, afex, car, cowplot, dplyr, effects,
# emmeans, ez, ggplot2, ggpubr, gridExtra, kableExtra, knitr, lme4, lmerTest,
# nloptr, purrr, readr, rmarkdown, smoother, tidyr
# Plus: pandoc, build-essential, cmake, libgit2-dev, libnlopt-dev, libssl-dev
```
