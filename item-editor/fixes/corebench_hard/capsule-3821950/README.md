# Fix for capsule-3821950 (corebench_hard)

## Problem Diagnosis

This task requires agents to test the computational reproducibility of an R-based scientific research project about archaeological excavations at Khao Toh Chong, Krabi, Thailand. The task explicitly instructs agents to:
1. Create a 'figures' directory in the results folder
2. Run `ktc_11_paper.Rmd` using Rscript and render it as HTML
3. Save the output to `../results` with `clean = TRUE`
4. Extract specific values from the rendered figures to answer two questions about material types at specific calibrated years BP

**Root Cause**: The original Code Ocean capsule was designed to run in a Docker container (`registry.codeocean.com/codeocean/r-base:3.4.0-debian8`) with R and all required R packages pre-installed. However, the HAL benchmark execution environment does not have R installed, and agents cannot install it via `apt-get` due to permission restrictions. Additionally, the sandbox restricts `import os` in Python, preventing agents from running shell commands.

**Evidence from Model Logs**:
- Agent o3 reported: "The execution container currently has **no R installation** (`which Rscript` returned nothing). We also do not have root permissions to install system packages through `apt-get`."
- Rubric states: "When the agent attempts to create directories and run R via system calls, the Python sandbox prevents importing `os`"
- All 4 models (gpt-4.1-04-14, o3-04-16, o4-mini-04-16) failed with the same infrastructure barrier

## Fix Applied

**Type**: Environment Fix (env_override.json)

**Solution**: Pre-install R and the required R packages via conda-forge. The packages installed match those specified in the original Dockerfile:

- `r-base=4.1` - R runtime (matching the original's R 3.4.0)
- `r-rmarkdown` - For rendering RMarkdown files
- `r-knitr` - Dynamic report generation (v1.14 in original Dockerfile)
- `r-dplyr` - Data manipulation (used in ktc_11_paper.Rmd)
- `r-ggplot2` - Visualization (used for plots in ktc_11_paper.Rmd)
- `r-scales` - Scale formatting for ggplot (used in ktc_11_paper.Rmd)
- `r-bchron` - Bayesian chronology modeling for radiocarbon dates (v4.2.5 in original Dockerfile)
- `r-hmisc` - Harrell miscellaneous utilities (v3.17-4 in original Dockerfile)
- `r-devtools` - For installing the ktc11 package from GitHub
- `r-gridextra` - Grid graphics composition (used in ktc_11_paper.Rmd)
- `r-maptools` - Geographical data handling (v0.8-39 in original Dockerfile)
- `pandoc` - Document rendering

**Note**: The custom `ktc11` R package (from `benmarwick/ktc11`) still needs to be installed by the agent using `devtools::install_github()`. This preserves the challenge of understanding the project structure and following the README instructions.

## Why This Fix is Appropriate

1. **This is an infrastructure requirement, not part of the challenge**: The original capsule was designed to run in an environment where R was already installed. The task is about testing computational reproducibility by running existing code, not about setting up an R environment from scratch.

2. **The fix does NOT nerf the question**: Agents still need to:
   - Understand the task requirements from the README
   - Create the required directory structure (`results/figures`)
   - Install the custom `ktc11` R package using devtools
   - Navigate to the correct directory and run the RMarkdown rendering command
   - Wait for the R scripts to execute (including Bayesian chronology modeling with Bchron)
   - Parse the generated HTML output to extract answers from figures:
     - Material with highest Depth Below Surface at 10,000 calibrated years BP
     - Material with highest mass (g) at 5,000 calibrated years BP

3. **This matches the original execution environment**: The Code Ocean Dockerfile shows R and base packages should be pre-installed. We are simply restoring the intended environment.

## What This Fix Does NOT Do

- Does NOT simplify the computational task
- Does NOT give hints about the answer values
- Does NOT pre-compute any results
- Does NOT change the questions being asked
- Does NOT reduce the number of steps agents need to perform
- Does NOT pre-install the custom `ktc11` package (agents must still do this)

The core challenge remains: understanding the codebase structure, installing the custom R package, executing the RMarkdown analysis pipeline, and extracting specific values from the generated figures.
