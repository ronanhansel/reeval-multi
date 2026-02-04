# Fix for capsule-8536428 (corebench_hard)

## Diagnosis

### Task Description
This task asks agents to test computational reproducibility of a fake news detection ML benchmark. Agents must:
1. Find and run Python training scripts for NB (Naive Bayes with n-gram) and k-NN (with Empath features) on the combined corpus
2. Extract accuracy, precision, recall, and F1 metrics from the output
3. Return results as a Python dictionary

### Observed Failures
All tested models (gpt-4.1, o3, o4-mini) failed to complete the task:

1. **gpt-4.1**: Made significant progress using `execute_bash` to run Python scripts, but got stuck fixing a pandas API compatibility issue (`DataFrame.drop(['Label'], 1)` should be `DataFrame.drop(['Label'], axis=1)` in modern pandas).

2. **o3 and o4-mini**: Got stuck in loops trying to:
   - Use `import os` in the `python_interpreter` tool (which has restricted imports)
   - Format bash commands incorrectly (harness requires `py` code blocks)

### Root Cause Analysis

The rubric scored this as 1.0 (Environmental Barrier), but the analysis reveals a **mixed issue**:

1. **Tool Confusion (Capability Issue)**: Models tried to use `import os` in the sandboxed `python_interpreter` tool instead of using `execute_bash()` for filesystem operations. The `execute_bash` tool IS available and works correctly (as demonstrated by gpt-4.1's partial success).

2. **Documentation Gap (Minor Environmental Issue)**: The task prompt doesn't clearly indicate that:
   - The `python_interpreter` tool has restricted imports for security
   - Filesystem operations and running Python scripts should use `execute_bash`
   - Files are located in `./environment/` directory structure

3. **Pandas API Compatibility (Legitimate Challenge)**: The original capsule code uses deprecated pandas syntax. This is a legitimate part of the reproducibility challenge - the code was written for older pandas versions.

## Fix Applied

**Type**: Input Override (input_override.json)

### Clarifications Added:
1. Explains that `execute_bash` should be used for filesystem operations and running Python scripts
2. Clarifies the file location structure (`./environment/code/` and `./environment/data/`)

### What This Fix Does NOT Do:
- Does NOT give hints about which specific scripts to run
- Does NOT reveal the answer values
- Does NOT fix the pandas compatibility issue (that's part of the reproducibility challenge)
- Does NOT reduce the computational requirements
- Does NOT pre-install any packages

## Justification

This fix preserves the core challenge of the task:
- Agents must still discover the relevant Python scripts
- Agents must still figure out how to run them
- Agents must still handle any code compatibility issues
- Agents must still extract and parse the metrics

The clarification simply helps agents understand which tools to use for what purpose - this is akin to documenting the evaluation environment, not simplifying the task.

## Expected Impact

With this fix, agents should be able to:
1. Use `execute_bash("find ./environment ...")` to discover files
2. Use `execute_bash("python ./environment/code/...")` to run scripts
3. Still need to fix the pandas API issue when they encounter it
4. Still need to parse output and extract metrics

The task remains challenging - agents must debug code compatibility issues and extract structured data from script output.
