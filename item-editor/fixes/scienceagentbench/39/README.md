# Task 39 Fix: ProLIF/MDAnalysis Environment Issue

## Root Cause Analysis

**IFE Type**: Execution Environment Issue - Missing Required Packages

### Problem Description
Task 39 requires computing protein-protein interaction fingerprints using ProLIF, which is built on MDAnalysis. The task involves:
1. Loading PDB and XTC trajectory files with MDAnalysis
2. Computing interaction fingerprints between protein frames using ProLIF
3. Calculating Tanimoto similarities and plotting results

### Evidence of IFE
From the rubric evaluations across all 4 models (GPT-4.1, O3, O4-mini-high, O4-mini-low):
- All agents encountered: `InterpreterError: Import of MDAnalysis is not allowed`
- The sandbox interpreter's `AUTHORIZED_IMPORTS` list doesn't include MDAnalysis, prolif, or matplotlib
- This is a universal failure pattern - **not an agent capability issue**

### Model Execution Results
The Docker evaluation shows:
```
valid_program: 0
codebert_score: 0.8012
success_rate: 0
log_info: DeprecationWarning from MDAnalysis...
```

The MDAnalysis package runs but with deprecation warnings. The `valid_program: 0` indicates the agent couldn't generate valid code because it couldn't test imports during development.

## Fix Applied

### Environment Override (`env_override.json`)
- **HAL_CONDA_CHANNELS**: conda-forge (for ProLIF)
- **HAL_PIP_PACKAGES**: prolif MDAnalysis matplotlib

### Why This Fix is Fair (Not a Nerf)
1. **Preserves Scientific Rigor**: The task still requires understanding protein-protein interactions, trajectory analysis, and Tanimoto similarity calculations
2. **No Hints Given**: We don't provide any solution logic - just ensure the required tools are available
3. **Standard Dependencies**: ProLIF, MDAnalysis, and matplotlib are explicitly required by the task specification
4. **Cross-Model Evidence**: All 4 models failed at the same import barrier, indicating infrastructure (not capability) issue

## Expected Outcome After Fix
- Agents can generate and test code that imports required packages
- Docker evaluation will have ProLIF/MDAnalysis/matplotlib available
- Task success depends entirely on agent's scientific understanding, not import restrictions

## Technical Notes
- ProLIF is a specialized library for Protein-Ligand Interaction Fingerprints
- MDAnalysis is the underlying molecular dynamics trajectory analysis library
- The task uses `.pdb` (structure) and `.xtc` (trajectory) file formats
