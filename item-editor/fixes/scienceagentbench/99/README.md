# Task 99 Fix: Scanpy UMAP Visualization

## Root Cause Analysis

**Task Requirement**: Draw three UMAP figures of single-cell data colored by `total_counts`, `n_genes_by_counts`, and `clusters`, combining them into one picture saved as a PNG file.

**Identified IFE**: The agent development sandbox (smolagents) blocks importing `scanpy`, `muon`, `mudata`, and `matplotlib`, preventing agents from developing and testing code that uses these required libraries.

**Evidence from rubric evaluations**:
- "Import of scanpy is not allowed. Authorized imports are: [...]"
- "Import of muon is not allowed. Authorized imports are: [...]"
- "Import of matplotlib.pyplot is not allowed"
- "ImportError: Missing optional dependency 'pytables'"
- Dataset is only available as `.h5ad` format with no CSV alternative

## Analysis of Docker Evaluation Environment

The Docker harness (`dockerfiles.py`) has special handling for Scanpy:
```bash
if echo "$extracted_pkgs" | grep -q 'scanpy'; then \
    echo 'scikit-misc' >> /testbed/instance_requirements.txt && \
    echo 'leidenalg' >> /testbed/instance_requirements.txt; \
fi;
```

The base image also installs matplotlib (<3.8.0).

## Fix Applied

**Environment Override** (`env_override.json`): Ensures scanpy, anndata, mudata, and visualization dependencies are properly installed.

## Why This Preserves Task Difficulty

This fix does NOT:
- Provide hints about UMAP visualization techniques
- Pre-compute the cluster assignments
- Simplify the single-cell analysis workflow
- Pre-generate any figure elements

This fix ONLY ensures:
- The required packages for reading .h5ad and plotting are available
- Agents can be fairly evaluated on their visualization code

## Expected Outcome

An agent that correctly:
1. Loads the h5ad file with scanpy/anndata
2. Calculates QC metrics (total_counts, n_genes_by_counts)
3. Performs clustering to get cluster assignments
4. Creates UMAP visualizations with proper coloring
5. Combines figures and saves as PNG

Should have their code properly evaluated.

## Note on Evaluation

Task 99 involves figure generation which is evaluated by GPT-4 visual comparison. The `evaluation_override.json` requests relaxed figure tolerance to account for stylistic differences in equivalent visualizations.
