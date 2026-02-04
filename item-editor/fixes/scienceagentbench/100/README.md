# Task 100 Fix: Scanpy Spatial Visualization

## Root Cause Analysis

**Task Requirement**: Visualize single-cell data in spatial coordinates in three figures colored by `total_counts`, `n_genes_by_counts`, and `clusters`, combining them into one picture saved as a PNG file.

**Identified IFE**: The agent development sandbox (smolagents) blocks importing `scanpy`, `h5py`, and even basic file operations (`open`), preventing agents from developing and testing code that uses these required libraries.

**Evidence from rubric evaluations**:
- "Import of scanpy is not allowed. Authorized imports are: [...]"
- "Import of h5py is not allowed. Authorized imports are: [...]"
- "Forbidden function evaluation: 'open' is not among the explicitly allowed tools..."
- The task requires scanpy's `pl.spatial` function for spatial visualization

## Analysis of Docker Evaluation Environment

The Docker harness has special handling for Scanpy (leidenalg, scikit-misc), but the base image doesn't include squidpy or spatial-specific dependencies.

**Key observation from execution log**:
```
FutureWarning: In the future, the default backend for leiden will be igraph instead of leidenalg.
sc.tl.leiden(adata, key_added="clusters")
Traceback (most recent call last):
```

This shows that scanpy WAS loaded and executed in Docker, but the program crashed during execution - suggesting additional spatial dependencies may be missing.

## Fix Applied

**Environment Override** (`env_override.json`): Ensures scanpy with spatial analysis capabilities (squidpy) and required dependencies are properly installed.

## Why This Preserves Task Difficulty

This fix does NOT:
- Provide hints about spatial visualization techniques
- Pre-compute cluster assignments or spatial coordinates
- Simplify the spatial analysis workflow
- Generate any pre-computed outputs

This fix ONLY ensures:
- The required packages for spatial single-cell analysis are available
- Agents can be fairly evaluated on their visualization code

## Expected Outcome

An agent that correctly:
1. Loads the h5ad file with scanpy/anndata
2. Calculates QC metrics
3. Performs clustering
4. Uses scanpy's spatial visualization (pl.spatial)
5. Combines figures and saves as PNG

Should have their code properly evaluated.

## Technical Note

Spatial transcriptomics visualization requires:
- `scanpy` for core single-cell analysis
- Properly formatted spatial coordinates in the AnnData object
- leidenalg/igraph for clustering
- matplotlib for rendering
