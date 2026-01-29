#!/bin/bash
# Reproduce HELM Analysis

# Use the reeval conda environment python
PYTHON_EXEC="/home/v-tatruong/miniconda3/envs/reeval/bin/python3"

# 1. Run Analysis (Fitting Models)
echo "Running HELM Analysis..."
$PYTHON_EXEC run_helm_analysis.py

# 2. Plot Results
echo "Plotting Results..."
$PYTHON_EXEC plot_helm_results.py

echo "Done. Results saved in helm/result/"