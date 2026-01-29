#!/bin/bash
# Reproduce HELM Analysis

# 1. Run Analysis (Fitting Models)
echo "Running HELM Analysis..."
python3 run_helm_analysis.py

# 2. Plot Results
echo "Plotting Results..."
python3 plot_helm_results.py

echo "Done. Results saved in helm/result/"
