#!/bin/bash
set -e

# PCA curve
echo "Running PCA sweep (N=1..54)..."
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type pca --n-samples all --model-type beta --lambda-tau 1.38

# SAE curve
echo "Running SAE sweep (N=1..54)..."
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type sae --n-samples all --model-type beta --lambda-tau 1.34

# RAW curve
echo "Running RAW sweep (N=1..54)..."
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type raw --n-samples all --model-type beta --lambda-tau 1.0

echo "All sweeps completed. Results saved to model/result/"
