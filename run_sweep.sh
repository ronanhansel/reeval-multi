#!/bin/bash
set -e

# PCA curve
echo "Running PCA sweep"
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type pca --n-samples all --model-type beta --lambda-tau 0.054

# SAE curve
echo "Running SAE sweep"
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type sae --n-samples all --model-type beta --lambda-tau 0.0535

# RAW curve
echo "Running RAW sweep"
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type raw --n-samples all --model-type beta --lambda-tau 0.029
# Pre-Revision N=1 (8 agents)
echo "Running Pre-Revision (8 agents) sweep"
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type beta --lambda-tau 0.12 --pre-revision 8

# Pre-Revision N=1 (Max agents)
echo "Running Pre-Revision (max agents) sweep"
~/miniconda3/bin/conda run -n hal python model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type beta --lambda-tau 0.16 --pre-revision max

echo "All sweeps completed. Results saved to model/result/"
