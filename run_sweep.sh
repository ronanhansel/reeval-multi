#!/bin/bash
set -e

# Setup environment
export PYTHONPATH=$(pwd)/model
PYTHON_EXEC="/home/v-qizhengli/miniconda3/envs/hal/bin/python -u"

# Seeds for 10 repetitions (N=1)
SEEDS="42 123 789 2024 1337 555 666 777 888 999"

# --- PCA Sweep ---
echo "Running PCA (N=max, BETA)..."
$PYTHON_EXEC model/amortized_irt.py --embedding-type pca --n-samples max --model-type beta --lambda-tau 0.054 --seed 42

echo "Running PCA (N=1, BERNOULLI) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type pca --n-samples 1 --model-type bernoulli --lambda-tau 0.0155 --seed $seed
done

# --- SAE Sweep ---
echo "Running SAE (N=max, BETA)..."
$PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples max --model-type beta --lambda-tau 0.0535 --seed 42

echo "Running SAE (N=1, BERNOULLI) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type bernoulli --lambda-tau 0.0159 --seed $seed
done

# --- RAW Sweep ---
echo "Running RAW (N=max, BETA)..."
$PYTHON_EXEC model/amortized_irt.py --embedding-type raw --n-samples max --model-type beta --lambda-tau 0.029 --seed 42

echo "Running RAW (N=1, BERNOULLI) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type raw --n-samples 1 --model-type bernoulli --lambda-tau 0.0151 --seed $seed
done

# --- Pre-Revision Stability (8 agents) ---
echo "Running Pre-Revision (8 agents, BERNOULLI) sweep with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type bernoulli --lambda-tau 0.0159 --pre-revision 8 --seed $seed
done

# --- Pre-Revision Max Agents (BETA) ---
echo "Running Pre-Revision (max agents, BETA) sweep..."
$PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type beta --lambda-tau 0.16 --pre-revision max --seed 42

echo "All sweeps completed. Results saved to model/result/"
