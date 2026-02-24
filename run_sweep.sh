#!/bin/bash
set -e

# Setup environment
export PYTHONPATH=$(pwd)/model
PYTHON_EXEC="/home/v-qizhengli/miniconda3/envs/hal/bin/python -u"

# Seeds for 10 repetitions
SEEDS="42 123 789 2024 1337 555 666 777 888 999"

# --- PCA Sweep ---
echo "Running PCA (N=max, BETA) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> PCA N=max Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type pca --n-samples max --model-type beta --lambda-tau 0.054 --seed $seed
done

echo "Running PCA (N=1, BERNOULLI) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> PCA N=1 Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type pca --n-samples 1 --model-type bernoulli --lambda-tau 0.0155 --seed $seed
done

# --- SAE Sweep ---
echo "Running SAE (N=max, BETA) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> SAE N=max Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples max --model-type beta --lambda-tau 0.0535 --seed $seed
done

echo "Running SAE (N=1, BERNOULLI) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> SAE N=1 Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type bernoulli --lambda-tau 0.0159 --seed $seed
done

# --- RAW Sweep ---
echo "Running RAW (N=max, BETA) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> RAW N=max Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type raw --n-samples max --model-type beta --lambda-tau 0.029 --seed $seed
done

echo "Running RAW (N=1, BERNOULLI) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> RAW N=1 Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type raw --n-samples 1 --model-type bernoulli --lambda-tau 0.0151 --seed $seed
done

# --- Pre-Revision Stability (8 agents) ---
echo "Running Pre-Revision (8 agents, BERNOULLI) sweep with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> Pre-8 Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type bernoulli --lambda-tau 0.0159 --pre-revision 8 --seed $seed
done

# --- Pre-Revision Max Agents (BETA) ---
echo "Running Pre-Revision (max agents, BETA) with 10 repetitions..."
for seed in $SEEDS
do
    echo " -> Pre-max Seed $seed"
    $PYTHON_EXEC model/amortized_irt.py --embedding-type sae --n-samples 1 --model-type beta --lambda-tau 0.16 --pre-revision max --seed $seed
done

echo "Universal 10-seed sweep completed. Consolidated results saved to model/result/"
