#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Generalisability Experiments
#
# Clean structure:
#   - embeddings.py: Handles Raw/Qwen, PCA, SAE embeddings
#   - models.py: Bernoulli and Beta IRT models
#   - plotting.py: Unified plotting with tueplots icml2024
#
# Usage:
#   ./reproduce.sh           # Interactive mode
#   ./reproduce.sh helm      # Run HELM evaluation
#   ./reproduce.sh colbench  # Run ColBench evaluation
#   ./reproduce.sh both      # Run both evaluations
###############################################################################

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate reeval
echo "[ENV] conda env: reeval (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/result"

echo "=========================================================="
echo "  GENERALISABILITY — Reproducibility Script"
echo "=========================================================="
echo ""
echo "  Working dir : ${SCRIPT_DIR}"
echo "  Output dir  : ${RESULT_DIR}"
echo ""

# Function to run HELM evaluation
run_helm() {
    echo "=========================================================="
    echo "  RUNNING: HELM Benchmark Evaluation"
    echo "=========================================================="
    echo ""

    # Clean caches
    echo "[CLEAN] Removing caches..."
    rm -rf /tmp/_sae_embeddings_ckpt
    rm -rf "${SCRIPT_DIR}/embeddings_cache"
    rm -f "${RESULT_DIR}/helm_results.csv"

    echo "[RUN] Training HELM baselines (Bernoulli IRT)..."
    set +e
    (cd "${SCRIPT_DIR}" && python models.py --benchmark helm --embedding-type pca 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "[ERROR] models.py --benchmark helm exited with code ${rc}"
        return 1
    fi

    echo ""
    echo "[RUN] Generating HELM plots..."
    set +e
    (cd "${SCRIPT_DIR}" && python plotting.py --plot helm 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "[ERROR] plotting.py --plot helm exited with code ${rc}"
        return 1
    fi

    echo ""
    echo "[OK] HELM Evaluation completed successfully."
    return 0
}

# Function to run ColBench evaluation
run_colbench() {
    echo "=========================================================="
    echo "  RUNNING: ColBench Aggregate Evaluation"
    echo "=========================================================="
    echo ""

    # Clean caches
    echo "[CLEAN] Removing caches..."
    rm -rf /tmp/_sae_embeddings_ckpt
    rm -f "${RESULT_DIR}/colbench_results.csv"

    echo "[RUN] Training ColBench models (Beta IRT)..."
    set +e
    (cd "${SCRIPT_DIR}" && python models.py --benchmark colbench --model beta --embedding-type pca 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "[ERROR] models.py --benchmark colbench exited with code ${rc}"
        return 1
    fi

    echo ""
    echo "[RUN] Generating ColBench plots..."
    set +e
    (cd "${SCRIPT_DIR}" && python plotting.py --plot colbench 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "[ERROR] plotting.py --plot colbench exited with code ${rc}"
        return 1
    fi

    echo ""
    echo "[OK] ColBench Evaluation completed successfully."
    return 0
}

# Function to run aggregate survey (n-holdout)
run_aggregate() {
    echo "=========================================================="
    echo "  RUNNING: Aggregate Survey (N-Holdout)"
    echo "=========================================================="
    echo ""

    # Clean caches
    echo "[CLEAN] Removing caches..."
    rm -rf /tmp/_sae_embeddings_ckpt
    rm -f "${RESULT_DIR}/convergence_results.csv"

    echo "[RUN] Running aggregate survey across n values..."
    set +e
    (cd "${SCRIPT_DIR}" && python aggregate.py --embedding-type pca --model beta 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "[ERROR] aggregate.py exited with code ${rc}"
        return 1
    fi

    echo ""
    echo "[RUN] Generating convergence plots..."
    set +e
    (cd "${SCRIPT_DIR}" && python plotting.py --plot convergence 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "[ERROR] plotting.py --plot convergence exited with code ${rc}"
        return 1
    fi

    echo ""
    echo "[OK] Aggregate Survey completed successfully."
    return 0
}

# Show menu
show_menu() {
    echo "Please select which evaluation to run:"
    echo ""
    echo "  1) HELM Evaluation"
    echo "     - Trains baseline models (Average, Rasch-IRT, Amortized variants)"
    echo "     - Uses Bernoulli IRT for binary response prediction"
    echo "     - Generates AUC comparison plot"
    echo ""
    echo "  2) ColBench Evaluation"
    echo "     - Trains on aggregated response matrices"
    echo "     - Uses Beta IRT for continuous [0,1] predictions"
    echo "     - Generates RMSE and AUC comparison plots"
    echo ""
    echo "  3) Aggregate Survey (N-Holdout)"
    echo "     - Runs experiments across varying n (1 to max response matrices)"
    echo "     - Shows performance improvement with more data"
    echo "     - Generates convergence plots"
    echo ""
    echo "  4) All evaluations (HELM + ColBench + Aggregate)"
    echo ""
    echo "  5) Exit"
    echo ""
}

# Parse command line or show menu
choice=""
if [ $# -gt 0 ]; then
    case "$1" in
        helm|HELM|1)
            choice="1"
            ;;
        colbench|COLBENCH|2)
            choice="2"
            ;;
        aggregate|AGGREGATE|survey|3)
            choice="3"
            ;;
        all|ALL|both|BOTH|4)
            choice="4"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [helm|colbench|aggregate|all]"
            exit 1
            ;;
    esac
else
    show_menu
    read -p "Enter your choice [1-5]: " choice
fi

# Create result directory
mkdir -p "${RESULT_DIR}"

# Execute based on choice
overall_ok=true
case "$choice" in
    1)
        run_helm || overall_ok=false
        ;;
    2)
        run_colbench || overall_ok=false
        ;;
    3)
        run_aggregate || overall_ok=false
        ;;
    4)
        run_helm || overall_ok=false
        echo ""
        run_colbench || overall_ok=false
        echo ""
        run_aggregate || overall_ok=false
        ;;
    5)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice: $choice"
        exit 1
        ;;
esac

# Summary
echo ""
echo "=========================================================="
echo "  OUTPUT SUMMARY"
echo "=========================================================="
echo ""
echo "  Output directory: ${RESULT_DIR}"
echo ""

echo "  Results (CSV):"
found=false
for f in "${RESULT_DIR}"/*.csv; do
    [ -e "$f" ] || continue
    found=true
    echo "    $(basename "$f")"
done
$found || echo "    (none)"

echo ""
echo "  Plots (PDF):"
found=false
for f in "${RESULT_DIR}"/*.pdf; do
    [ -e "$f" ] || continue
    found=true
    echo "    $(basename "$f")"
done
$found || echo "    (none)"

echo ""
if $overall_ok; then
    echo "All evaluations completed successfully."
else
    echo "Some evaluations had errors - check output above."
    exit 1
fi
