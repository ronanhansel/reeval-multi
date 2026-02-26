#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Run all experiments from scratch and generate plots
#
# Usage:
#   ./reproduce.sh          # Quick run (single seed for main configs)
#   ./reproduce.sh --full   # Full sweep (10 seeds for all configs - SOTA)
###############################################################################

eval "$(conda shell.bash hook)"
conda activate hal
echo "[ENV] conda env: hal (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
RESULT_DIR="${SCRIPT_DIR}/result"

# ── Parameters ─────────────────────────────────────────────────────────────
SEEDS="42"
FULL_SWEEP=false
ONLY_PLOT=false

for arg in "$@"; do
    case $arg in
        --full)
            FULL_SWEEP=true
            SEEDS="42 123 789 2024 1337 555 666 777 888 999"
            ;;
        --plot)
            ONLY_PLOT=true
            ;;
    esac
done

if $FULL_SWEEP; then
    echo "[MODE] Configured for FULL sweep (10 seeds)..."
else
    echo "[MODE] Configured for QUICK reproduction (1 seed)..."
fi

if $ONLY_PLOT; then
    echo "[MODE] ONLY_PLOT enabled. Skipping experiment execution."
fi

# ── HuggingFace cache configuration ─────────────────────────────────────────
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "${HF_HOME}"

echo "=========================================================="
echo "  REPRODUCE — Amortized IRT"
echo "=========================================================="
echo "  Working dir : ${SCRIPT_DIR}"
echo "  Output dir  : ${RESULT_DIR}"
echo ""

# ── Clean/Verify results ───────────────────────────────────────────────────
if $ONLY_PLOT; then
    echo "[VERIFY] Checking for required data files..."
    check_file() {
        if [[ ! -f "${RESULT_DIR}/$1" ]]; then
            echo "ERROR: Missing required result file: $1"
            exit 1
        fi
    }
    
    # Core Post-Revision files
    check_file "amortized_irt_sae_beta_n_max.csv"
    check_file "amortized_irt_sae_bernoulli_n_1.csv"
    check_file "amortized_irt_pca_beta_n_max.csv"
    check_file "amortized_irt_pca_bernoulli_n_1.csv"
    check_file "amortized_irt_raw_beta_n_max.csv"
    check_file "amortized_irt_raw_bernoulli_n_1.csv"
    
    if $FULL_SWEEP; then
        # Pre-Revision files
        check_file "amortized_irt_sae_bernoulli_pre_8_n_1.csv"
        check_file "amortized_irt_sae_beta_pre_max_n_max.csv"
        check_file "amortized_irt_pca_bernoulli_pre_8_n_1.csv"
        check_file "amortized_irt_pca_beta_pre_max_n_max.csv"
        check_file "amortized_irt_raw_bernoulli_pre_8_n_1.csv"
        check_file "amortized_irt_raw_beta_pre_max_n_max.csv"
    fi
    echo "[VERIFY] All required data files found."
else
    echo "[CLEAN] Removing previous results …"
    rm -rf "${RESULT_DIR}"
    mkdir -p "${RESULT_DIR}"
fi
echo ""

# ── Run Function ─────────────────────────────────────────────────────────────
run_exp() {
    local emb=$1
    local n=$2
    local model=$3
    local tau=$4
    local pre=${5:-false}
    local seed=$6
    
    local cmd="python ${SCRIPT_DIR}/amortized_irt.py --embedding-type $emb --n-samples $n --model-type $model --lambda-tau $tau --seed $seed"
    if [[ "$pre" != "false" ]]; then
        cmd="$cmd --pre-revision $pre"
    fi
    
    echo " -> Running: $emb (N=$n, $model) Tau=$tau Seed=$seed"
    eval $cmd
}

# ── Execution ───────────────────────────────────────────────────────────────

if ! $ONLY_PLOT; then
    for seed in $SEEDS; do
        echo "--- Seed $seed ---"
        
        # 1. PCA Embeddings
        run_exp pca max beta 0.054 false $seed
        if $FULL_SWEEP; then
            run_exp pca 1 bernoulli 0.0155 false $seed
        fi
        
        # 2. SAE Embeddings
        run_exp sae max beta 0.0535 false $seed
        if $FULL_SWEEP; then
            run_exp sae 1 bernoulli 0.0159 false $seed
        fi
        
        # 3. RAW Embeddings
        run_exp raw max beta 0.029 false $seed
        if $FULL_SWEEP; then
            run_exp raw 1 bernoulli 0.0151 false $seed
        fi
        
        # 4. Pre-Revision Checks (Full Sweep only)
        if $FULL_SWEEP; then
            # SAE Pre-Revision
            run_exp sae 1 bernoulli 0.0159 8 $seed
            run_exp sae max beta 0.16 max $seed
            
            # PCA Pre-Revision
            run_exp pca 1 bernoulli 0.0155 8 $seed
            run_exp pca max beta 0.054 max $seed
            
            # RAW Pre-Revision
            run_exp raw 1 bernoulli 0.0151 8 $seed
            run_exp raw max beta 0.029 max $seed
        fi
    done
fi

# ── Generate plots ───────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  GENERATING PLOTS"
echo "=========================================================="
cd "${REPO_ROOT}"
PYTHONPATH=. python3 -m model.plotting.main --all

echo ""
echo "=========================================================="
echo "  REPRODUCTION COMPLETE"
echo "=========================================================="
echo "Plots saved in paper/figures/"
echo "CSV results in model/result/"
