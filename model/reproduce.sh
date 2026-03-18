#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Run all experiments from scratch and generate plots
#
# Usage:
#   ./reproduce.sh          # Quick run (single seed for main configs)
#   ./reproduce.sh --full   # Full sweep (100 seeds for all configs - SOTA)
###############################################################################

eval "$(conda shell.bash hook)"
conda activate hal
echo "[ENV] conda env: hal (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
RESULT_DIR="${SCRIPT_DIR}/result"
# ── Parameters ─────────────────────────────────────────────────────────────
SEEDS="42"
NUM_SEEDS=50
SHARED_TAUS="0.002 0.004 0.005 0.006 0.008 0.010 0.012 0.014 0.015 0.0151 0.0155 0.0159 0.016 0.018 0.020 0.022 0.024 0.025 0.026 0.028 0.029 0.030 0.032 0.034 0.035 0.036 0.038 0.040 0.042 0.044 0.045 0.046 0.048 0.050 0.052 0.0535 0.054 0.055 0.056 0.058 0.060 0.062 0.064 0.065 0.066 0.068 0.070 0.072 0.074 0.075 0.076 0.078 0.080 0.082 0.084 0.085 0.086 0.088 0.090 0.092 0.094 0.095 0.096 0.098 0.100 0.105 0.110 0.115 0.120 0.125 0.130 0.135 0.140 0.145 0.150 0.155 0.160 0.165 0.170 0.175 0.180 0.185 0.190 0.195 0.200 0.210 0.220 0.230 0.250 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 5.0 10.0 20.0 30.0 50.0 75.0 100.0 200.0 500.0 1000.0"
FULL_SWEEP=false
ONLY_PLOT=false
PARALLEL=1
CLEAN_RESULTS=false
OVERRIDE_RESULTS=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_SWEEP=true
            # Generate seed list up to NUM_SEEDS-1
            SEEDS="$(seq -s ' ' 0 $((NUM_SEEDS - 1)))"
            shift
            ;;
        --parallel)
            if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                PARALLEL="$2"
                shift 2
            else
                echo "Error: --parallel requires a numeric value"
                exit 1
            fi
            ;;
        --clean)
            CLEAN_RESULTS=true
            shift
            ;;
        --continue)
            OVERRIDE_RESULTS=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if $FULL_SWEEP; then
    echo "[MODE] Configured for FULL sweep ($NUM_SEEDS seeds)..."
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
        check_file "amortized_irt_sae_bernoulli_pre_32_n_1.csv"
        check_file "amortized_irt_sae_beta_pre_max_n_max.csv"
        check_file "amortized_irt_pca_bernoulli_pre_32_n_1.csv"
        check_file "amortized_irt_pca_beta_pre_max_n_max.csv"
        check_file "amortized_irt_raw_bernoulli_pre_32_n_1.csv"
        check_file "amortized_irt_raw_beta_pre_max_n_max.csv"
    fi
    echo "[VERIFY] All required data files found."
else
    if ! $CLEAN_RESULTS && ! $OVERRIDE_RESULTS; then
        if [ -d "${RESULT_DIR}" ] && [ "$(ls -A ${RESULT_DIR})" ]; then
            read -p "[WARNING] Output directory (${RESULT_DIR}) is not empty. Do you want to [o]verwrite, or [c]ontinue? (o/c): " choice
            case "$choice" in 
              o|O|overwrite|Overwrite ) CLEAN_RESULTS=true ;;
              c|C|continue|Continue ) OVERRIDE_RESULTS=true ;;
              * ) echo "Invalid choice. Exiting."; exit 1 ;;
            esac
        fi
    fi

    if $CLEAN_RESULTS; then
        echo "[CLEAN] Removing previous results from ${RESULT_DIR} …"
        rm -rf "${RESULT_DIR}"
        mkdir -p "${RESULT_DIR}"
    else
        echo "[OVERRIDE] Keeping previous results. New results will be appended/overridden in ${RESULT_DIR}."
        mkdir -p "${RESULT_DIR}"
    fi

    echo "[CLEAN] Verifying integrity of existing CSVs..."
    # Quick purge of any corrupted CSVs that would break the append logic later
    python -c "
import os, pandas as pd
for f in os.listdir('${RESULT_DIR}'):
    if not f.endswith('.csv'): continue
    p = os.path.join('${RESULT_DIR}', f)
    try:
        pd.read_csv(p)
    except Exception as e:
        print(f'   -> Removed corrupted file: {f}')
        os.remove(p)
"
fi
echo ""

# ── Run Function ─────────────────────────────────────────────────────────────
run_exp() {
    local emb=$1
    local n=$2
    local model=$3
    local taus=$4
    local pre=${5:-false}
    local seeds=$6
    local out_dir=${7:-""}
    local no_tau=${8:-false}
    local save_weights=${9:-false}
    local j_pct=${10:-1.0}
    
    # Replace spaces with commas for Python arg parsing
    local taus_csv=${taus// /,}
    local seeds_csv=${seeds// /,}

    local cmd="python ${SCRIPT_DIR}/amortized_irt.py --embedding-type $emb --n-samples $n --model-type $model --lambda-tau $taus_csv --seed $seeds_csv"
    if [[ "$pre" != "false" ]]; then
        cmd="$cmd --pre-revision $pre"
    fi
    if [[ "$no_tau" == "true" ]]; then
        cmd="$cmd --no-tau"
    fi
    if $QUIET; then
        cmd="$cmd --quiet"
    fi
    if [[ "$save_weights" == "true" ]]; then
        cmd="$cmd --save-weights"
    fi
    if [[ "$j_pct" != "1.0" ]]; then
        cmd="$cmd --j-percentage $j_pct"
    fi
    if [[ -n "$out_dir" ]]; then
        mkdir -p "$out_dir"
        local suffix=""
        if [[ "$pre" != "false" ]]; then suffix="_pre_$pre"; fi
        local n_suffix="_n_$n"
        if [[ "$n" == "max" ]]; then n_suffix="_n_max"; fi
        local notau_suffix=""
        if [[ "$no_tau" == "true" ]]; then notau_suffix="_notau"; fi
        local j_suffix=""
        if [[ "$j_pct" != "1.0" ]]; then j_suffix="_j${j_pct}"; fi
        local out_file="${out_dir}/amortized_irt_${emb}_${model}${suffix}${n_suffix}${notau_suffix}${j_suffix}.csv"
        
        cmd="$cmd --output $out_file"
    fi
    if [[ "$PARALLEL" -gt 1 ]]; then
        cmd="$cmd --parallel $PARALLEL"
    fi
    
    echo " -> Running: $emb (N=$n, $model) Taus=[$taus_csv] Seeds=[$seeds_csv]"
    eval "$cmd"
}

# ── Execution ───────────────────────────────────────────────────────────────

if ! $ONLY_PLOT; then
    run_tau_sweep() {
        local emb=$1
        local n=$2
        local model=$3
        local base_tau=$4
        local pre=${5:-false}
        local j_pct=${6:-1.0}
        
        # If full sweep, run the shared global tau sweep, otherwise just run base_tau
        local taus="$base_tau"
        if $FULL_SWEEP; then
            taus="$SHARED_TAUS"
        fi
        
        run_exp $emb $n $model "$taus" $pre "$SEEDS" "${RESULT_DIR}" false false $j_pct
    }

    echo "[MODE] Running Experiments..."

    # Standalone Baselines (Rasch 2PL and Non-Amortized MIRT)
    # These are run at the top to ensure they are available for comparison independently.
    echo " -> Running Standalone Baselines (N=32)..."
    run_exp rasch_2pl 32 bernoulli 1.0 false "$SEEDS" "${RESULT_DIR}"
    run_exp rasch_2pl max beta 1.0 false "$SEEDS" "${RESULT_DIR}"
    run_exp nonamortised_mirt 32 bernoulli 1.0 false "$SEEDS" "${RESULT_DIR}"
    run_exp nonamortised_mirt max beta 1.0 false "$SEEDS" "${RESULT_DIR}"
    
    # [SCALING LAW]: Item Scaling Study (N=32)
    # We fix N=32 and sweep J-percentages (0.1 through 0.9)
    if $FULL_SWEEP; then
        echo " -> Starting Item Scaling Law Study (N=32, Full Tau Sweep)..."
        for j in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            run_tau_sweep sae 1 bernoulli 0.0159 32 $j
            run_tau_sweep sae max beta 0.16 32 $j
            run_tau_sweep pca 1 bernoulli 0.0155 32 $j
            run_tau_sweep pca max beta 0.054 32 $j
            run_tau_sweep raw 1 bernoulli 0.0151 32 $j
            run_tau_sweep raw max beta 0.029 32 $j
            
            # Baselines (No Tau Sweep, matching N=32 setup)
            run_exp rasch_2pl 1 bernoulli 1.0 32 "$SEEDS" "${RESULT_DIR}" false false $j
            run_exp rasch_2pl max beta 1.0 32 "$SEEDS" "${RESULT_DIR}" false false $j
            run_exp nonamortised_mirt 1 bernoulli 1.0 32 "$SEEDS" "${RESULT_DIR}" false false $j
            run_exp nonamortised_mirt max beta 1.0 32 "$SEEDS" "${RESULT_DIR}" false false $j
        done
    fi

    # 0. Primary Model Exports (Required for Interpretability Plots)
    echo " -> Exporting primary SAE weights..."
    run_exp sae max beta 0.16 max "$SEEDS" "${RESULT_DIR}" false true
    run_exp sae max beta 0.0535 false "$SEEDS" "${RESULT_DIR}" false true

    # 1. PCA Embeddings
    run_tau_sweep pca max beta 0.054 false
    if $FULL_SWEEP; then
        run_tau_sweep pca 1 bernoulli 0.0155 false
    fi
    
    # 2. SAE Embeddings
    run_tau_sweep sae max beta 0.0535 false
    if $FULL_SWEEP; then
        run_tau_sweep sae 1 bernoulli 0.0159 false
    fi
    
    # 3. RAW Embeddings
    run_tau_sweep raw max beta 0.029 false
    if $FULL_SWEEP; then
        run_tau_sweep raw 1 bernoulli 0.0151 false
    fi
    
    # 4. Pre-Revision Checks (Full Sweep only)
    if $FULL_SWEEP; then
        # SAE (Symmetric Sweep)
        run_tau_sweep sae 1 bernoulli 0.0159 4
        run_tau_sweep sae 1 bernoulli 0.0159 8
        run_tau_sweep sae 1 bernoulli 0.0159 16
        run_tau_sweep sae 1 bernoulli 0.0159 32
        run_tau_sweep sae 1 bernoulli 0.0159 64
        run_tau_sweep sae 1 bernoulli 0.0159 max
        run_tau_sweep sae max beta 0.16 max
        run_tau_sweep sae max beta 0.16 64
        run_tau_sweep sae max beta 0.16 32
        run_tau_sweep sae max beta 0.16 16
        run_tau_sweep sae max beta 0.16 8
        run_tau_sweep sae max beta 0.16 4
        
        # PCA/RAW (Main stage only for baseline)
        run_tau_sweep pca 1 bernoulli 0.0155 4
        run_tau_sweep pca 1 bernoulli 0.0155 8
        run_tau_sweep pca 1 bernoulli 0.0155 16
        run_tau_sweep pca 1 bernoulli 0.0155 32
        run_tau_sweep pca 1 bernoulli 0.0155 64
        run_tau_sweep pca 1 bernoulli 0.0155 max
        run_tau_sweep pca max beta 0.054 max
        run_tau_sweep pca max beta 0.054 64
        run_tau_sweep pca max beta 0.054 32
        run_tau_sweep pca max beta 0.054 16
        run_tau_sweep pca max beta 0.054 8
        run_tau_sweep pca max beta 0.054 4

        run_tau_sweep raw 1 bernoulli 0.0151 4
        run_tau_sweep raw 1 bernoulli 0.0151 8
        run_tau_sweep raw 1 bernoulli 0.0151 16
        run_tau_sweep raw 1 bernoulli 0.0151 32
        run_tau_sweep raw 1 bernoulli 0.0151 64
        run_tau_sweep raw 1 bernoulli 0.0151 max
        run_tau_sweep raw max beta 0.029 max
        run_tau_sweep raw max beta 0.029 64
        run_tau_sweep raw max beta 0.029 32
        run_tau_sweep raw max beta 0.029 16
        run_tau_sweep raw max beta 0.029 8
        run_tau_sweep raw max beta 0.029 4
        
        # Baselines Scaling (N-Sweep)
        echo " -> Starting Baseline Scaling Study (N-Sweep, No Tau)..."
        for n in 4 8 16 32 64 max; do
            run_exp rasch_2pl 1 bernoulli 1.0 $n "$SEEDS" "${RESULT_DIR}"
            run_exp rasch_2pl max beta 1.0 $n "$SEEDS" "${RESULT_DIR}"
            run_exp nonamortised_mirt 1 bernoulli 1.0 $n "$SEEDS" "${RESULT_DIR}"
            run_exp nonamortised_mirt max beta 1.0 $n "$SEEDS" "${RESULT_DIR}"
        done
        
        # 5. Ablation Studies (Full Sweep only)
        # Setup 1: No TAU (w/ SAE, PCA, RAW embeddings)
        run_exp sae max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp pca max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp raw max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true
        # Setup 2: No Embeddings (w/ TAU search)
        run_exp ones max beta "$SHARED_TAUS" false "$SEEDS" "${RESULT_DIR}" false
        # Setup 3: No TAU & No Embeddings (Beta N=max)
        run_exp ones max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true

        # Setup 4: No TAU (w/ SAE, PCA, RAW) - Bernoulli N=1
        run_exp sae 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp pca 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp raw 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true

        # Setup 5: No Embeddings (w/ TAU search) - Bernoulli N=1
        run_exp ones 1 bernoulli "$SHARED_TAUS" false "$SEEDS" "${RESULT_DIR}" false

        # Setup 6: No TAU & No Embeddings - Bernoulli N=1
        run_exp ones 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true

        # Setup 7: No TAU (w/ SAE, PCA, RAW) - Pre-max N=max
        run_exp sae max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true
        run_exp pca max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true
        run_exp raw max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true

        # Setup 8: No Embeddings (w/ TAU search) - Pre-max N=max
        run_exp ones max beta "$SHARED_TAUS" max "$SEEDS" "${RESULT_DIR}" false

        # Setup 9: No TAU & No Embeddings - Pre-max N=max
        run_exp ones max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true

        # Setup 10: No TAU (w/ SAE, PCA, RAW) - Pre-32 N=1
        run_exp sae 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true
        run_exp pca 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true
        run_exp raw 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true

        # Setup 11: No Embeddings (w/ TAU search) - Pre-32 N=1
        run_exp ones 1 bernoulli "$SHARED_TAUS" 32 "$SEEDS" "${RESULT_DIR}" false

        # Setup 12: No TAU & No Embeddings - Pre-32 N=1
        run_exp ones 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true
    fi
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
