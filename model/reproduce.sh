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
NUM_SEEDS=100
FULL_SWEEP=false
ONLY_PLOT=false
PARALLEL=1
CLEAN_RESULTS=false
OVERRIDE_RESULTS=false

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
        --override)
            OVERRIDE_RESULTS=true
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
        check_file "amortized_irt_sae_bernoulli_pre_8_n_1.csv"
        check_file "amortized_irt_sae_beta_pre_max_n_max.csv"
        check_file "amortized_irt_pca_bernoulli_pre_8_n_1.csv"
        check_file "amortized_irt_pca_beta_pre_max_n_max.csv"
        check_file "amortized_irt_raw_bernoulli_pre_8_n_1.csv"
        check_file "amortized_irt_raw_beta_pre_max_n_max.csv"
    fi
    echo "[VERIFY] All required data files found."
else
    if ! $CLEAN_RESULTS && ! $OVERRIDE_RESULTS; then
        if [ -d "${RESULT_DIR}" ] && [ "$(ls -A ${RESULT_DIR})" ]; then
            read -p "[WARNING] Output directory (${RESULT_DIR}) is not empty. Do you want to [c]lean it, or [o]verride/continue? (c/o): " choice
            case "$choice" in 
              c|C|clean|Clean ) CLEAN_RESULTS=true ;;
              o|O|override|Override ) OVERRIDE_RESULTS=true ;;
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
    local tau=$4
    local pre=${5:-false}
    local seed=$6
    local out_dir=${7:-""}
    
    local cmd="python ${SCRIPT_DIR}/amortized_irt.py --embedding-type $emb --n-samples $n --model-type $model --lambda-tau $tau --seed $seed"
    if [[ "$pre" != "false" ]]; then
        cmd="$cmd --pre-revision $pre"
    fi
    if [[ -n "$out_dir" ]]; then
        mkdir -p "$out_dir"
        local suffix=""
        if [[ "$pre" != "false" ]]; then suffix="_pre_$pre"; fi
        local n_suffix="_n_$n"
        if [[ "$n" == "max" ]]; then n_suffix="_n_max"; fi
        local out_file="${out_dir}/amortized_irt_${emb}_${model}${suffix}${n_suffix}.csv"
        
        # Resume Check: See if this combination of seed & tau already exists
        if [[ -f "$out_file" ]]; then
            # We specifically grep for lines that match both lambda_tau and seed.
            # Format: seed,lambda_tau,... (exact columns depend on pandas export order)
            # Using awk to check if any row has BOTH the specific seed and tau.
            # Convert values to float format for safe comparison
            local exists=$(awk -F',' -v s="$seed" -v t="$tau" '
                NR>1 {
                    # Find column indexes on first line if we want to be robust, 
                    # but simple string match on the row is usually enough for these unique values.
                    # Or we just check if the line contains both (since seed is int, tau is float).
                    if ($0 ~ "(^|,)" s "(,|$)" && $0 ~ "(^|,)" t "(,|$)") {
                        print "1"; exit;
                    }
                }
            ' "$out_file")
            
            if [[ "$exists" == "1" ]]; then
                echo " [SKIP] $emb (N=$n, $model) Tau=$tau Seed=$seed already completed."
                return 0
            fi
        fi
        
        cmd="$cmd --output $out_file"
    fi
    if [[ "$PARALLEL" -gt 1 ]]; then
        local gpu_id=$(( RANDOM % NUM_GPUS ))
        local full_cmd="echo ' -> Running: $emb (N=$n, $model) Tau=$tau Seed=$seed on GPU $gpu_id' && CUDA_VISIBLE_DEVICES=$gpu_id $cmd"
        echo "$full_cmd" >> "$CMD_FILE"
    else
        echo " -> Running: $emb (N=$n, $model) Tau=$tau Seed=$seed"
        eval "$cmd"
    fi
}

# ── Execution ───────────────────────────────────────────────────────────────

CMD_FILE=$(mktemp)
trap 'rm -f $CMD_FILE' EXIT

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [[ $NUM_GPUS -eq 0 ]]; then NUM_GPUS=1; fi

if ! $ONLY_PLOT; then
    run_tau_sweep() {
        local emb=$1
        local n=$2
        local model=$3
        local base_tau=$4
        local pre=${5:-false}
        
        # If full sweep, run 11 steps from 0.5x to 1.5x, otherwise just run base_tau
        local taus="$base_tau"
        if $FULL_SWEEP; then
            taus=$(awk -v t="$base_tau" 'BEGIN{for(i=0.5; i<=1.501; i+=0.1) printf "%.5f ", t*i}')
        fi
        
        for tau in $taus; do
            for seed in $SEEDS; do
                run_exp $emb $n $model $tau $pre $seed "${RESULT_DIR}"
            done
        done
    }

    echo "[MODE] Running Experiments..."
    
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
        run_tau_sweep sae 1 bernoulli 0.0159 8
        run_tau_sweep sae max beta 0.16 max
        run_tau_sweep pca 1 bernoulli 0.0155 8
        run_tau_sweep pca max beta 0.054 max
        run_tau_sweep raw 1 bernoulli 0.0151 8
        run_tau_sweep raw max beta 0.029 max
    fi
fi

if [[ "$PARALLEL" -gt 1 ]] && [[ -s "$CMD_FILE" ]]; then
    echo ""
    echo "[MODE] Executing queued jobs with $PARALLEL parallel workers across $NUM_GPUS GPUs..."
    cat "$CMD_FILE" | xargs -d '\n' -P "$PARALLEL" -I {} bash -c "{}"
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
