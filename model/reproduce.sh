#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Run all experiments from scratch and generate plots
#
# Usage:
#   ./reproduce.sh          # Quick run (single seed for main configs)
#   ./reproduce.sh --full   # Full sweep (50 seeds for all configs)
###############################################################################

eval "$(conda shell.bash hook)"
conda activate hal
echo "[ENV] conda env: hal (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
AMORTIZED_IRT_CMD="PYTHONPATH=${REPO_ROOT} python -m model.amortized_irt"
SUPPORT_THINNING_REBUILD_CMD="PYTHONPATH=${REPO_ROOT} python -m model.analysis.rebuild_support_thinning_summary"
RESULT_ROOT="${SCRIPT_DIR}/result"
MAIN_RESULT_DIR="${RESULT_ROOT}/main"
RESULT_DIR="${MAIN_RESULT_DIR}"
BASELINE_CSV="${RESULT_DIR}/baselines/baseline_metrics.csv"
MIRT_SWEEP_CSV="${RESULT_DIR}/baselines/mirt_sweep.csv"
THIN_RESULT_DIR="${RESULT_ROOT}/support_thinning_study"
SAMPLE_SIZE_RESULT_DIR="${RESULT_ROOT}/sample_size_study"
RESTORE_COMMIT="${RESTORE_COMMIT:-1b3737c3ddca6e8554948f967cb21e7fab2ef8aa}"

# ── Parameters ───────────────────────────────────────────────────────────────
SEEDS="42"
NUM_SEEDS=50
SHARED_TAUS="0.002 0.004 0.005 0.006 0.008 0.010 0.012 0.014 0.015 0.0151 0.0155 0.0159 0.016 0.018 0.020 0.022 0.024 0.025 0.026 0.028 0.029 0.030 0.032 0.034 0.035 0.036 0.038 0.040 0.042 0.044 0.045 0.046 0.048 0.050 0.052 0.0535 0.054 0.055 0.056 0.058 0.060 0.062 0.064 0.065 0.066 0.068 0.070 0.072 0.074 0.075 0.076 0.078 0.080 0.082 0.084 0.085 0.086 0.088 0.090 0.092 0.094 0.095 0.096 0.098 0.100 0.105 0.110 0.115 0.120 0.125 0.130 0.135 0.140 0.145 0.150 0.155 0.160 0.165 0.170 0.175 0.180 0.185 0.190 0.195 0.200 0.210 0.220 0.230 0.250 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 5.0 10.0 20.0 30.0 50.0 75.0 100.0 200.0 500.0 1000.0"
FULL_SWEEP=false
ONLY_PLOT=false
PARALLEL=1
CLEAN_RESULTS=false
OVERRIDE_RESULTS=false
QUIET=false
SUPPORT_THINNING_STUDY=false
SAMPLE_SIZE_STUDY=false
MIRT_DIM_MIN=2
MIRT_DIM_MAX=30
THIN_RETENTIONS=("0.05" "0.1" "0.25" "0.5" "1.0")
THIN_ARAF_EMBEDDINGS=("raw" "pca")
THIN_KNN_EMBEDDINGS=("raw" "pca")
THIN_K_VALUES=("5" "10" "20" "50")
SAMPLE_USER_LEVELS=("4" "8" "16" "32" "max")
SAMPLE_J_LEVELS=("0.1" "0.3" "0.5" "0.7" "0.9")
MAIN_KNN_GRID="${MAIN_KNN_GRID:-5,10,20,50}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_SWEEP=true
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
        --support-thinning-study)
            SUPPORT_THINNING_STUDY=true
            shift
            ;;
        --sample-size-study|--sample-size|--sample_size)
            SAMPLE_SIZE_STUDY=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done
RUN_SUPPORT_THINNING_STUDY=false
if $SUPPORT_THINNING_STUDY; then
    RUN_SUPPORT_THINNING_STUDY=true
fi
RUN_MAIN_EXPERIMENTS=true
if $SUPPORT_THINNING_STUDY || $SAMPLE_SIZE_STUDY; then
    RUN_MAIN_EXPERIMENTS=false
fi
RUN_SAMPLE_SIZE_STUDY=false
if $SAMPLE_SIZE_STUDY || { $FULL_SWEEP && $RUN_MAIN_EXPERIMENTS; }; then
    RUN_SAMPLE_SIZE_STUDY=true
fi

THIN_ONLY_MODE=false
if $SUPPORT_THINNING_STUDY && ! $FULL_SWEEP; then
    THIN_ONLY_MODE=true
fi

STUDY_ONLY_MODE=false
if ! $RUN_MAIN_EXPERIMENTS; then
    STUDY_ONLY_MODE=true
fi

ACTIVE_RESULT_DIRS=()
if $RUN_MAIN_EXPERIMENTS; then
    ACTIVE_RESULT_DIRS+=("${MAIN_RESULT_DIR}")
fi
if ! $RUN_MAIN_EXPERIMENTS; then
    if $RUN_SUPPORT_THINNING_STUDY; then
        ACTIVE_RESULT_DIRS+=("${THIN_RESULT_DIR}")
    fi
    if $RUN_SAMPLE_SIZE_STUDY; then
        ACTIVE_RESULT_DIRS+=("${SAMPLE_SIZE_RESULT_DIR}")
    fi
fi

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
echo "  Main dir    : ${MAIN_RESULT_DIR}"
echo "  Result root : ${RESULT_ROOT}"
echo "  MIRT sweep  : ${MIRT_DIM_MIN}-${MIRT_DIM_MAX} dims (cache: ${MIRT_SWEEP_CSV})"
echo ""

restore_main_results_from_commit() {
    local commit_ref="${1:-${RESTORE_COMMIT}}"
    local target_dir="${MAIN_RESULT_DIR}"
    echo "[RESTORE] Restoring canonical main results from ${commit_ref} into ${target_dir}"
    python - <<PY
import subprocess
from pathlib import Path

repo_root = Path(r"""${REPO_ROOT}""")
commit_ref = r"""${commit_ref}"""
target_dir = Path(r"""${target_dir}""")
target_dir.mkdir(parents=True, exist_ok=True)

paths = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", commit_ref, "model/result"],
    cwd=repo_root,
    text=True,
).splitlines()

restored = 0
for rel in paths:
    if not (
        rel.startswith("model/result/amortized_irt_")
        or rel.startswith("model/result/baselines/")
        or rel in {
            "model/result/comprehensive_results.csv",
            "model/result/comprehensive_results.md",
        }
    ):
        continue

    suffix = Path(rel).relative_to("model/result")
    out_path = target_dir / suffix
    if out_path.exists():
        continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(["git", "show", f"{commit_ref}:{rel}"], cwd=repo_root)
    out_path.write_bytes(data)
    restored += 1

print(f"Restored {restored} file(s).")
PY
}

# ── Clean/Verify results ─────────────────────────────────────────────────────
if $ONLY_PLOT; then
    restore_main_results_from_commit
    echo "[VERIFY] Checking for required data files..."
    check_file() {
        if [[ ! -f "${MAIN_RESULT_DIR}/$1" ]]; then
            echo "ERROR: Missing required result file: $1"
            exit 1
        fi
    }

    check_file "amortized_irt_sae_beta_n_max.csv"
    check_file "amortized_irt_sae_bernoulli_n_1.csv"
    check_file "amortized_irt_pca_beta_n_max.csv"
    check_file "amortized_irt_pca_bernoulli_n_1.csv"
    check_file "amortized_irt_raw_beta_n_max.csv"
    check_file "amortized_irt_raw_bernoulli_n_1.csv"

    if $FULL_SWEEP; then
        check_file "amortized_irt_sae_bernoulli_pre_32_n_1.csv"
        check_file "amortized_irt_sae_beta_pre_max_n_max.csv"
        check_file "amortized_irt_pca_bernoulli_pre_32_n_1.csv"
        check_file "amortized_irt_pca_beta_pre_max_n_max.csv"
        check_file "amortized_irt_raw_bernoulli_pre_32_n_1.csv"
        check_file "amortized_irt_raw_beta_pre_max_n_max.csv"
    fi
    echo "[VERIFY] All required data files found."
else
    active_dirs_nonempty=false
    for target_dir in "${ACTIVE_RESULT_DIRS[@]}"; do
        if [ -d "${target_dir}" ] && [ "$(ls -A "${target_dir}" 2>/dev/null)" ]; then
            active_dirs_nonempty=true
            break
        fi
    done

    if $FULL_SWEEP && ! $OVERRIDE_RESULTS; then
        CLEAN_RESULTS=true
        echo "[CLEAN] FULL sweep requested. Resetting active result directories: ${ACTIVE_RESULT_DIRS[*]}"
    fi

    if ! $CLEAN_RESULTS && ! $OVERRIDE_RESULTS; then
        if $active_dirs_nonempty; then
            read -p "[WARNING] Active output directories are not empty (${ACTIVE_RESULT_DIRS[*]}). Do you want to [o]verwrite, or [c]ontinue? (o/c): " choice
            case "$choice" in
              o|O|overwrite|Overwrite) CLEAN_RESULTS=true ;;
              c|C|continue|Continue) OVERRIDE_RESULTS=true ;;
              *) echo "Invalid choice. Exiting."; exit 1 ;;
            esac
        fi
    fi

    if $CLEAN_RESULTS; then
        for target_dir in "${ACTIVE_RESULT_DIRS[@]}"; do
            echo "[CLEAN] Removing previous results from ${target_dir} ..."
            rm -rf "${target_dir}"
            mkdir -p "${target_dir}"
        done
    else
        echo "[OVERRIDE] Keeping previous results in active target dirs: ${ACTIVE_RESULT_DIRS[*]}"
        for target_dir in "${ACTIVE_RESULT_DIRS[@]}"; do
            mkdir -p "${target_dir}"
        done
    fi

    echo "[CLEAN] Verifying integrity of existing CSVs..."
    active_dirs_py=""
    for target_dir in "${ACTIVE_RESULT_DIRS[@]}"; do
        active_dirs_py+="r'''${target_dir}''',"
    done
    python -c "
import os
from pathlib import Path
import pandas as pd

targets = [Path(p) for p in [${active_dirs_py}]]
if not targets:
    targets = [Path('${MAIN_RESULT_DIR}')]
for root in targets:
    if not root.exists():
        continue
    for p in root.rglob('*.csv'):
        try:
            pd.read_csv(p)
        except Exception:
            print(f'   -> Removed corrupted file: {p}')
            p.unlink(missing_ok=True)
"
fi
echo ""

if $STUDY_ONLY_MODE; then
    echo "[BASELINE] Study-only mode: skipping global main-result baseline migration."
else
    echo "[BASELINE] Migrating existing CSVs to separated baseline schema..."
    baseline_migrate_cmd="${AMORTIZED_IRT_CMD} --migrate-all-csvs --migrate-source-dir ${MAIN_RESULT_DIR} --baseline-output ${BASELINE_CSV} --mirt-sweep-output ${MIRT_SWEEP_CSV}"
    if $QUIET; then
        baseline_migrate_cmd="$baseline_migrate_cmd --quiet"
    fi
    eval "$baseline_migrate_cmd"
fi

# ── Run Functions ────────────────────────────────────────────────────────────
run_baseline() {
    local n=$1
    local model=$2
    local pre=${3:-false}
    local seeds=$4
    local j_pct=${5:-1.0}
    local baseline_emb=${6:-raw}
    local knn_k=${7:-10}
    local baseline_profile=${8:-full}
    local train_retention=${9:-1.0}
    local cross_revision_post_binary=${10:-false}
    local user_count=${11:-""}
    local pre_match="none"
    if [[ "$pre" != "false" && "$pre" != "max" ]]; then
        pre_match="transport_binary_strength"
    fi

    local seeds_csv=${seeds// /,}
    local cmd="${AMORTIZED_IRT_CMD} --baseline-only --embedding-type ${baseline_emb} --baseline-embedding-type ${baseline_emb} --knn-k ${knn_k} --knn-k-grid ${MAIN_KNN_GRID} --baseline-profile ${baseline_profile} --n-samples $n --model-type $model --seed $seeds_csv --lambda-tau 1.0 --baseline-output ${BASELINE_CSV} --mirt-sweep-output ${MIRT_SWEEP_CSV} --mirt-dim-min ${MIRT_DIM_MIN} --mirt-dim-max ${MIRT_DIM_MAX}"

    if [[ "$pre" != "false" ]]; then
        cmd="$cmd --pre-revision $pre"
    fi
    if [[ "$j_pct" != "1.0" ]]; then
        cmd="$cmd --j-percentage $j_pct"
    fi
    if [[ "$train_retention" != "1.0" ]]; then
        cmd="$cmd --train-retention $train_retention"
    fi
    if [[ "$cross_revision_post_binary" == "true" ]]; then
        cmd="$cmd --cross-revision-post-binary"
    fi
    if [[ -n "$user_count" ]]; then
        cmd="$cmd --user-count ${user_count}"
    fi
    if [[ "$PARALLEL" -gt 1 ]]; then
        cmd="$cmd --parallel $PARALLEL"
    fi
    if $QUIET; then
        cmd="$cmd --quiet"
    fi

    echo " -> Baseline cache: n=$n model=$model pre=$pre match=${pre_match} eff_pre=${pre:-none} users=${user_count:-all} j=$j_pct retention=${train_retention} base_emb=${baseline_emb} k=${knn_k} profile=${baseline_profile} seeds=[$seeds_csv]"
    eval "$cmd"
}

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
    local pair_efficiency_output=${11:-""}
    local neighbor_support_output=${12:-""}
    local support_thinning_output=${13:-""}
    local outlier_robustness_output=${14:-""}
    local train_retention=${15:-1.0}
    local baseline_emb=${16:-raw}
    local knn_k=${17:-10}
    local baseline_profile=${18:-full}
    local precompute_baseline=${19:-true}
    local cross_revision_post_binary=${20:-false}
    local cross_revision_araf_mode=${21:-transfer}
    local user_count=${22:-""}
    local pre_match="none"
    if [[ "$pre" != "false" && "$pre" != "max" ]]; then
        pre_match="transport_binary_strength"
    fi

    # Precompute/reuse baselines for all amortized-style runs.
    if [[ "$emb" != "rasch_2pl" && "$emb" != "nonamortised_mirt" && "$precompute_baseline" == "true" ]]; then
        run_baseline "$n" "$model" "$pre" "$seeds" "$j_pct" "$baseline_emb" "$knn_k" "$baseline_profile" "$train_retention" "$cross_revision_post_binary" "$user_count"
    fi

    local taus_csv=${taus// /,}
    local seeds_csv=${seeds// /,}

    local cmd="${AMORTIZED_IRT_CMD} --embedding-type $emb --baseline-embedding-type ${baseline_emb} --knn-k ${knn_k} --knn-k-grid ${MAIN_KNN_GRID} --baseline-profile ${baseline_profile} --n-samples $n --model-type $model --lambda-tau $taus_csv --seed $seeds_csv --baseline-output ${BASELINE_CSV} --mirt-sweep-output ${MIRT_SWEEP_CSV} --mirt-dim-min ${MIRT_DIM_MIN} --mirt-dim-max ${MIRT_DIM_MAX}"
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
        local baseline_suffix=""
        if [[ "${baseline_emb}" != "raw" || "${knn_k}" != "10" ]]; then
            baseline_suffix="_b${baseline_emb}_k${knn_k}"
        fi
        local user_suffix=""
        if [[ -n "${user_count}" ]]; then user_suffix="_u${user_count}"; fi
        local out_file="${out_dir}/amortized_irt_${emb}_${model}${suffix}${user_suffix}${n_suffix}${notau_suffix}${j_suffix}${baseline_suffix}.csv"

        cmd="$cmd --output $out_file"
    fi
    if [[ -n "$pair_efficiency_output" ]]; then
        mkdir -p "$(dirname "$pair_efficiency_output")"
        cmd="$cmd --pair-efficiency-output $pair_efficiency_output"
    fi
    if [[ -n "$neighbor_support_output" ]]; then
        mkdir -p "$(dirname "$neighbor_support_output")"
        cmd="$cmd --neighbor-support-output $neighbor_support_output"
    fi
    if [[ -n "$support_thinning_output" ]]; then
        mkdir -p "$(dirname "$support_thinning_output")"
        cmd="$cmd --support-thinning-output $support_thinning_output"
    fi
    if [[ -n "$outlier_robustness_output" ]]; then
        mkdir -p "$(dirname "$outlier_robustness_output")"
        cmd="$cmd --outlier-robustness-output $outlier_robustness_output"
    fi
    if [[ "$train_retention" != "1.0" ]]; then
        cmd="$cmd --train-retention $train_retention"
    fi
    if [[ "$cross_revision_post_binary" == "true" ]]; then
        cmd="$cmd --cross-revision-post-binary"
        cmd="$cmd --cross-revision-araf-mode ${cross_revision_araf_mode}"
    fi
    if [[ -n "$user_count" ]]; then
        cmd="$cmd --user-count ${user_count}"
    fi
    if [[ "$PARALLEL" -gt 1 ]]; then
        cmd="$cmd --parallel $PARALLEL"
    fi

    echo " -> Running: $emb (N=$n, pre=$pre, match=${pre_match}, eff_pre=${pre:-none}, users=${user_count:-all}, $model, base_emb=${baseline_emb}, k=${knn_k}, profile=${baseline_profile}, cross_mode=${cross_revision_araf_mode}) Taus=[$taus_csv] Seeds=[$seeds_csv]"
    eval "$cmd"
}

seed_support_thinning_araf_file() {
    local retention_dir=$1
    local araf_dir=$2
    local araf_emb=$3
    local target_file="${araf_dir}/amortized_irt_${araf_emb}_bernoulli_pre_max_n_max.csv"

    mkdir -p "${araf_dir}"

    local best_source=""
    local best_rows=-1
    while IFS= read -r candidate; do
        [[ -z "${candidate}" ]] && continue
        local rows
        rows=$(wc -l < "${candidate}" 2>/dev/null || echo 0)
        if [[ "${rows}" -gt "${best_rows}" ]]; then
            best_rows="${rows}"
            best_source="${candidate}"
        fi
    done < <(find "${retention_dir}" -type f -name "amortized_irt_${araf_emb}_bernoulli_pre_max_n_max*.csv" ! -path "${araf_dir}/*" 2>/dev/null)

    if [[ -f "${target_file}" ]]; then
        local target_rows
        target_rows=$(wc -l < "${target_file}" 2>/dev/null || echo 0)
        if [[ "${target_rows}" -ge "${best_rows}" ]]; then
            return
        fi
    fi

    if [[ -n "${best_source}" && "${best_rows}" -gt 0 ]]; then
        cp "${best_source}" "${target_file}"
        echo " -> Seeded ${target_file##*/} from ${best_source#${retention_dir}/} (${best_rows} lines)"
    fi
}

run_tau_sweep() {
    local emb=$1
    local n=$2
    local model=$3
    local base_tau=$4
    local pre=${5:-false}
    local j_pct=${6:-1.0}

    local taus="$base_tau"
    if $FULL_SWEEP; then
        taus="$SHARED_TAUS"
    fi

    run_exp "$emb" "$n" "$model" "$taus" "$pre" "$SEEDS" "${RESULT_DIR}" false false "$j_pct"
}

run_support_thinning_study() {
    local saved_result_dir="${RESULT_DIR}"
    local saved_baseline_csv="${BASELINE_CSV}"
    local saved_mirt_sweep_csv="${MIRT_SWEEP_CSV}"

    mkdir -p "${THIN_RESULT_DIR}"

    echo " -> Running post-matrix support-thinning study on the revised oracle..."
    echo " -> Thinning only the observed train support after the standard 90/10 post item split."
    echo " -> Sweeping both Bernoulli and Beta variants for ARAF and kNN."
    echo " -> Priming shared Rasch and MIRT baselines (with MIRT dimension sweep) per thinning level."
    echo " -> Also preserving the legacy pre-max thinning branch for Binary Pre comparisons."
    echo " -> Using full tau sweep for ARAF across embeddings: ${THIN_ARAF_EMBEDDINGS[*]}"
    echo " -> Reusing cached outputs when seed/tau rows already exist."
    for retention in "${THIN_RETENTIONS[@]}"; do
        local ret_label
        ret_label=$(printf "retain_%0.3f" "$retention")

        local araf_dir="${THIN_RESULT_DIR}/${ret_label}/araf_sweeps"
        local shared_baseline_dir="${THIN_RESULT_DIR}/${ret_label}/shared_baselines"
        RESULT_DIR="${araf_dir}"
        BASELINE_CSV="${shared_baseline_dir}/baseline_metrics.csv"
        MIRT_SWEEP_CSV="${shared_baseline_dir}/mirt_sweep.csv"
        mkdir -p "${araf_dir}" "${shared_baseline_dir}"

        local pre_revision="false"
        local j_percentage="1.0"
        for model_type in bernoulli beta; do
            RESULT_DIR="${araf_dir}"
            BASELINE_CSV="${shared_baseline_dir}/baseline_metrics.csv"
            MIRT_SWEEP_CSV="${shared_baseline_dir}/mirt_sweep.csv"
            run_baseline max "${model_type}" "${pre_revision}" "${SEEDS}" "${j_percentage}" "raw" "10" "full" "${retention}" "false" "32"
            for araf_emb in "${THIN_ARAF_EMBEDDINGS[@]}"; do
                local taus="$SHARED_TAUS"
                run_exp "${araf_emb}" max "${model_type}" "${taus}" "${pre_revision}" "${SEEDS}" "${araf_dir}" false false "${j_percentage}" "" "" "" "" "${retention}" "${araf_emb}" "10" "knn_only" "false" "false" "transfer" "32"
            done
            for knn_emb in "${THIN_KNN_EMBEDDINGS[@]}"; do
                for knn_k in "${THIN_K_VALUES[@]}"; do
                    local combo_dir="${THIN_RESULT_DIR}/${ret_label}/knn_${model_type}_${knn_emb}_k${knn_k}"
                    local baselines_dir="${combo_dir}/baselines"
                    RESULT_DIR="${combo_dir}"
                    BASELINE_CSV="${baselines_dir}/baseline_metrics.csv"
                    MIRT_SWEEP_CSV="${baselines_dir}/mirt_sweep.csv"
                    mkdir -p "${combo_dir}"
                    run_baseline max "${model_type}" "${pre_revision}" "${SEEDS}" "${j_percentage}" "${knn_emb}" "${knn_k}" "knn_only" "${retention}" "false" "32"
                done
            done
        done

        local legacy_pre_revision="max"
        local legacy_model_type="beta"
        RESULT_DIR="${araf_dir}"
        BASELINE_CSV="${shared_baseline_dir}/baseline_metrics.csv"
        MIRT_SWEEP_CSV="${shared_baseline_dir}/mirt_sweep.csv"
        run_baseline max "${legacy_model_type}" "${legacy_pre_revision}" "${SEEDS}" "1.0" "raw" "10" "full" "${retention}"
        for araf_emb in "${THIN_ARAF_EMBEDDINGS[@]}"; do
            local taus="$SHARED_TAUS"
            run_exp "${araf_emb}" max "${legacy_model_type}" "${taus}" "${legacy_pre_revision}" "${SEEDS}" "${araf_dir}" false false "1.0" "" "" "" "" "${retention}" "raw" "10" "knn_only" "false"
        done
        for knn_emb in "${THIN_KNN_EMBEDDINGS[@]}"; do
            for knn_k in "${THIN_K_VALUES[@]}"; do
                local legacy_combo_dir="${THIN_RESULT_DIR}/${ret_label}/knn_${knn_emb}_k${knn_k}"
                local legacy_baselines_dir="${legacy_combo_dir}/baselines"
                RESULT_DIR="${legacy_combo_dir}"
                BASELINE_CSV="${legacy_baselines_dir}/baseline_metrics.csv"
                MIRT_SWEEP_CSV="${legacy_baselines_dir}/mirt_sweep.csv"
                mkdir -p "${legacy_combo_dir}"
                run_baseline max "${legacy_model_type}" "${legacy_pre_revision}" "${SEEDS}" "1.0" "${knn_emb}" "${knn_k}" "knn_only" "${retention}"
            done
        done
    done

    echo " -> Rebuilding support-thinning summary from completed sweep files..."
    local rebuild_cmd="${SUPPORT_THINNING_REBUILD_CMD}"
    if $QUIET; then
        rebuild_cmd="$rebuild_cmd"
    fi
    eval "$rebuild_cmd"

    RESULT_DIR="${saved_result_dir}"
    BASELINE_CSV="${saved_baseline_csv}"
    MIRT_SWEEP_CSV="${saved_mirt_sweep_csv}"
}

run_sample_size_study() {
    local saved_result_dir="${RESULT_DIR}"
    local saved_baseline_csv="${BASELINE_CSV}"
    local saved_mirt_sweep_csv="${MIRT_SWEEP_CSV}"

    mkdir -p "${SAMPLE_SIZE_RESULT_DIR}"

    echo " -> Running dedicated post-revision sample-size study into ${SAMPLE_SIZE_RESULT_DIR} ..."
    echo " -> Varying post user count N on the fixed revised oracle, then varying J at fixed N=32."

    local model_type
    local emb
    for user_level in "${SAMPLE_USER_LEVELS[@]}"; do
        RESULT_DIR="${SAMPLE_SIZE_RESULT_DIR}/users_${user_level}"
        BASELINE_CSV="${RESULT_DIR}/baselines/baseline_metrics.csv"
        MIRT_SWEEP_CSV="${RESULT_DIR}/baselines/mirt_sweep.csv"
        mkdir -p "${RESULT_DIR}" "$(dirname "${BASELINE_CSV}")"
        local run_user_count=""
        if [[ "${user_level}" != "max" ]]; then
            run_user_count="${user_level}"
        fi
        for model_type in bernoulli beta; do
            for emb in sae pca raw; do
                local tau="0.0159"
                if [[ "${emb}" == "pca" ]]; then tau="0.0155"; fi
                if [[ "${emb}" == "raw" ]]; then tau="0.0151"; fi
                if [[ "${model_type}" == "beta" ]]; then
                    tau="0.0535"
                    if [[ "${emb}" == "pca" ]]; then tau="0.054"; fi
                    if [[ "${emb}" == "raw" ]]; then tau="0.029"; fi
                fi
                run_exp "${emb}" max "${model_type}" "${tau}" false "${SEEDS}" "${RESULT_DIR}" false false "1.0" "" "" "" "" "1.0" "${emb}" "10" "full" "true" "false" "transfer" "${run_user_count}"
            done
        done
    done

    RESULT_DIR="${SAMPLE_SIZE_RESULT_DIR}/items"
    BASELINE_CSV="${RESULT_DIR}/baselines/baseline_metrics.csv"
    MIRT_SWEEP_CSV="${RESULT_DIR}/baselines/mirt_sweep.csv"
    mkdir -p "${RESULT_DIR}" "$(dirname "${BASELINE_CSV}")"
    for j in "${SAMPLE_J_LEVELS[@]}" "1.0"; do
        for model_type in bernoulli beta; do
            for emb in sae pca raw; do
                local tau="0.0159"
                if [[ "${emb}" == "pca" ]]; then tau="0.0155"; fi
                if [[ "${emb}" == "raw" ]]; then tau="0.0151"; fi
                if [[ "${model_type}" == "beta" ]]; then
                    tau="0.0535"
                    if [[ "${emb}" == "pca" ]]; then tau="0.054"; fi
                    if [[ "${emb}" == "raw" ]]; then tau="0.029"; fi
                fi
                run_exp "${emb}" max "${model_type}" "${tau}" false "${SEEDS}" "${RESULT_DIR}" false false "${j}" "" "" "" "" "1.0" "${emb}" "10" "full" "true" "false" "transfer" "32"
            done
        done
    done

    RESULT_DIR="${saved_result_dir}"
    BASELINE_CSV="${saved_baseline_csv}"
    MIRT_SWEEP_CSV="${saved_mirt_sweep_csv}"
}

# ── Execution ───────────────────────────────────────────────────────────────
if ! $ONLY_PLOT && $RUN_MAIN_EXPERIMENTS; then
    echo "[MODE] Running Experiments..."
    # Prime baseline cache for canonical post-revision setups.
    echo " -> Priming baseline cache (Post-32 Bernoulli, Post-max Beta)..."
    run_baseline 32 bernoulli false "$SEEDS" 1.0
    run_baseline max beta false "$SEEDS" 1.0

    # [SCALING LAW]: Item Scaling Study (N=32)
    if $FULL_SWEEP; then
        echo " -> Starting Item Scaling Law Study (N=32, Full Tau Sweep)..."
        for j in 0.1 0.3 0.5 0.7 0.9; do
            run_tau_sweep sae max beta 0.16 32 $j
            run_tau_sweep pca max beta 0.054 32 $j
            run_tau_sweep raw max beta 0.029 32 $j

            run_tau_sweep sae 1 bernoulli 0.0159 32 $j
            run_tau_sweep pca 1 bernoulli 0.0155 32 $j
            run_tau_sweep raw 1 bernoulli 0.0151 32 $j
        done
    fi

    # 0. Primary Model Exports (Required for Interpretability Plots)
    echo " -> Exporting primary SAE weights..."
    run_exp sae max beta 0.16 max "$SEEDS" "${RESULT_DIR}" false true
    run_exp sae max beta 0.0535 false "$SEEDS" "${RESULT_DIR}" false true

    # 1. PCA Embeddings
    run_tau_sweep pca max beta 0.054 false
    run_tau_sweep pca 1 bernoulli 0.0155 false

    # 2. SAE Embeddings
    run_tau_sweep sae max beta 0.0535 false
    run_tau_sweep sae 1 bernoulli 0.0159 false

    # 3. RAW Embeddings
    run_tau_sweep raw max beta 0.029 false
    run_tau_sweep raw 1 bernoulli 0.0151 false

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

        # PCA/RAW
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

        # Prime baseline cache for N-sweep pre-revision settings.
        echo " -> Priming Baseline Cache for N-Sweep pre-revision settings..."
        for n in 4 8 16 32 64 max; do
            run_baseline 1 bernoulli $n "$SEEDS" 1.0
            run_baseline max beta $n "$SEEDS" 1.0
        done

        # 5. Ablation Studies
        run_exp sae max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp pca max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp raw max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true

        run_exp ones max beta "$SHARED_TAUS" false "$SEEDS" "${RESULT_DIR}" false
        run_exp ones max beta "1.0" false "$SEEDS" "${RESULT_DIR}" true

        run_exp sae 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp pca 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true
        run_exp raw 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true

        run_exp ones 1 bernoulli "$SHARED_TAUS" false "$SEEDS" "${RESULT_DIR}" false
        run_exp ones 1 bernoulli "1.0" false "$SEEDS" "${RESULT_DIR}" true

        run_exp sae max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true
        run_exp pca max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true
        run_exp raw max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true

        run_exp ones max beta "$SHARED_TAUS" max "$SEEDS" "${RESULT_DIR}" false
        run_exp ones max beta "1.0" max "$SEEDS" "${RESULT_DIR}" true

        run_exp sae 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true
        run_exp pca 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true
        run_exp raw 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true

        run_exp ones 1 bernoulli "$SHARED_TAUS" 32 "$SEEDS" "${RESULT_DIR}" false
        run_exp ones 1 bernoulli "1.0" 32 "$SEEDS" "${RESULT_DIR}" true
    fi

    if $RUN_SUPPORT_THINNING_STUDY; then
        run_support_thinning_study
    fi
    if $RUN_SAMPLE_SIZE_STUDY; then
        run_sample_size_study
    fi
fi

if ! $ONLY_PLOT && ! $RUN_MAIN_EXPERIMENTS; then
    echo "[MODE] Running requested study only..."
    if $SUPPORT_THINNING_STUDY; then
        run_support_thinning_study
    fi
    if $SAMPLE_SIZE_STUDY; then
        run_sample_size_study
    fi
fi

# ── Generate plots ───────────────────────────────────────────────────────────
echo ""
echo "=========================================================="
echo "  GENERATING PLOTS"
echo "=========================================================="
cd "${REPO_ROOT}"
if $SUPPORT_THINNING_STUDY || $SAMPLE_SIZE_STUDY; then
    if $SUPPORT_THINNING_STUDY; then
        PYTHONPATH=. python3 -m model.plotting.main --support-thinning-study
    fi
    if $SAMPLE_SIZE_STUDY; then
        PYTHONPATH=. python3 -m model.plotting.main --sample-size
    fi
else
    PYTHONPATH=. python3 -m model.plotting.main --all
fi

echo ""
echo "=========================================================="
echo "  REPRODUCTION COMPLETE"
echo "=========================================================="
echo "Plots saved in paper/figures/"
echo "Main CSV results in model/result/main/"
echo "Study CSV results in model/result/*_study/"
