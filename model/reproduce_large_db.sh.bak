#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/result/measurement_db_raw}"
SOURCE="${SOURCE:-source}"
SOURCE_DIR="${SOURCE_DIR:-${RESULT_ROOT}/data_cache/measurement-db-source}"
BUILD_SOURCE=false
ALL_READY=true
DATASETS=""
PARALLEL="${PARALLEL:-100}"
ARAF_PARALLEL="${ARAF_PARALLEL:-4}"
QUIET=true
FORCE=false
MODEL_TYPE="bernoulli"
SEED="42"
SEEDS_COUNT=""
N_SAMPLES="1"
# Match main pipeline ARAF sweep by default (see model/reproduce.sh:SHARED_TAUS).
LAMBDA_TAU="0.002,0.004,0.005,0.006,0.008,0.010,0.012,0.014,0.015,0.0151,0.0155,0.0159,0.016,0.018,0.020,0.022,0.024,0.025,0.026,0.028,0.029,0.030,0.032,0.034,0.035,0.036,0.038,0.040,0.042,0.044,0.045,0.046,0.048,0.050,0.052,0.0535,0.054,0.055,0.056,0.058,0.060,0.062,0.064,0.065,0.066,0.068,0.070,0.072,0.074,0.075,0.076,0.078,0.080,0.082,0.084,0.085,0.086,0.088,0.090,0.092,0.094,0.095,0.096,0.098,0.100,0.105,0.110,0.115,0.120,0.125,0.130,0.135,0.140,0.145,0.150,0.155,0.160,0.165,0.170,0.175,0.180,0.185,0.190,0.195,0.200,0.210,0.220,0.230,0.250,0.30,0.40,0.50,0.75,1.0,1.5,2.0,3.0,5.0,10.0,20.0,30.0,50.0,75.0,100.0,200.0,500.0,1000.0"
ARAF_LATENT_DIMS="30"
ARAF_DROPOUTS="0.5"
EPOCHS="${EPOCHS:-1000}"
TRAIN_RETENTION="1.0"
TEST_SIZE="0.1"
KNN_K_GRID="${KNN_K_GRID:-5,10,20,50}"
KNN_K="10"
MIN_SUBJECTS="4"
MIN_ITEMS="50"
RESPONSE_POLICY="bounded01"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
MAX_CHARS="${MAX_CHARS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
MEASUREMENT_DB_GIT="${MEASUREMENT_DB_GIT:-https://github.com/aims-foundations/measurement-db.git}"

usage() {
  cat <<'EOF'
Usage:
  bash model/reproduce_large_db.sh [options]

RAW-only measurement-db run: prepare corpus, embed items, run kNN baseline,
run RAW ARAF, and write aggregation-ready summaries.

Options:
  --source hf|source        Data source. Default: source.
  --build-source           Clone/update measurement-db and run reproduce.py --no-upload.
  --no-build-source        Use existing source dir without rebuilding.
  --source-dir PATH        Source checkout/output dir for --source source.
  --all-ready              Use all available/source-built datasets. Default.
  --datasets a,b,c         Use explicit dataset list.
  --parallel N             Worker count for model runner. Default: 100.
  --araf-parallel N        Worker count for ARAF sweep. Default: 8 to avoid GPU OOM.
  --quiet                  Suppress model output to terminal. Default.
  --verbose                Disable quiet flag for Python commands.
  --epochs N               ARAF epochs. Default: 1000.
  --seed N[,M]             Seed list. Default: 42.
  --seeds K                Use first K seeds: 0..K-1 (overrides --seed). Example: --seeds 3 -> 0,1,2.
  --lambda-tau X[,Y]       Tau list. Default: same shared ARAF sweep as model/reproduce.sh.
  --araf-latent-dims K[,..]   ARAF latent dimension K sweep. Default: 30.
  --araf-dropouts D[,..]     ARAF dropout sweep. Default: 0.5.
  --result-root PATH       Stable output root. Default: model/result/measurement_db_raw.
  --force                  Rebuild corpus/embeddings even if config matches.
  --help                   Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --build-source) BUILD_SOURCE=true; shift ;;
    --no-build-source) BUILD_SOURCE=false; shift ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --all-ready) ALL_READY=true; DATASETS=""; shift ;;
    --datasets) DATASETS="$2"; ALL_READY=false; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --araf-parallel) ARAF_PARALLEL="$2"; shift 2 ;;
    --quiet) QUIET=true; shift ;;
    --verbose) QUIET=false; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --seeds) SEEDS_COUNT="$2"; shift 2 ;;
    --lambda-tau) LAMBDA_TAU="$2"; shift 2 ;;
    --araf-latent-dims) ARAF_LATENT_DIMS="$2"; shift 2 ;;
    --araf-dropouts) ARAF_DROPOUTS="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "${SEEDS_COUNT}" ]]; then
  if ! [[ "${SEEDS_COUNT}" =~ ^[0-9]+$ ]] || [[ "${SEEDS_COUNT}" -lt 1 ]]; then
    echo "ERROR: --seeds must be positive int" >&2
    exit 2
  fi
  SEED="$(python - <<PY
k=int(${SEEDS_COUNT@Q})
print(",".join(str(i) for i in range(k)))
PY
)"
fi

if [[ "${SOURCE}" != "hf" && "${SOURCE}" != "source" ]]; then
  echo "ERROR: --source must be hf or source" >&2
  exit 2
fi

if [[ "${SOURCE}" == "source" && ! -d "${SOURCE_DIR}" && "${BUILD_SOURCE}" != true ]]; then
  echo "[INFO] Source dir missing; enabling --build-source for ${SOURCE_DIR}"
  BUILD_SOURCE=true
fi

eval "$(conda shell.bash hook)"
conda activate hal

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p \
  "${RESULT_ROOT}/data_cache" \
  "${RESULT_ROOT}/embeddings" \
  "${RESULT_ROOT}/baselines" \
  "${RESULT_ROOT}/araf" \
  "${RESULT_ROOT}/summaries" \
  "${RESULT_ROOT}/logs"

echo "[ENV] repo=${REPO_ROOT}"
echo "[ENV] result_root=${RESULT_ROOT}"
echo "[ENV] source=${SOURCE}"
echo "[ENV] parallel=${PARALLEL}"
echo "[ENV] araf_parallel=${ARAF_PARALLEL}"
echo "[ENV] quiet=${QUIET}"
echo "[ENV] python=$(which python)"

QUIET_FLAG=()
if [[ "${QUIET}" == true ]]; then
  QUIET_FLAG=(--quiet)
fi
FORCE_FLAG=()
if [[ "${FORCE}" == true ]]; then
  FORCE_FLAG=(--force)
fi
DATASET_FLAGS=()
if [[ "${ALL_READY}" == true ]]; then
  DATASET_FLAGS=(--all-ready)
else
  DATASET_FLAGS=(--datasets "${DATASETS}")
fi
BUILD_FLAGS=()
if [[ "${BUILD_SOURCE}" == true ]]; then
  BUILD_FLAGS=(--build-source)
fi

PREP_ENV="${RESULT_ROOT}/summaries/latest_paths.env"
PREP_LOG="${RESULT_ROOT}/logs/00_prepare.log"
echo "[RUN] prepare corpus + RAW embeddings -> ${PREP_LOG}"
if ! python -m model.measurement_db_raw prepare \
  --result-root "${RESULT_ROOT}" \
  --source "${SOURCE}" \
  --source-dir "${SOURCE_DIR}" \
  --measurement-db-git "${MEASUREMENT_DB_GIT}" \
  "${BUILD_FLAGS[@]}" \
  "${DATASET_FLAGS[@]}" \
  --min-subjects "${MIN_SUBJECTS}" \
  --min-items "${MIN_ITEMS}" \
  --response-policy "${RESPONSE_POLICY}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --max-chars "${MAX_CHARS}" \
  --batch-size "${BATCH_SIZE}" \
  --chunk-size "${CHUNK_SIZE}" \
  "${QUIET_FLAG[@]}" \
  "${FORCE_FLAG[@]}" \
  > "${PREP_ENV}" 2> "${PREP_LOG}"; then
  touch "${PREP_LOG}.failed"
  echo "[FAIL] prepare failed; see ${PREP_LOG}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${PREP_ENV}"

echo "[OK] corpus=${CORPUS_SLUG}"
echo "[OK] embeddings=${EMBEDDING_SLUG}"
echo "[OK] paths=${PREP_ENV}"

SPLIT_ENV="${RESULT_ROOT}/summaries/latest_split_paths.env"
SPLIT_LOG="${RESULT_ROOT}/logs/05_split_inventory.log"
echo "[RUN] split inventory -> ${SPLIT_LOG}"
if ! python -m model.measurement_db_raw split-inventory \
  --result-root "${RESULT_ROOT}" \
  --corpus-path "${CORPUS_PATH}" \
  --item-content-path "${ITEM_CONTENT_PATH}" \
  --corpus-slug "${CORPUS_SLUG}" \
  --seed "${SEED%%,*}" \
  --test-size "${TEST_SIZE}" \
  > "${SPLIT_ENV}" 2> "${SPLIT_LOG}"; then
  touch "${SPLIT_LOG}.failed"
  echo "[FAIL] split inventory failed; see ${SPLIT_LOG}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${SPLIT_ENV}"

EMBED_MODEL_TAG="${EMBEDDING_MODEL##*/}"
# Keep filenames under FS component limits. Full exact config saved in sidecar JSON.
COMMON_SLUG="corpus-${CORPUS_SLUG}_emb-raw-${EMBED_MODEL_TAG}-mc${MAX_CHARS}_model-${MODEL_TYPE}_seed-${SEED}_test-${TEST_SIZE}_ret-${TRAIN_RETENTION}_n-${N_SAMPLES}"
COMMON_SLUG="$(python - <<PY
from model.measurement_db_raw import slugify
print(slugify(${COMMON_SLUG@Q}))
PY
)"
KNN_SLUG="${COMMON_SLUG}_k-${KNN_K_GRID//,/-}"
TAU_COUNT="$(python - <<PY2
print(len([x for x in ${LAMBDA_TAU@Q}.replace(',', ' ').split() if x.strip()]))
PY2
)"
KNN_SLUG="${COMMON_SLUG}_k-${KNN_K_GRID//,/-}"
KNN_SLUG="$(python - <<PY
from model.measurement_db_raw import slugify
print(slugify(${KNN_SLUG@Q}))
PY
)"

BASELINE_CSV="${RESULT_ROOT}/baselines/baseline_metrics_${KNN_SLUG}.csv"
MIRT_CSV="${RESULT_ROOT}/baselines/mirt_sweep_${KNN_SLUG}.csv"
KNN_LOG="${RESULT_ROOT}/logs/10_knn_${KNN_SLUG}.log"
SUMMARY_LOG="${RESULT_ROOT}/logs/30_summary.log"
KNN_CONFIG_TMP="${BASELINE_CSV}.config.json.tmp"

cat > "${KNN_CONFIG_TMP}" <<JSON
{
  "kind": "measurement_db_raw_knn",
  "data_source": "${SOURCE}",
  "dataset_selector": "$([[ "${ALL_READY}" == true ]] && echo all-ready || echo "${DATASETS}")",
  "corpus_slug": "${CORPUS_SLUG}",
  "corpus_config_path": "${CORPUS_CONFIG_PATH}",
  "embedding_slug": "${EMBEDDING_SLUG}",
  "embedding_config_path": "${EMBEDDING_CONFIG_PATH}",
  "model_type": "${MODEL_TYPE}",
  "seed": "${SEED}",
  "test_size": ${TEST_SIZE},
  "train_retention": ${TRAIN_RETENTION},
  "n_samples": ${N_SAMPLES},
  "knn_k_grid": "${KNN_K_GRID}",
  "train_item_count": ${TRAIN_ITEM_COUNT},
  "test_item_count": ${TEST_ITEM_COUNT},
  "train_observed_count": ${TRAIN_OBSERVED_COUNT},
  "test_observed_count": ${TEST_OBSERVED_COUNT},
  "split_inventory_path": "${SPLIT_INVENTORY_PATH}",
  "log_path": "${KNN_LOG}"
}
JSON

echo "[RUN] RAW kNN baseline -> ${KNN_LOG}"
if [[ -f "${BASELINE_CSV}.config.json" ]] &&
   cmp -s "${KNN_CONFIG_TMP}" "${BASELINE_CSV}.config.json" &&
   python -m model.measurement_db_raw check-result --kind knn --path "${BASELINE_CSV}" >/dev/null 2>&1; then
  echo "[SKIP] RAW kNN complete: ${BASELINE_CSV}"
  rm -f "${KNN_CONFIG_TMP}"
  python - <<PY >/dev/null 2>&1
import model.baseline_cache as bc
bc.write_baseline_manifest(${BASELINE_CSV@Q})
bc.write_mirt_sweep_manifest(${MIRT_CSV@Q})
PY
else
  mv "${KNN_CONFIG_TMP}" "${BASELINE_CSV}.config.json"
  if ! PYTHONPATH="${REPO_ROOT}" python -m model.amortized_irt \
    --data-source measurement_db_raw \
    --mdb-corpus-path "${CORPUS_PATH}" \
    --mdb-raw-embeddings-path "${EMBEDDING_PATH}" \
    --embedding-type raw \
    --baseline-embedding-type raw \
    --model-type "${MODEL_TYPE}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}" \
    --baseline-only \
    --baseline-profile knn_only \
    --baseline-output "${BASELINE_CSV}" \
    --mirt-sweep-output "${MIRT_CSV}" \
    --train-retention "${TRAIN_RETENTION}" \
    --knn-k "${KNN_K}" \
    --knn-k-grid "${KNN_K_GRID}" \
    --parallel "${PARALLEL}" \
    "${QUIET_FLAG[@]}" \
    > "${KNN_LOG}" 2>&1; then
    touch "${BASELINE_CSV}.failed"
    touch "${KNN_LOG}.failed"
    echo "[FAIL] RAW kNN failed; see ${KNN_LOG}" >&2
    exit 1
  fi
  python - <<PY >/dev/null 2>&1
import model.baseline_cache as bc
bc.write_baseline_manifest(${BASELINE_CSV@Q})
bc.write_mirt_sweep_manifest(${MIRT_CSV@Q})
PY
fi

IFS=',' read -r -a ARAF_K_VALUES <<< "${ARAF_LATENT_DIMS}"
IFS=',' read -r -a ARAF_DROPOUT_VALUES <<< "${ARAF_DROPOUTS}"
ARAF_CSVS=()
for ARAF_LATENT_DIM in "${ARAF_K_VALUES[@]}"; do
  for ARAF_DROPOUT in "${ARAF_DROPOUT_VALUES[@]}"; do
    ARAF_LATENT_DIM="$(echo "${ARAF_LATENT_DIM}" | xargs)"
    ARAF_DROPOUT="$(echo "${ARAF_DROPOUT}" | xargs)"
    [[ -n "${ARAF_LATENT_DIM}" && -n "${ARAF_DROPOUT}" ]] || continue
    ARAF_SLUG="${COMMON_SLUG}_araf-k-${ARAF_LATENT_DIM}_dropout-${ARAF_DROPOUT}_tau-sweep-${TAU_COUNT}_epochs-${EPOCHS}"
    ARAF_SLUG="$(python - <<PY
from model.measurement_db_raw import slugify
print(slugify(${ARAF_SLUG@Q}))
PY
)"
    ARAF_CSV="${RESULT_ROOT}/araf/araf_raw_${ARAF_SLUG}.csv"
    ARAF_LOG="${RESULT_ROOT}/logs/20_araf_${ARAF_SLUG}.log"
    ARAF_CONFIG_TMP="${ARAF_CSV}.config.json.tmp"
    ARAF_CSVS+=("${ARAF_CSV}")

    cat > "${ARAF_CONFIG_TMP}" <<JSON
{
  "kind": "measurement_db_raw_araf",
  "data_source": "${SOURCE}",
  "dataset_selector": "$([[ "${ALL_READY}" == true ]] && echo all-ready || echo "${DATASETS}")",
  "corpus_slug": "${CORPUS_SLUG}",
  "corpus_config_path": "${CORPUS_CONFIG_PATH}",
  "embedding_slug": "${EMBEDDING_SLUG}",
  "embedding_config_path": "${EMBEDDING_CONFIG_PATH}",
  "model_type": "${MODEL_TYPE}",
  "seed": "${SEED}",
  "test_size": ${TEST_SIZE},
  "train_retention": ${TRAIN_RETENTION},
  "n_samples": ${N_SAMPLES},
  "lambda_tau": "${LAMBDA_TAU}",
  "epochs": ${EPOCHS},
  "araf_latent_dim": ${ARAF_LATENT_DIM},
  "araf_dropout": ${ARAF_DROPOUT},
  "train_item_count": ${TRAIN_ITEM_COUNT},
  "test_item_count": ${TEST_ITEM_COUNT},
  "train_observed_count": ${TRAIN_OBSERVED_COUNT},
  "test_observed_count": ${TEST_OBSERVED_COUNT},
  "split_inventory_path": "${SPLIT_INVENTORY_PATH}",
  "log_path": "${ARAF_LOG}"
}
JSON

    echo "[RUN] RAW ARAF k=${ARAF_LATENT_DIM} dropout=${ARAF_DROPOUT} -> ${ARAF_LOG}"
    if [[ -f "${ARAF_CSV}.config.json" ]] &&
       cmp -s "${ARAF_CONFIG_TMP}" "${ARAF_CSV}.config.json" &&
       python -m model.measurement_db_raw check-result --kind araf --path "${ARAF_CSV}" >/dev/null 2>&1; then
      echo "[SKIP] RAW ARAF complete: ${ARAF_CSV}"
      rm -f "${ARAF_CONFIG_TMP}"
    else
      mv "${ARAF_CONFIG_TMP}" "${ARAF_CSV}.config.json"
      if ! PYTHONPATH="${REPO_ROOT}" python -m model.amortized_irt \
        --data-source measurement_db_raw \
        --mdb-corpus-path "${CORPUS_PATH}" \
        --mdb-raw-embeddings-path "${EMBEDDING_PATH}" \
        --embedding-type raw \
        --baseline-embedding-type raw \
        --model-type "${MODEL_TYPE}" \
        --n-samples "${N_SAMPLES}" \
        --seed "${SEED}" \
        --lambda-tau "${LAMBDA_TAU}" \
        --epochs "${EPOCHS}" \
        --output "${ARAF_CSV}" \
        --train-retention "${TRAIN_RETENTION}" \
        --araf-latent-dim "${ARAF_LATENT_DIM}" \
        --araf-dropout "${ARAF_DROPOUT}" \
        --parallel "${ARAF_PARALLEL}" \
        "${QUIET_FLAG[@]}" \
        > "${ARAF_LOG}" 2>&1; then
        touch "${ARAF_CSV}.failed"
        touch "${ARAF_LOG}.failed"
        echo "[FAIL] RAW ARAF failed; see ${ARAF_LOG}" >&2
        exit 1
      fi
    fi
  done
done
ARAF_CSV="${ARAF_CSVS[*]}"

echo "[RUN] aggregate summaries -> ${SUMMARY_LOG}"
if ! python -m model.measurement_db_raw aggregate \
  --result-root "${RESULT_ROOT}" \
  > "${RESULT_ROOT}/summaries/latest_summary_paths.env" 2> "${SUMMARY_LOG}"; then
  touch "${SUMMARY_LOG}.failed"
  echo "[FAIL] summary aggregation failed; see ${SUMMARY_LOG}" >&2
  exit 1
fi

echo "[DONE] RAW measurement-db pipeline prepared and runnable."
echo "[DONE] result_root=${RESULT_ROOT}"
echo "[DONE] baseline_csv=${BASELINE_CSV}"
echo "[DONE] araf_csv=${ARAF_CSV}"
echo "[DONE] summaries=${RESULT_ROOT}/summaries"
