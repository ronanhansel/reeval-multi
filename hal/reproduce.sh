#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Run all experiments from scratch and generate plots
###############################################################################

eval "$(conda shell.bash hook)"
conda activate reeval
echo "[ENV] conda env: reeval  (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/result"

echo "=========================================================="
echo "  REPRODUCE — Fresh Run"
echo "=========================================================="
echo ""
echo "  Working dir : ${SCRIPT_DIR}"
echo "  Output dir  : ${RESULT_DIR}"
echo ""

# ── Clean caches ──────────────────────────────────────────────────────────────
echo "[CLEAN] Removing all caches …"
rm -f  "${SCRIPT_DIR}/pca_aggregate_survey_cache.pkl"
rm -rf "${SCRIPT_DIR}/.checkpoints"
rm -rf "${SCRIPT_DIR}/checkpoints"
rm -f  "${SCRIPT_DIR}/feature_descriptions_sae.pkl"
rm -rf /tmp/_reproduce_sae_ckpt /tmp/_reproduce_sae_ckpt2
mkdir -p "${RESULT_DIR}"
echo "[CLEAN] Done."
echo ""

# ── Run scripts ──────────────────────────────────────────────────────────────
SCRIPTS=(
    "pca_aggregate_survey.py"
    "plotting.py"
    "sae_beta_irt.py"
    "plot_judge_iterations.py"
)

overall_ok=true

for script in "${SCRIPTS[@]}"; do
    echo "=========================================================="
    echo "  RUNNING: ${script}"
    echo "=========================================================="

    set +e
    (cd "${SCRIPT_DIR}" && python "${SCRIPT_DIR}/${script}" 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo ""
        echo "  [ERROR] ${script} exited with code ${rc}"
        overall_ok=false
    else
        echo ""
        echo "  [OK] ${script} completed successfully."
    fi
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=========================================================="
echo "  OUTPUT SUMMARY"
echo "=========================================================="
echo ""
echo "  Output directory: ${RESULT_DIR}"
echo ""

echo "  Plots (PDF):"
found=false
for f in "${RESULT_DIR}"/*.pdf; do
    [ -e "$f" ] || continue
    found=true
    echo "    $(ls -lh "$f" | awk '{print $5, $NF}')"
done
$found || echo "    (none)"

echo ""
if $overall_ok; then
    echo "All scripts completed successfully."
else
    echo "Some scripts had errors — check output above."
fi
