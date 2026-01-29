#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Run all experiments from scratch and generate plots
#
# Data is automatically downloaded from HuggingFace if not present locally.
###############################################################################

eval "$(conda shell.bash hook)"
conda activate reeval
echo "[ENV] conda env: reeval  (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/result"

# ── HuggingFace cache configuration ─────────────────────────────────────────
# Set HF_HOME to avoid permission issues with default cache location
export HF_HOME="${HF_HOME:-/lfs/skampere1/0/sttruong/.cache/huggingface}"
mkdir -p "${HF_HOME}"
echo "[ENV] HF_HOME: ${HF_HOME}"

echo "=========================================================="
echo "  REPRODUCE — Fresh Run"
echo "=========================================================="
echo ""
echo "  Working dir : ${SCRIPT_DIR}"
echo "  Output dir  : ${RESULT_DIR}"
echo ""

# ── Install LaTeX packages for tueplots rendering ────────────────────────────
echo "[LATEX] Checking/installing required LaTeX packages..."
if command -v tlmgr &> /dev/null; then
    # Install packages needed for tueplots ICML style (skip verification due to gpg issues)
    tlmgr --verify-repo=none install type1cm cm-super underscore 2>/dev/null || true
    echo "[LATEX] Done."
else
    echo "[LATEX] tlmgr not found. If LaTeX rendering fails, install: type1cm, cm-super, underscore"
fi
echo ""

# ── Clean previous results ───────────────────────────────────────────────────
echo "[CLEAN] Removing previous results …"
rm -rf "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"
echo "[CLEAN] Done."
echo ""

# ── Run experiments ──────────────────────────────────────────────────────────
# amortized_irt.py auto-downloads data from HuggingFace if not present

echo "=========================================================="
echo "  RUNNING: amortized_irt.py (PCA embeddings)"
echo "=========================================================="
python "${SCRIPT_DIR}/amortized_irt.py" --embedding-type pca
echo ""

echo "=========================================================="
echo "  RUNNING: amortized_irt.py (SAE embeddings)"
echo "=========================================================="
python "${SCRIPT_DIR}/amortized_irt.py" --embedding-type sae
echo ""

# ── Generate plots ───────────────────────────────────────────────────────────
echo "=========================================================="
echo "  RUNNING: plotting.py"
echo "=========================================================="
python "${SCRIPT_DIR}/plotting.py"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=========================================================="
echo "  OUTPUT SUMMARY"
echo "=========================================================="
echo ""
echo "  Output directory: ${RESULT_DIR}"
echo ""

echo "  Results (CSV):"
for f in "${RESULT_DIR}"/*.csv; do
    [ -e "$f" ] || continue
    echo "    $(ls -lh "$f" | awk '{print $5, $NF}')"
done

echo ""
echo "  Plots (PDF/PNG):"
for f in "${RESULT_DIR}"/*.pdf "${RESULT_DIR}"/*.png; do
    [ -e "$f" ] || continue
    echo "    $(ls -lh "$f" | awk '{print $5, $NF}')"
done

echo ""
echo "All scripts completed successfully."
