#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce.sh — Convert all hal/ notebooks to Python and run them from
#                scratch (no caches), saving every plot to ./result/
###############################################################################

# ── Activate conda env ────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate reeval
echo "[ENV] Using conda env: reeval  (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_DIR="${SCRIPT_DIR}"              # script now lives inside hal/
RESULT_DIR="${HAL_DIR}/result"
WORK_DIR="${HAL_DIR}"                # notebooks expect to run from hal/
CONVERTED_DIR="${RESULT_DIR}/_converted_scripts"

NOTEBOOKS=(
    "pca_aggregate_survey"
    "plotting"
    "sae_beta_irt"
)

echo "=========================================================="
echo "  REPRODUCE — Notebooks → Python → Fresh Run"
echo "=========================================================="
echo ""
echo "  hal dir      : ${HAL_DIR}"
echo "  Output dir   : ${RESULT_DIR}"
echo ""

# ── 1. Create output directory ────────────────────────────────────────────────
mkdir -p "${RESULT_DIR}"
mkdir -p "${CONVERTED_DIR}"

# ── 2. Purge every cache so results are computed from scratch ─────────────────
echo "[CLEAN] Removing all caches …"
rm -f  "${HAL_DIR}/pca_aggregate_survey_cache.pkl"
rm -rf "${HAL_DIR}/.checkpoints"
rm -rf "${HAL_DIR}/checkpoints"
rm -f  "${HAL_DIR}/feature_descriptions_sae.pkl"
echo "[CLEAN] Done."
echo ""

# ── 3. Convert notebooks → .py via nbconvert ─────────────────────────────────
echo "[CONVERT] Converting notebooks to Python scripts …"
for nb_name in "${NOTEBOOKS[@]}"; do
    nb_file="${HAL_DIR}/${nb_name}.ipynb"
    if [ -f "${nb_file}" ]; then
        echo "  ${nb_name}.ipynb → ${CONVERTED_DIR}/${nb_name}.py"
        python -m nbconvert --to script "${nb_file}" \
            --output-dir="${CONVERTED_DIR}" 2>&1 | sed 's/^/    /'
    else
        echo "  [WARN] ${nb_file} not found — skipping."
    fi
done
echo ""

# ── 4. Patch the generated scripts ───────────────────────────────────────────
echo "[PATCH] Patching generated Python scripts …"
for nb_name in "${NOTEBOOKS[@]}"; do
    py_file="${CONVERTED_DIR}/${nb_name}.py"
    [ -f "${py_file}" ] || continue
    echo "  Patching ${nb_name}.py …"

    python - "${py_file}" "${RESULT_DIR}" <<'PATCH_EOF'
import sys, re, os

py_file    = sys.argv[1]
result_dir = sys.argv[2]

with open(py_file, "r") as f:
    code = f.read()

# ── a) Force non-interactive Agg backend before any other matplotlib use ──
header = (
    "import matplotlib\n"
    'matplotlib.use("Agg")\n\n'
)
code = header + code

# ── b) Remove IPython magic / cell markers ────────────────────────────────
code = re.sub(r"# In\[.*?\]:\n?", "", code)
code = re.sub(r"^get_ipython\(\).*$", "", code, flags=re.MULTILINE)

# ── c) Replace plt.show() (no-op in batch mode) ──────────────────────────
code = code.replace("plt.show()", "# plt.show()  # disabled for batch mode")

# ── d) Redirect plot saves:  'plots/' → <result_dir>/ ────────────────────
code = code.replace("'plots/", f"'{result_dir}/")

# ── e) After every savefig, inject a print showing the saved path ─────────
#       Matches both single- and double-quoted paths.
code = re.sub(
    r"""(plt\.savefig\(\s*['"])(.*?)(['"].*?\))""",
    lambda m: f'{m.group(0)}\nprint("[OUTPUT] Saved plot: {m.group(2)}")',
    code,
)

# ── f) Disable pickle-based cache in pca_aggregate_survey ─────────────────
code = code.replace(
    "cache = load_cache()",
    "cache = {}  # [reproduce] cache disabled — fresh run",
)
# Only replace the call (indented), not the function definition "def save_cache(cache):"
code = re.sub(
    r"^(\s+)save_cache\(cache\)$",
    r"\1pass  # [reproduce] cache saving disabled",
    code,
    flags=re.MULTILINE,
)
# Remove the "if n_current_files in cache: … continue" shortcut.
# The pattern spans 3 lines in the converted script.
code = re.sub(
    r"    if n_current_files in cache:.*?continue\n",
    "",
    code,
    flags=re.DOTALL,
)

# ── g) Remove CACHE_FILE constant reference to avoid confusion ────────────
code = re.sub(
    r"^CACHE_FILE\s*=.*$",
    "CACHE_FILE = '/dev/null'  # [reproduce] caching disabled",
    code,
    flags=re.MULTILINE,
)

# ── h) Delete SAE/model checkpoint dirs referenced in train_sae() ─────────
#       We already deleted them on disk; also set checkpoint_dir to a temp dir
#       so the library doesn't reload stale weights.
code = code.replace(
    "checkpoint_dir='.checkpoints/hal_sae_temp'",
    "checkpoint_dir='/tmp/_reproduce_sae_ckpt'",
)
code = code.replace(
    "checkpoint_dir='checkpoints/hal_sae_temp'",
    "checkpoint_dir='/tmp/_reproduce_sae_ckpt2'",
)

# ── i) Fix sae_beta_irt: 'resmats' dir doesn't exist, use colbench data ──
code = code.replace(
    "resmat_dir = 'resmats'",
    "resmat_dir = '../data-reeval-multi/colbench'",
)

# ── j) Fix sae_beta_irt: embedding file path ──────────────────────────────
code = code.replace(
    "emb_file = 'result/all_benchmarks_embeddings_4096_8B.pkl'",
    "emb_file = '../data-reeval-multi/hal/all_benchmarks_embeddings_4096_8B.pkl'",
)

with open(py_file, "w") as f:
    f.write(code)

print(f"    ✓ {os.path.basename(py_file)}")
PATCH_EOF

done
echo ""

# ── 5. Run each converted script ─────────────────────────────────────────────
overall_ok=true

for nb_name in "${NOTEBOOKS[@]}"; do
    py_file="${CONVERTED_DIR}/${nb_name}.py"
    [ -f "${py_file}" ] || continue

    echo "=========================================================="
    echo "  RUNNING: ${nb_name}.py"
    echo "=========================================================="

    set +e
    (cd "${WORK_DIR}" && python "${py_file}" 2>&1)
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo ""
        echo "  [ERROR] ${nb_name}.py exited with code ${rc}"
        overall_ok=false
    else
        echo ""
        echo "  [OK] ${nb_name}.py completed successfully."
    fi
    echo ""
done

# ── 6. Summary ───────────────────────────────────────────────────────────────
echo "=========================================================="
echo "  OUTPUT SUMMARY"
echo "=========================================================="
echo ""
echo "  Output directory: ${RESULT_DIR}"
echo ""

echo "  Plots (PDF):"
found_pdf=false
for f in "${RESULT_DIR}"/*.pdf; do
    [ -e "$f" ] || continue
    found_pdf=true
    echo "    $(ls -lh "$f" | awk '{print $5, $NF}')"
done
$found_pdf || echo "    (none)"

echo ""
echo "  Plots (PNG):"
found_png=false
for f in "${RESULT_DIR}"/*.png; do
    [ -e "$f" ] || continue
    found_png=true
    echo "    $(ls -lh "$f" | awk '{print $5, $NF}')"
done
$found_png || echo "    (none)"

echo ""
echo "  Converted scripts:"
for f in "${CONVERTED_DIR}"/*.py; do
    [ -e "$f" ] || continue
    echo "    ${f}"
done

echo ""
if $overall_ok; then
    echo "All notebooks completed successfully."
else
    echo "Some notebooks had errors — check output above."
fi
