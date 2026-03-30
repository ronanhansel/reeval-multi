# agent-eval

Research code and artifacts for amortized agent evaluation and item-level benchmark remediation.

This repository contains two connected pipelines:

1. `model/`: fit ARAF and baseline models that predict held-out agent performance from a partially observed response matrix.
2. `item-editor/`: diagnose benchmark defects, generate non-destructive fixes, rerun benchmarks with those fixes, and materialize the revised response matrices consumed by the model.

The modeling code currently operates on the four canonical benchmarks used throughout the paper:

- `colbench_backend_programming`
- `corebench_hard`
- `scicode`
- `scienceagentbench`

The raw data needed for reproducibility is available through [`data-collection/download_agent_eval_datasets.py`](data-collection/download_agent_eval_datasets.py), which downloads the checked-in response-matrix bundle and trace bundle into `item-editor/`.

## Current Snapshot

The repository already includes a substantial set of generated artifacts.

- Main experiment CSVs live in `model/result/main/`.
- Additional checked-in study outputs currently include `model/result/sample_size_study/` and `model/result/support_thinning_study/`.
- Paper figures are written to `paper/figures/`.
- The current checked-in post-revision matrix ensemble contains 54 ColBench revisions and 11 revisions each for CoreBench Hard, SciCode, and ScienceAgentBench.
- The current checked-in fix inventory contains 79 ColBench fixes, 11 CoreBench Hard fixes, 34 SciCode fixes, and 24 ScienceAgentBench fixes under `item-editor/result/fixes/`.

Some useful headline numbers from the current checked-in result files:

- Best checked-in post-revision Beta ARAF run: `RAW`, AUC `0.745`, RMSE `0.233` from `model/result/main/amortized_irt_raw_beta_n_max.csv`.
- Best checked-in post-revision Bernoulli ARAF run: `PCA`, AUC `0.711`, RMSE `0.250` from `model/result/main/amortized_irt_pca_bernoulli_n_1.csv`.
- Full configuration summaries are exported in `model/result/main/comprehensive_results.md` and `model/result/main/comprehensive_results.csv`.

## Repository Structure

```text
agent-eval/
├── data-collection/          # Dataset download and trace/response-matrix utilities
├── item-editor/              # Benchmark defect diagnosis and runtime fix pipeline
│   ├── config/               # Rubrics and model-to-benchmark maps
│   ├── docent/               # Submodule
│   ├── hal-harness/          # Submodule
│   ├── eval_response_matrix/ # Downloaded and generated matrices used by modeling
│   ├── eval_traces/          # Downloaded rubric outputs and traces
│   ├── result/fixes/         # Generated fix packages
│   └── script/               # Item-fixing workflow scripts
├── model/                    # ARAF training, baselines, studies, and analysis
│   ├── analysis/             # Diagnostics, summaries, appendix tables, study post-processing
│   ├── plotting/             # Figure generation
│   ├── processed_embeddings/ # PCA/SAE embeddings and interpretations
│   ├── result/               # Main and study outputs
│   └── utility/              # Shared helpers and embedding-generation tooling
├── paper/                    # LaTeX paper and generated figures
├── traces/                   # Optional raw HAL traces
└── README.md
```

## Reproducibility Setup

### 1. Clone the repository

```bash
git clone --recursive https://github.com/aims-foundation/agent-eval.git
cd agent-eval
git submodule update --init --recursive
```

### 2. Create the modeling environment

`model/reproduce.sh` expects a Conda environment named `hal`.

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n hal python=3.10 -y
conda activate hal
pip install -r requirements.txt
```

### 3. Download the canonical datasets

This is the main data bootstrap step for reproducing the checked-in modeling results.

```bash
export HF_TOKEN=hf_...
python data-collection/download_agent_eval_datasets.py
```

This script downloads:

- `item-editor/eval_response_matrix/`
- `item-editor/eval_traces/`

After download, you should have at least:

- `item-editor/eval_response_matrix/pre-revision/...`
- `item-editor/eval_response_matrix/post-revision/...`
- `item-editor/eval_response_matrix/all_benchmarks_embeddings_4096_8B.pkl`

### 4. Optional: regenerate processed embeddings

The repo already includes `model/processed_embeddings/`. If you want to rebuild PCA or SAE features, run:

```bash
python model/utility/generate_embeddings.py
```

If processed embeddings are missing, `model/amortized_irt.py` falls back to the raw embedding pickle.

## How The Data Flows

The modeling pipeline does not read benchmark tasks directly. It reads response matrices and item embeddings:

- Pre-revision matrices come from `item-editor/eval_response_matrix/pre-revision/`.
- Post-revision matrices come from `item-editor/eval_response_matrix/post-revision/`.
- Raw embeddings come from `item-editor/eval_response_matrix/all_benchmarks_embeddings_4096_8B.pkl`.
- PCA and SAE embeddings come from `model/processed_embeddings/`.

Two important conventions in the current code:

1. `model/amortized_irt.py` only trains on the four canonical benchmarks listed above, even though the pre-revision directory contains additional benchmark folders.
2. The post-revision oracle is an ensemble over multiple revised response matrices, not a single file. ColBench contributes 54 matrices in this snapshot; the other three canonical benchmarks contribute 11 each.

## Reproducing The Model Results

### One-command reproduction

The main entrypoint is [`model/reproduce.sh`](model/reproduce.sh).

Quick reproduction:

```bash
bash model/reproduce.sh --clean
```

Research-faithful multi-seed sweep:

```bash
bash model/reproduce.sh --full --clean
```

Important notes:

- The script activates the Conda environment named `hal`.
- If `model/result/` is non-empty, the script will otherwise prompt whether to overwrite or continue. Use `--clean` or `--continue` to make that explicit.
- Main experiment outputs land in `model/result/main/`.
- Study outputs land in `model/result/*_study/`.

### Study-specific reproductions

Each study can be run on its own from the same downloaded response matrices.

Pair-efficiency study:

```bash
bash model/reproduce.sh --pair-efficiency-study --clean
python -m model.plotting.main --pair-efficiency-study
```

Neighbor-support study:

```bash
bash model/reproduce.sh --neighbor-support-study --clean
```

Support-thinning study:

```bash
bash model/reproduce.sh --support-thinning-study --clean
python model/analysis/rebuild_support_thinning_summary.py
python -m model.plotting.main --support-thinning-study
```

`model/reproduce.sh --support-thinning-study` already rebuilds the summary grid internally, but rerunning `python model/analysis/rebuild_support_thinning_summary.py` is the safe manual step if you add or edit thinning outputs afterwards.

Sample-size study:

```bash
bash model/reproduce.sh --sample-size-study --clean
python -m model.plotting.main --sample-size
```

Outlier-robustness study:

```bash
bash model/reproduce.sh --outlier-robustness-study --clean
```

Full multi-study sweep:

```bash
bash model/reproduce.sh --full --pair-efficiency-study --sample-size-study
```

### Direct model fitting

The workflow below matches the way the experiments are fit in practice: use the downloaded response matrices and embeddings, then run `model/amortized_irt.py` either for a single configuration or via `model/reproduce.sh`.

Examples:

Prime the baseline cache only:

```bash
python model/amortized_irt.py \
  --baseline-only \
  --embedding-type raw \
  --baseline-embedding-type raw \
  --model-type beta \
  --n-samples max \
  --seed 42 \
  --baseline-output model/result/main/baselines/baseline_metrics.csv \
  --mirt-sweep-output model/result/main/baselines/mirt_sweep.csv
```

Run the canonical quick post-revision RAW Beta configuration:

```bash
python model/amortized_irt.py \
  --embedding-type raw \
  --baseline-embedding-type raw \
  --model-type beta \
  --n-samples max \
  --lambda-tau 0.029 \
  --seed 42 \
  --output model/result/main/amortized_irt_raw_beta_n_max.csv \
  --baseline-output model/result/main/baselines/baseline_metrics.csv \
  --mirt-sweep-output model/result/main/baselines/mirt_sweep.csv
```

Run the canonical quick post-revision PCA Bernoulli configuration:

```bash
python model/amortized_irt.py \
  --embedding-type pca \
  --baseline-embedding-type pca \
  --model-type bernoulli \
  --n-samples 1 \
  --lambda-tau 0.0155 \
  --seed 42 \
  --output model/result/main/amortized_irt_pca_bernoulli_n_1.csv \
  --baseline-output model/result/main/baselines/baseline_metrics.csv \
  --mirt-sweep-output model/result/main/baselines/mirt_sweep.csv
```

Useful direct-fit knobs:

- `--pre-revision 4|8|16|32|64|max`: switch from post-revision to balanced pre-revision training.
- `--j-percentage 0.1 ... 1.0`: subsample the fraction of observed items.
- `--user-count 4|8|16|32`: subsample the number of observed agents in post-revision runs.
- `--baseline-only`: compute only classical baselines.
- `--parallel N`: parallelize seed and tau jobs.
- `--save-weights`: save learned weights for interpretability plots.

## Generating Plots

The checked-in plot entrypoint is:

```bash
python -m model.plotting.main --all
```

Selective plot generation:

```bash
python -m model.plotting.main --benchmarks
python -m model.plotting.main --comparison
python -m model.plotting.main --sample-size
python -m model.plotting.main --pair-efficiency-study
python -m model.plotting.main --support-thinning-study
python -m model.plotting.main --interpretability
python -m model.plotting.main --rubrics
python -m model.plotting.main --appendix
```

Outputs are written under `paper/figures/`.

Current plotting coverage in this snapshot:

- Supported modules: benchmark heatmaps, comparison, sample-size, pair-efficiency, support-thinning, interpretability, rubric statistics, appendix figures.
- `neighbor-support` and `outlier-robustness` study CSVs can be generated by the experiment pipeline, but no dedicated plotting module is currently checked in for them.

If you only want to regenerate figures from existing result CSVs, you do not need to rerun `model/reproduce.sh`; the plotting commands above are sufficient.

## Analysis Utilities

The study and summary scripts now live under `model/analysis/`.

Useful analysis commands:

```bash
python model/analysis/export_best_results.py
python model/analysis/generate_appendix_table.py
python model/analysis/calculate_agreement.py
python model/analysis/diagnose_pre_post_stability.py
python model/analysis/calibration_study.py
python model/analysis/rebuild_support_thinning_summary.py
```

What they produce:

- `export_best_results.py`: refreshes `model/result/main/comprehensive_results.csv` and `.md`.
- `generate_appendix_table.py`: writes the appendix LaTeX table for all setups.
- `calculate_agreement.py`: refreshes `agreement_results.md`.
- `diagnose_pre_post_stability.py`: regenerates pre/post variance and stability diagnostics.
- `calibration_study.py`: runs the standalone post-hoc calibration analysis.
- `rebuild_support_thinning_summary.py`: rebuilds `model/result/support_thinning_study/support_thinning_grid.csv` from thinning sweep outputs.

## Item-Fixing Reproducibility Guide

The `item-editor/` pipeline is used to identify intrinsic benchmark defects, synthesize fixes, rerun the affected tasks, and materialize revised matrices.

### 1. Create the item-editor environment

```bash
cd item-editor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./hal-harness
pip install -e ./docent/docent/
pip install -e ./docent/
cd ..
```

### 2. Apply the pinned submodule patches

```bash
cd item-editor
bash patch/apply_patches.sh
cd ..
```

This script resets the submodules to pinned base commits and reapplies local patches:

- `hal-harness` base commit: `edfbc3023173e0017625401e99045263ff61f3d1`
- `docent` base commit: `9700ec0ac41b0f02e8ae32c6d987363448f5a364`

That is the reproducible path, but it is destructive inside those submodule worktrees.

### 3. Configure secrets

```bash
cp item-editor/hal-harness/.env.template item-editor/hal-harness/.env
cp item-editor/docent/.env.template item-editor/docent/.env
```

Populate the keys required by the models and backends you intend to use. The runtime scripts default to `item-editor/hal-harness/.env` unless you override `HAL_DOTENV_PATH`.

### 4. Run the item-fixing loop

Example for `scicode`:

```bash
cd item-editor

python script/utils/prebuild_all_images.py scicode

python script/fix/runtime_fixes.py \
  --benchmark scicode \
  --prefix base_ \
  --docker

python script/trace/collect_upload_traces.py \
  --prefix base_ \
  --output eval_traces

python script/trace/merge_traces.py \
  --input 'eval_traces/traces/*base_*' \
  --output result/.hal_data/traces/merged_base.json

python script/eval/eval_rubric.py \
  --trace-file result/.hal_data/traces/merged_base.json \
  --rubric config/rubric/scicode.txt \
  --failed-only -y

python script/eval/judge.py \
  --pattern "base_*" \
  --rubric-dir result/.hal_data/rubrics_output/scicode \
  --model openai:gpt-4o -y

python script/fix/claude_fixer.py \
  --benchmark scicode \
  --ife-only \
  --judge-csv result/.hal_data/judge_output/scicode_verdict.csv

python script/fix/runtime_fixes.py \
  --benchmark scicode \
  --prefix fixed_ \
  --docker \
  --fix-only
```

The main outputs of that loop are:

- Generated fixes in `item-editor/result/fixes/<benchmark>/<task_id>/`
- Rerun traces in `item-editor/result/.hal_data/`
- Updated response-matrix artifacts under `item-editor/eval_response_matrix/`

### Runtime fix package layout

Each task-level fix directory may contain files such as:

- `instruction_override.json`
- `evaluation_override.json`
- `env_override.json`
- `dependency_override.json`
- `input_override.json`
- `simulated_user_override.json`
- `README.md`

These are runtime overlays. The pipeline is designed to avoid editing the original benchmark sources directly.

## What To Cite Or Inspect For Results

For a concise audit trail, the most useful checked-in result files are:

- `model/result/main/comprehensive_results.md`
- `model/result/statistics.md`
- `agreement_results.md`
- `pre_post_stability.md`

Helpful regeneration utilities:

```bash
python model/analysis/export_best_results.py
python model/analysis/calculate_agreement.py
python model/analysis/diagnose_pre_post_stability.py
python data-collection/analyze_pre_revision_rubrics.py
python data-collection/analyze_traces.py
```

## Practical Recommendations

- If your goal is to reproduce the paper figures and headline model numbers, download the datasets, use the `hal` environment, run `model/reproduce.sh`, and regenerate plots from `model.plotting.main`.
- If your goal is to continue item remediation, work inside `item-editor/`, apply the pinned patches first, and treat `item-editor/eval_response_matrix/` as the interface between the fixing pipeline and the modeling pipeline.
- If you want a clean modeling rerun without rerunning HAL, you only need `item-editor/eval_response_matrix/`, `model/processed_embeddings/`, and the root Python dependencies.
