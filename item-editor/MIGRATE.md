# Migration Guide - Project Reorganization

The project structure has been reorganized into a more streamlined hierarchy under the `item-editor/` root.

## New Structure

```
item-editor/
├── docent/                     # Docent library
├── hal-harness/                # HAL evaluation harness
├── eval_traces/                # Evaluation traces (input/aggregated)
├── eval_response_matrix/       # Response matrices (input/aggregated)
├── patch/                      # Patch diff files
├── result/                     # Results and outputs
│   ├── fixes/                  # Generated fixes for IFEs
│   └── .hal_data/              # Internal data (logs, traces, results, cache)
├── config/                     # Configuration files
│   ├── model_config/           # Model-specific benchmark configs
│   └── rubric/                 # Rubric templates (.txt files only)
└── script/                     # Python scripts and utilities
    └── rubric_evaluator/       # Rubric evaluation logic (docent bridge)
```

## Key Changes

1.  **Centralized Data**: All outputs (logs, traces, results, LLM cache) are now written to `item-editor/result/.hal_data/`.
2.  **Streamlined Rubrics**: Rubric templates are located in `item-editor/config/rubric/` and contain only `.txt` files to minimize redundancy.
3.  **Consolidated Scripts**: All scripts were moved to `item-editor/script/`, and `rubric_evaluator` was moved inside as a package.
4.  **Path Resolution**: `hal_common.py` and other scripts have been updated to resolve paths relative to the new `item-editor/` root.

## How to Run

Most commands should be run from the `item-editor/` directory.

### Rubric Evaluation
```bash
python script/eval_rubric.py \
    --trace-file result/.hal_data/traces/some_trace_UPLOAD.json \
    --rubric config/rubric/colbench.txt \
    --rubric-model azure_openai:gpt-5.2 \
    --failed-only -y
```

### Build Response Matrix
```bash
python script/build_response_matrix.py --prefix "moon18_" --benchmark colbench
```

### Judge IFEs
```bash
python script/judge.py \
    --pattern "*.csv" \
    --rubric-dir result/.hal_data/rubrics_output/scicode \
    --model azure_openai:gpt-5.2 \
    -y
```

### Fix IFEs with Claude
```bash
python script/claude_fixer.py --benchmark scicode --ife-only
```

## Internal Path Updates

The following environment variables and defaults were updated:
- `HAL_RESULTS_DIR` defaults to `result/.hal_data/results`
- `HAL_TRACES_DIR` defaults to `result/.hal_data/traces`
- `HAL_TMP_DIR` defaults to `result/.hal_data/tmp`
- `HAL_LOGS_DIR` defaults to `result/.hal_data/logs`
- `LLM_CACHE_PATH` defaults to `result/.hal_data/.llm_cache`

# HAL Reproducibility & Migration Guide

This document provides detailed usage instructions for the consolidated HAL evaluation scripts. It follows the end-to-end sequential workflow required for full reproducibility.

---

## 1. Infrastructure & Maintenance

### **Comprehensive Prebuild** (`scripts/prebuild_all_images.py`)
Ensures all Docker images and datasets are ready. Matches HAL's internal hash-based naming exactly.

**Usage:**
```bash
python scripts/prebuild_all_images.py [benchmarks] [OPTIONS]
```

**Options:**
- `benchmarks`: (Positional) Comma-separated or space-separated list of benchmarks to build for (e.g., `scicode scienceagentbench`). Defaults to all.
- `--force`: Rebuild images even if they already exist.

---

### **Unified Cleanup Tool** (`scripts/cleanup.py`)
Replaces the legacy `kill_all.py` and `cleanup_docker.py`.

**Usage:**
```bash
python scripts/cleanup.py [OPTIONS]
```

**Options:**
- `--aggressive`: Kill and remove **ALL** Docker containers on the system, not just benchmark ones.
- `--images`: Remove ephemeral `agent-env-*` images and prune the build cache.
- `--only-processes`: Only kill local Python/Shell processes, don't touch Docker.
- `--only-docker`: Only perform Docker cleanup, don't kill local processes.

---

## 2. Execution Engine

### **Unified Execution Engine** (`scripts/runtime_fixes.py`)
Consolidates `run_benchmark_fixes.py` and `run_all_benchmarks.py`. Replaces `bin/run_all_benchmarks.sh`.

**Usage:**
```bash
# Run a single configuration
python scripts/runtime_fixes.py --benchmark scicode --config gpt-5_scicode_tool_calling --docker

# Bulk Mode: Run multiple benchmarks sequentially
python scripts/runtime_fixes.py --prefix moon1_ --benchmarks scicode,scienceagentbench --docker

# Loop Mode: Repeat until a prefix number is reached
python scripts/runtime_fixes.py --prefix moon1_ --benchmarks scicode --until 20 --docker
```

**Complete Flag List:**

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--benchmark, -b` | None | Run a single benchmark by name (e.g. `scicode`). |
| `--benchmarks` | All | Comma-separated list of benchmarks for bulk mode. |
| `--config, -c` | All | Specific config key(s) to run (can be repeated). |
| `--model-filter, -m` | None | Filter configurations by model/key name pattern. |
| `--prefix` | `moon1_` | Global prefix for traces and Run IDs. |
| `--until` | None | Increment prefix number until reaching this integer. |
| `--repeat` | `0` | Number of times to repeat the entire execution loop. |
| `--parallel-models` | `1` | Number of model configurations to run concurrently. |
| `--parallel-tasks` | `10` | Concurrent tasks per model (passed to HAL). |
| `--docker` | `False` | Run evaluations in isolated Docker containers. |
| `--trace-mode` | None | Override `HAL_TRACE_MODE` (e.g. `local`, `weave`). |
| `--fix-only` | `False` | Only run tasks that have fixes defined in `fixes/`. |
| `--no-fix` | `False` | Baseline mode: run all tasks but ignore any fixes. |
| `--timeout` | None | Per-task timeout in seconds (overrides default). |
| `--max-tasks` | None | Limit the number of tasks per configuration. |
| `--sample-tasks` | None | Randomly sample N tasks from the dataset. |

---

### **Log Monitoring** (`scripts/watch_all.py`)
Provides real-time, color-coded tailing of benchmark logs based on your prefix.

**Usage:**
```bash
python scripts/watch_all.py --prefix moon1_
```

**Options:**
- `--prefix`: (Required) The prefix used in your `runtime_fixes.py` run. It automatically detects new log files as they are created.

---

## 3. IFE Diagnosis

### **Unified Claude Fixer** (`scripts/claude_fixer.py`)
Consolidates all legacy `claude_fixer_[benchmark].py` scripts.

**Usage:**
```bash
python scripts/claude_fixer.py --benchmark [bench] [OPTIONS]
```

**Options:**
- `--benchmark, -b`: (Required) Benchmark to process.
- `--rubric-dir`: Directory containing rubric CSVs.
- `--judge-csv`: Path to judge verdict CSV.
- `--parallel N`: Number of parallel Claude sessions.
- `--tasks-per-batch N`: Number of tasks per session.
- `--ife-only`: Only process tasks with judge verdict = 1.
- `--dry-run`: Preview prompts without running Claude.

---

## 4. Trace Processing & Consolidation

### **Trace Collector** (`scripts/collect_upload_traces.py`)
Gathers distributed `UPLOAD.json` and `RAW_SUBMISSIONS.jsonl` files.

**Usage:**
```bash
python scripts/collect_upload_traces.py --prefix [pfx] --output eval_traces
```

---

### **Merge Traces** (`scripts/merge_traces.py`)
Merges task-level traces into agent-level traces for evaluation.

**Usage:**
```bash
python scripts/merge_traces.py --input '[pattern]' --output [file].json
```

---

### **ColBench Trace Utility** (`scripts/colbench_trace_util.py`)
Specialized tools for collaborative traces.

**Usage:**
```bash
python scripts/colbench_trace_util.py [command] [OPTIONS]
```

**Commands:**
- `reconstruct`: Rebuilds `raw_logging_results` from trajectories.
- `inject-dialogues`: Adds dialogue history to merged traces.

---

## 5. Evaluation & Final Reporting

### **Rubric Evaluation** (`scripts/eval_rubric.py`)
Primary qualitative evaluation tool using Docent.

**Usage:**
```bash
python scripts/eval_rubric.py --prefix [pfx] --rubric rubric_templates/[name].txt
```

---

### **Judge Verdicts** (`scripts/judge.py`)
Aggregates evaluations into a binary IFE verdict.

**Usage:**
```bash
python scripts/judge.py --rubric-dir [dir] --prefix [pfx] --model [model]
```

---

### **Response Matrix** (`scripts/build_response_matrix.py`)
Generates the final binary result matrix and detailed metrics.

**Usage:**
```bash
python scripts/build_response_matrix.py --prefix [pfx] --traces-dir eval_traces
```

---

## Sequential Roadmap (Reproducibility Steps)

1.  **Preparation**: `scripts/prebuild_all_images.py`
2.  **Benchmark Run**: `scripts/runtime_fixes.py --prefix MY_PREFIX --parallel-models 10 --docker`
3.  **Trace Centralization**: `scripts/collect_upload_traces.py --prefix MY_PREFIX --output eval_traces`
4.  **Trace Merging**: `scripts/merge_traces.py --input 'eval_traces/traces/*MY_PREFIX*' --output MY_PREFIX_MERGED.json`
5.  **IFE Diagnosis**: `scripts/eval_rubric.py` followed by `scripts/judge.py`
6.  **Final Report**: `scripts/build_response_matrix.py --prefix MY_PREFIX --extract-subscores`
