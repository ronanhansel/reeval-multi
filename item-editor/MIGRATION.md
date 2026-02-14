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

### **Unified Fix Runner** (`scripts/run_benchmark_fixes.py`)
The primary tool for running evaluations with optional fix application.

**Usage:**
```bash
python scripts/run_benchmark_fixes.py --benchmark [bench] [OPTIONS]
```

**Core Options:**
- `--benchmark, -b`: (Required) Benchmark name (`scicode`, `scienceagentbench`, `corebench`, `colbench`, `usaco`).
- `--all-benchmarks`: Run ALL benchmarks that have fixes available.
- `--config, -c`: Specific config key to run (e.g., `gpt-5_scicode_tool_calling`). Can be repeated.
- `--all-configs`: Run all configurations in the benchmark's config file.
- `--docker`: Run with Docker isolation (highly recommended).
- `--prefix, -p`: Output prefix for traces (default: `run_`).

**Execution Options:**
- `--all-tasks`: Run the entire benchmark, applying fixes where available in `fixes/`.
- `--no-fix`: Run the entire benchmark WITHOUT applying any fixes (baseline).
- `--parallel-models N`: Run N model configurations concurrently.
- `--parallel-tasks N`: Run N tasks concurrently within each evaluation (uses HAL `--max_concurrent`).
- `--resume`: Resume prior runs when available (uses HAL `--continue_run`).
- `--sample-tasks N`: Randomly sample N tasks from the dataset.
- `--sample-seed N`: Seed for reproducible sampling.
- `--max-tasks N`: Limit number of tasks per configuration.
- `--timeout N`: Per-task timeout in seconds (overrides default).

**Filtering & Inspection:**
- `--agent, -a`: Filter configs by agent name (e.g., `scicode_tool_calling_agent`).
- `--model-filter, -m`: Filter configs by model pattern (e.g., `gpt-5`).
- `--list-configs`: List available configurations and exit.
- `--list-fixes`: List task IDs with fixes for the benchmark.
- `--dry-run`: Show what would run without executing.

---

### **Bulk Runner** (`scripts/run_all_benchmarks.py`)
Wrapper for sequential multi-benchmark execution.

**Usage:**
```bash
python scripts/run_all_benchmarks.py --prefix [pfx] [OPTIONS]
```

**Options:**
- `--prefix`: (Default: `moon1_`) Global prefix for all benchmark runs.
- `--benchmarks`: Comma-separated list of benchmarks to run.
- `--parallel-models N`: Concurrent configurations per benchmark.
- `--parallel-tasks N`: Concurrent tasks per configuration.
- `--fix-only`: Only run tasks that have corresponding fixes.
- `--no-fix`: Run all tasks without applying fixes.
- `--timeout N`: Per-task timeout.
- `--repeat N`: Repeat the entire benchmark loop N times.
- `--until N`: Run until the prefix number reaches N.

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
- `--judge-csv`: Path to judge verdict CSV (used for `--ife-only`).
- `--trace-files`: Trace files to extract context from.
- `--task-ids`: Specific task IDs to process.
- `--max-tasks N`: Limit number of tasks to process.
- `--min-grade F`: Min rubric grade to consider as IFE (default: 0.5).
- `--skip-existing`: Skip tasks that already have fixes.
- `--parallel N`: Number of parallel Claude sessions.
- `--tasks-per-batch N`: Number of tasks per session.
- `--ife-only`: Only process tasks with judge verdict = 1.
- `--list-ife-tasks`: List all tasks with detected IFEs and exit.
- `--dry-run`: Preview prompts without running Claude.

---

## 4. Trace Processing & Consolidation

### **Trace Collector** (`scripts/collect_upload_traces.py`)
Gathers distributed `UPLOAD.json` and `RAW_SUBMISSIONS.jsonl` files.

**Usage:**
```bash
python scripts/collect_upload_traces.py --prefix [pfx] --output eval_traces
```

**Options:**
- `--prefix`: (Required) Prefix pattern (string or regex) to match run directories.
- `--output`: (Required) Destination directory for collected traces.
- `--run-root`: Override run root path.
- `--benchmark`: Select specific benchmark(s) to process.

---

### **Merge Traces** (`scripts/merge_traces.py`)
Merges task-level traces into agent-level traces for evaluation.

**Usage:**
```bash
python scripts/merge_traces.py --input '[pattern]' --output [file].json [OPTIONS]
```

**Options:**
- `--input`: (Required) Glob pattern for trace files. Can be repeated.
- `--output`: (Required) Destination path for merged JSON.
- `--run-id`: Override the stored `run_id`.
- `--agent-name`: Override the stored `agent_name`.
- `--date`: Override the stored `date`.
- `--force`: Overwrite existing output.

---

### **ColBench Trace Utility** (`scripts/colbench_trace_util.py`)
Specialized tools for collaborative traces.

**Usage:**
```bash
python scripts/colbench_trace_util.py [command] [OPTIONS]
```

**Commands:**
- `reconstruct`: Rebuilds `raw_logging_results` from raw JSONL trajectories.
    - `--prefix`: (Required) Prefix to match.
    - `--traces-dir`: Path to traces.
    - `--raw-dir`: Path to raw submissions.
- `inject-dialogues`: Adds dialogue history to merged traces.
    - `trace_file`: Merged trace path.
    - `--results-dir`: Path to raw results.
    - `--run-pattern`: Pattern to match task subdirs.
    - `--output`: Result path.

---

## 5. Evaluation & Final Reporting

### **Rubric Evaluation** (`scripts/eval_rubric.py`)
Primary qualitative evaluation tool.

**Usage:**
```bash
python scripts/eval_rubric.py [OPTIONS]
```

**Options:**
- `--trace-file`: Trace JSON path. Can be repeated.
- `--prefix`: Regex prefix to group files by.
- `--rubric`: Path to rubric `.txt` template.
- `--rubric-model`: Model ID (e.g., `openai:gpt-5.2`).
- `--failed-only`: Only evaluate tasks in the `failed_tasks` list.
- `--fixes-only`: Only evaluate tasks that have fixes in `fixes/`.
- `--no-cache`: Disable LLM response caching.
- `--max-batch-messages N`: Maximum messages per API call.

---

### **Judge Verdicts** (`scripts/judge.py`)
Aggregates evaluations into a binary IFE verdict.

**Usage:**
```bash
python scripts/judge.py --rubric-dir [dir] --model [model] [OPTIONS]
```

**Options:**
- `--prefix`: Regex prefix to group CSVs.
- `--rubric-dir`: (Required) Directory containing evaluation CSVs.
- `--model`: (Required) Judge model ID.
- `--original`: Treat as pre-revision baseline data.
- `--common-only`: Only judge tasks existing in ALL matched files.
- `--parallel N`: Concurrent LLM requests.

---

### **Response Matrix** (`scripts/build_response_matrix.py`)
Generates the final binary result matrix and detailed metrics.

**Usage:**
```bash
python scripts/build_response_matrix.py --prefix [pfx] [OPTIONS]
```

**Options:**
- `--prefix`: (Required) Prefix pattern to match.
- `--traces-dir`: Path to collected traces folder.
- `--reeval`: Execute agent code in Docker to verify success bits.
- `--extract-subscores`: Save separate CSVs for Success Rate, CodeBERT, etc.
- `--original`: Save to `eval_response_matrix/pre-revision`.
- `--output`: Override output directory.

---

## Sequential Roadmap (Reproducibility Steps)

1.  **Preparation**: `scripts/prebuild_all_images.py`
2.  **Benchmark Run**: `scripts/run_all_benchmarks.py --prefix MY_PREFIX`
3.  **Trace Centralization**: `scripts/collect_upload_traces.py --prefix MY_PREFIX --output eval_traces`
4.  **Dialogue Fix** (ColBench Only): `scripts/colbench_trace_util.py reconstruct --prefix MY_PREFIX`
5.  **Trace Merging**: `scripts/merge_traces.py --input 'eval_traces/traces/*MY_PREFIX*' --output MY_PREFIX_MERGED.json`
6.  **IFE Diagnosis**: `scripts/eval_rubric.py` followed by `scripts/judge.py`
7.  **Final Report**: `scripts/build_response_matrix.py --prefix MY_PREFIX --extract-subscores`
