# Migration Guide: Script Consolidation & Streamlining

This document outlines the recent refactoring of the `scripts/` directory. Multiple benchmark-specific or redundant scripts have been merged into unified utilities to simplify the workflow and reduce codebase clutter.

---

## Summary of Changes

| Old Script(s) | New Unified Script | Primary Purpose |
| :--- | :--- | :--- |
| `claude_fixer_colbench.py`<br>`claude_fixer_corebench.py`<br>`claude_fixer_scicode.py`<br>`claude_fixer_scienceagentbench.py` | **`scripts/claude_fixer.py`** | Use Claude Code CLI to diagnose and fix IFEs across all benchmarks. |
| `prebuild_agent_envs.py`<br>`build_scicode_image.py`<br>`ensure_corebench_data.py` | **`scripts/prebuild_all_images.py`** | One-stop prebuild for base images, agent envs, and benchmark data. |
| `kill_all.py`<br>`cleanup_docker.py` | **`scripts/cleanup.py`** | Terminate hung processes and perform thorough Docker maintenance. |
| `fix_colbench_traces.py`<br>`add_colbench_dialogues.py` | **`scripts/colbench_trace_util.py`** | Post-processing utilities specifically for ColBench traces. |
| `get_task_counts.py` | *Integrated into `run_all_benchmarks.py`* | Task counting is now automatic during benchmark runs. |

---

## 1. Unified Claude Fixer (`scripts/claude_fixer.py`)

Handles IFE (Intrinsic Formation Error) analysis and fix generation using the Claude Code CLI.

### Usage
```bash
python scripts/claude_fixer.py --benchmark [benchmark] [options]
```

### Key Flags
- `--benchmark, -b`: (Required) `scicode`, `scienceagentbench`, `corebench`, or `colbench`.
- `--task-ids`: Specific tasks to process (e.g., `11 74`).
- `--ife-only`: Only process tasks with a judge verdict of 1 (requires `--judge-csv`).
- `--parallel N`: Run N Claude sessions concurrently.
- `--tasks-per-batch N`: Number of tasks for each Claude instance to handle sequentially.
- `--dry-run`: Preview prompts without executing Claude Code.
- `--list-ife-tasks`: Show all tasks with potential IFEs from rubrics and exit.

---

## 2. Comprehensive Prebuild (`scripts/prebuild_all_images.py`)

Ensures all infrastructure requirements are met before a benchmark run. It detects your HAL configuration automatically.

### Usage
```bash
# Build everything
python scripts/prebuild_all_images.py

# Build only specific benchmarks
python scripts/prebuild_all_images.py scicode scienceagentbench
```

### Key Flags
- `--force`: Rebuild images even if they already exist.
- `[benchmarks]`: Positional arguments to limit builds to specific benchmarks.

---

## 3. Unified Cleanup Tool (`scripts/cleanup.py`)

Aggressively cleans up the environment. Replaces the old "kill" and "docker cleanup" scripts.

### Usage
```bash
python scripts/cleanup.py [options]
```

### Key Flags
- `--aggressive`: Kill and remove **ALL** Docker containers on the system.
- `--images`: Remove ephemeral `agent-env-*` images and prune the build cache.
- `--only-processes`: Only kill local Python/Shell processes, don't touch Docker.
- `--only-docker`: Only perform Docker cleanup, don't kill local processes.

---

## 4. ColBench Trace Utility (`scripts/colbench_trace_util.py`)

A specialized tool for managing ColBench's unique multi-turn dialogue traces.

### Commands
#### `reconstruct`
Fixes missing `raw_logging_results` in ColBench traces by pulling from `_RAW_SUBMISSIONS.jsonl`.
```bash
python scripts/colbench_trace_util.py reconstruct --prefix moon18
```

#### `inject-dialogues`
Adds full dialogue history and task metadata to merged traces for rubric evaluation.
```bash
python scripts/colbench_trace_util.py inject-dialogues [trace_file] --results-dir [dir] --run-pattern [pattern] --output [file]
```

---

## 5. Main Runner Improvements (`scripts/run_all_benchmarks.py`)

- **Automatic Task Counting**: You no longer need to run `get_task_counts.py`. Phase 2 now prints a comprehensive summary automatically.
- **Integrated Prebuild**: Uses the new `prebuild_all_images.py` logic to ensure data/images are ready in Phase 1.

---

## Legacy Scripts
The following scripts have been **removed**. If you have automated pipelines using them, please update to the new unified equivalents listed above:
- `scripts/claude_fixer_colbench.py`
- `scripts/claude_fixer_corebench.py`
- `scripts/claude_fixer_scicode.py`
- `scripts/claude_fixer_scienceagentbench.py`
- `scripts/prebuild_agent_envs.py`
- `scripts/build_scicode_image.py`
- `scripts/ensure_corebench_data.py`
- `scripts/kill_all.py`
- `scripts/cleanup_docker.py`
- `scripts/get_task_counts.py`
- `scripts/fix_colbench_traces.py`
- `scripts/add_colbench_dialogues.py`
