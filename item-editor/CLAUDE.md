# CLAUDE.md - AI Assistant Guide

This file provides Claude with everything needed to understand and operate this repository.

## Project Overview

**HAL Agent Debug Pipeline** - Automated detection and fixing of Intrinsic Formation Errors (IFEs) in AI benchmark evaluations.

**Core Concept**: When AI agents fail benchmark tasks, failures can be either:
- **Score 0**: Agent capability issue (a better agent could succeed)
- **Score 1**: Benchmark defect / IFE (no agent could succeed due to benchmark bugs)

This pipeline detects IFEs through cross-model consensus and generates fixes without changing benchmark source code.

## Repository Structure

```
agent-debug/
├── README.md                      # Full documentation
├── INSTRUCTIONS_NEW_BENCHMARK.md  # Adding new benchmarks
├── requirements.txt               # Python dependencies
│
├── bin/                           # Shell scripts
│   ├── run_all_benchmarks.sh      # Main entry point
│   ├── FINAL_COMMANDS.sh          # Step 1: Merge traces
│   ├── RUBRIC_COMMANDS.sh         # Step 2: Rubric evaluation
│   ├── JUDGE_COMMANDS.sh          # Step 3: Judge aggregation
│   └── ...                        # Other utilities
│
├── scripts/                       # Python scripts
│   ├── merge_traces.py            # Merge individual traces
│   ├── extract_weave_traces.py    # Extract from W&B Weave
│   ├── eval_rubric.py             # LLM rubric evaluation
│   ├── judge.py                   # Cross-model verdict aggregation
│   ├── run_benchmark_fixes.py     # Apply fixes and re-run
│   └── claude_fixer_*.py          # Fix generation (per-benchmark)
│
├── model_configs/                 # Model configurations
│   └── model_to_baseline_*.json   # Per-benchmark model configs
│
├── rubric_templates/              # LLM grading rubrics
│   ├── scicode.txt
│   ├── corebench.txt
│   ├── colbench.txt
│   └── scienceagentbench.txt
│
├── fixes/                         # Generated fix packages
├── traces/                        # Agent execution traces
├── rubrics_output/                # Rubric evaluation CSVs
├── judge_output/                  # Final verdict CSVs
│
├── hal-harness/                   # HAL evaluation framework (submodule)
└── docent/                        # Rubric evaluation library (submodule)
```

## Supported Benchmarks

| Benchmark | Config File | Fixer Script |
|-----------|-------------|--------------|
| SciCode | `model_configs/model_to_baseline_scicode.json` | `claude_fixer_scicode.py` |
| CoreBench | `model_configs/model_to_baseline_corebench.json` | `claude_fixer_corebench.py` |
| ColBench | `model_configs/model_to_baseline_colbench.json` | `claude_fixer_colbench.py` |
| ScienceAgentBench | `model_configs/model_to_baseline_scienceagentbench.json` | `claude_fixer_scienceagentbench.py` |

## Pipeline Stages

```
1. Run Benchmarks    → ./bin/run_all_benchmarks.sh
2. Merge Traces      → ./bin/FINAL_COMMANDS.sh
3. Rubric Evaluation → ./bin/RUBRIC_COMMANDS.sh
4. Judge Aggregation → ./bin/JUDGE_COMMANDS.sh
5. Generate Fixes    → python scripts/claude_fixer_*.py
6. Apply & Re-run    → python scripts/run_benchmark_fixes.py
```

## Quick Commands

### Run Full Benchmark Suite
```bash
./bin/run_all_benchmarks.sh --prefix sun1_ --benchmarks colbench --parallel-models 10 --parallel-tasks 50 --trace-mode local
```

### Run Specific Benchmark
```bash
# SciCode
./bin/run_all_benchmarks.sh --prefix sci1_ --benchmarks scicode --parallel-models 5 --parallel-tasks 10 --docker

# ColBench
./bin/run_all_benchmarks.sh --prefix col1_ --benchmarks colbench --parallel-models 10 --parallel-tasks 50 --docker

# CoreBench
./bin/run_all_benchmarks.sh --prefix cb1_ --benchmarks corebench --parallel-models 5 --parallel-tasks 10 --docker

# ScienceAgentBench
./bin/run_all_benchmarks.sh --prefix sab1_ --benchmarks scienceagentbench --parallel-models 5 --parallel-tasks 10 --docker
```

### Process Existing Traces
```bash
# Step 1: Merge traces + extract dialogues
./bin/FINAL_COMMANDS.sh scicode      # or: corebench, sab, colbench, all

# Step 2: Run rubric evaluation
./bin/RUBRIC_COMMANDS.sh scicode

# Step 3: Aggregate verdicts
./bin/JUDGE_COMMANDS.sh scicode
```

### Generate Fixes
```bash
python scripts/claude_fixer_scicode.py \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --ife-only \
    --dry-run  # Remove for actual run
```

### Apply Fixes and Re-run
```bash
# List available fixes
python scripts/run_benchmark_fixes.py --list-benchmarks

# Dry run
python scripts/run_benchmark_fixes.py --benchmark scicode --dry-run

# Actual run
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --all-configs \
    --prefix fixed_ \
    --docker \
    --parallel-tasks 10
```

### Utility Commands
```bash
# Kill all hung processes
./bin/kill_all.sh

# Clean up Docker
./bin/cleanup_docker.sh

# Check available fixes
python scripts/run_benchmark_fixes.py --benchmark scicode --list-fixes
```

## Key Scripts Reference

### merge_traces.py
Merges individual task traces into per-model files.
```bash
python scripts/merge_traces.py \
    --input 'traces/scicode_*_UPLOAD.json' \
    --output traces/scicode_merged.json \
    --force
```

### eval_rubric.py
Runs LLM-based rubric evaluation on traces.
```bash
python scripts/eval_rubric.py \
    --trace-file traces/scicode_*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only \
    -y
```

### judge.py
Aggregates rubric evaluations into final verdicts.
```bash
python scripts/judge.py \
    --pattern "scicode_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

### run_benchmark_fixes.py
Unified runner for all benchmarks with fix application.
```bash
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --all-configs \
    --all-tasks \
    --prefix iter1_ \
    --docker \
    --parallel-models 5 \
    --parallel-tasks 10 \
    --resume
```

## Fix Package Structure

Fixes are stored in `fixes/<benchmark>/<task_id>/`:
```
fixes/scicode/12/
├── README.md                   # Human-readable explanation
├── dependency_override.json    # Additional allowed dependencies
├── instruction_override.json   # Clarified instructions
├── evaluation_override.json    # Evaluation criteria adjustments
└── status.json                 # Fix metadata
```

## Environment Setup

### Prerequisites
- Python 3.11 or 3.12 (via conda)
- Docker Engine 20.10+
- Azure CLI (for TRAPI access)

### Installation
```bash
# Create environment
conda create -n hal python=3.12 -y
conda activate hal

# Install dependencies
pip install -r requirements.txt
pip install -e ./hal-harness
pip install -e ./docent
pip install -e ./docent/docent/

# Azure authentication
az login
```

### Environment Variables
Create `hal-harness/.env`:
```bash
OPENAI_API_KEY=sk-your-key
WANDB_API_KEY=your-wandb-key
HF_TOKEN=hf_your-token
USE_DIRECT_AZURE=true
TRAPI_ENDPOINT=https://trapi.research.microsoft.com/gcr/shared
TRAPI_API_VERSION=2025-03-01-preview
TRAPI_SCOPE=api://trapi/.default
```

## Troubleshooting

### Azure Auth Issues
```bash
az logout && az login
az account get-access-token --resource api://trapi/.default
```

### Docker Issues
```bash
./bin/cleanup_docker.sh
docker builder prune -f
```

### Hung Processes
```bash
./bin/kill_all.sh
```

## Key Principles

**Make evaluation FAIR, not EASY.**

Valid fixes:
- Missing dependencies that ALL agents need
- Ambiguous output format requirements
- Overly strict numerical tolerances

Invalid fixes:
- Hints about solutions
- Simplified problems
- Pre-computed partial results

## Common Workflows

### 1. Full Pipeline for New Benchmark Run
```bash
# Run benchmarks
./bin/run_all_benchmarks.sh --prefix exp1_ --benchmarks scicode --parallel-models 10 --parallel-tasks 50 --docker

# Process results
./bin/FINAL_COMMANDS.sh scicode
./bin/RUBRIC_COMMANDS.sh scicode
./bin/JUDGE_COMMANDS.sh scicode

# Generate fixes (IFE-only)
python scripts/claude_fixer_scicode.py \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --ife-only

# Re-run with fixes
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --all-configs \
    --prefix fixed_ \
    --docker
```

### 2. Quick Test Run
```bash
# Dry run to verify setup
python scripts/run_benchmark_fixes.py --benchmark scicode --dry-run

# Run single task
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --config gpt-4.1_scicode \
    --prefix test_ \
    --docker
```

### 3. Check Existing Results
```bash
# View verdict summary
head judge_output/scicode_verdict.csv

# List fixes
python scripts/run_benchmark_fixes.py --benchmark scicode --list-fixes

# Check traces
ls traces/scicode_*.json | head
```

## File Locations Quick Reference

| What | Where |
|------|-------|
| Shell scripts | `bin/` |
| Python scripts | `scripts/` |
| Model configs | `model_configs/` |
| Rubric templates | `rubric_templates/` |
| Fixes | `fixes/<benchmark>/<task_id>/` |
| Traces | `traces/` |
| Rubric outputs | `rubrics_output/<benchmark>/` |
| Verdicts | `judge_output/` |
