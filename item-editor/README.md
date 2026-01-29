# Item Editor Pipeline

This directory contains a standalone implementation of the **Automated Item-Level Fixing Pipeline** for benchmark evaluation. The pipeline identifies and fixes Intrinsic Formation Errors (IFEs) in AI agent benchmarks without modifying the benchmark source code.

## Overview

The pipeline consists of 5 steps:

```
Traces → Rubric Evaluation → Verdict Aggregation → Fix Generation → Fix Application
```

1. **Rubric Evaluation**: Grade failed tasks against benchmark-specific rubrics using LLMs
2. **Verdict Aggregation**: Combine cross-model evaluations into final IFE verdicts
3. **Fix Generation**: Use Claude to diagnose IFEs and generate fix packages
4. **Fix Application**: Apply fixes and re-run benchmark evaluations

## Directory Structure

```
item-editor/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── scripts/                  # Pipeline scripts
│   ├── eval_rubric.py        # Rubric evaluation
│   ├── judge.py              # Verdict aggregation
│   ├── claude_fixer_*.py     # Fix generation (per benchmark)
│   ├── run_*_fixes.py        # Fix application (per benchmark)
│   ├── merge_traces.py       # Trace merging utility
│   ├── extract_weave_traces.py   # Weave log extraction
│   └── add_colbench_dialogues.py # ColBench dialogue extraction
├── prompts/                  # Rubric templates
│   ├── scicode.txt
│   ├── corebench.txt
│   ├── scienceagentbench.txt
│   ├── colbench.txt
│   ├── swebench.txt
│   ├── usaco.txt
│   ├── assistantbench.txt
│   └── *.schema.json         # Output schemas
├── configs/                  # Model configurations
│   └── model_to_baseline_*.json
└── examples/                 # Example outputs (optional)
```

## Prerequisites

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n item-editor python=3.11 -y
conda activate item-editor

# Install requirements
pip install -r requirements.txt

# Clone and install docent (required for rubric evaluation)
git clone https://github.com/TransluceAI/docent.git
pip install -e docent/docent/
pip install -e docent/
```

### 2. Set Environment Variables

Create a `.env` file or export these variables:

```bash
# Required for rubric evaluation and judging
export OPENAI_API_KEY="your-openai-key"

# For Azure/TRAPI (Microsoft internal)
export TRAPI_ENDPOINT="https://trapi.research.microsoft.com/gcr/shared"
export TRAPI_API_VERSION="2025-03-01-preview"

# For Claude fixer
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Weave trace extraction (optional)
export WANDB_API_KEY="your-wandb-key"
```

### 3. Download Traces from HuggingFace

```bash
# Install huggingface_hub CLI
pip install huggingface_hub

# Download traces to data-reeval-multi/traces/
# Download traces (update with your dataset path)
huggingface-cli download ronanhansel/data-reeval-multi \
    --local-dir ../data-reeval-multi \
    --repo-type dataset

# Your traces will be in data-reeval-multi/traces/
```

## Pipeline Steps

### Step 1: Rubric Evaluation

Evaluate failed tasks against benchmark-specific rubrics.

```bash
# Basic usage
python scripts/eval_rubric.py \
    --trace-file ../data-reeval-multi/traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --rubric prompts/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only \
    -y

# With output directory
python scripts/eval_rubric.py \
    --trace-file ../data-reeval-multi/traces/*.json \
    --rubric prompts/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --output-dir rubrics_output/scicode \
    --failed-only \
    --parallel 5 \
    -y
```

**Key Options:**
- `--trace-file`: Path to merged trace JSON file(s)
- `--rubric`: Path to rubric template
- `--rubric-model`: LLM for evaluation (e.g., `openai:gpt-5.2`, `openai:o3-mini`)
- `--failed-only`: Only evaluate tasks that failed
- `--parallel N`: Run N tasks in parallel
- `--reasoning-effort`: For reasoning models (low/medium/high)
- `-y`: Auto-confirm prompts

**Output:** CSV files in `rubrics_output/{benchmark}/` with columns:
- `task_id`, `criteria`, `grade` (0=no IFE, 1=IFE detected), `explanation`, `model_run`

### Step 2: Verdict Aggregation (Judging)

Aggregate cross-model rubric evaluations into final verdicts.

```bash
python scripts/judge.py \
    --pattern "scicode_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 5 \
    -y
```

**Key Options:**
- `--pattern`: Glob pattern for rubric CSV files to aggregate
- `--rubric-dir`: Directory containing rubric CSVs
- `--output`: Output verdict CSV path
- `--model`: LLM for verdict synthesis
- `--priority-override`: Prioritize certain models (e.g., `o3,o4-mini`)

**Output:** Verdict CSV with columns:
- `task_id`, `final_grade`, `satisfies_rubric`, `num_evaluations`, `model_runs`, `reasoning`

### Step 3: Fix Generation (Claude Fixer)

Generate fix packages for IFE tasks using Claude Code.

```bash
# List IFE tasks (Grade=1)
python scripts/claude_fixer_scicode.py --list-ife-tasks

# Fix a specific task
python scripts/claude_fixer_scicode.py \
    --task-id 12 \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --trace-files ../data-reeval-multi/traces/scicode_*.json

# Fix all IFE tasks
python scripts/claude_fixer_scicode.py \
    --all-ife \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --trace-files ../data-reeval-multi/traces/scicode_*.json \
    --skip-existing

# Batch mode (more efficient)
python scripts/claude_fixer_scicode.py \
    --all-ife \
    --batch \
    --tasks-per-batch 3 \
    --skip-existing
```

**Available Fixer Scripts:**
| Benchmark | Script |
|-----------|--------|
| SciCode | `claude_fixer_scicode.py` |
| CoreBench | `claude_fixer_corebench.py` |
| ScienceAgentBench | `claude_fixer_scienceagentbench.py` |
| ColBench | `claude_fixer_colbench.py` |

**Output:** Fix packages in `fixes/{benchmark}/{task_id}/`:
```
fixes/scicode/12/
├── README.md                   # Human-readable explanation
├── env_override.json           # Environment fixes (packages, libs, timeouts)
├── evaluation_override.json    # Evaluation criteria adjustments
├── instruction_override.json   # Task clarifications
└── status.json                 # Application status
```

### Step 4: Fix Application

Apply fixes and re-run HAL evaluations.

```bash
# List available fixes
python scripts/run_scicode_fixes.py --list-fixes

# Dry run (preview what would happen)
python scripts/run_scicode_fixes.py \
    --task-id 12 \
    --prefix fixed_ \
    --dry-run

# Run fixes for specific tasks
python scripts/run_scicode_fixes.py \
    --task-id 12 \
    --task-id 35 \
    --prefix iter1_ \
    --docker

# Run all available fixes
python scripts/run_scicode_fixes.py \
    --all \
    --prefix honey_ \
    --docker \
    --parallel 3
```

**Available Fix Runners:**
| Benchmark | Script |
|-----------|--------|
| SciCode | `run_scicode_fixes.py` |
| CoreBench | `run_corebench_fixes.py` |
| ScienceAgentBench | `run_scienceagentbench_fixes.py` |
| ColBench | `run_colbench_fixes.py` |
| USACO | `run_usaco_fixes.py` |

**Key Options:**
- `--task-id ID`: Run fix for specific task(s)
- `--all`: Run all available fixes
- `--prefix PREFIX`: Prefix for new trace filenames (e.g., `honey_`)
- `--docker`: Run in Docker container
- `--dry-run`: Preview without executing
- `--model MODEL`: Override model (defaults to original failing model)

### Step 5: Before/After Comparison

Re-run rubric evaluation and judging on fixed traces to measure improvement.

```bash
# Re-evaluate fixed traces
python scripts/eval_rubric.py \
    --trace-file traces/honey_*.json \
    --rubric prompts/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --output-dir rubrics_output/scicode_after \
    --failed-only -y

# Re-aggregate verdicts
python scripts/judge.py \
    --pattern "*.csv" \
    --rubric-dir rubrics_output/scicode_after \
    --output judge_output/scicode_after_verdict.csv \
    --model openai:gpt-5.2 -y

# Compare defect rates
echo "Before fixes:"
grep ",1," rubrics_output/scicode/*.csv | wc -l
echo "After fixes:"
grep ",1," rubrics_output/scicode_after/*.csv | wc -l
```

## Trace Processing Utilities

### Merge Individual Traces

```bash
python scripts/merge_traces.py \
    --pattern "scicode_honey__*.json" \
    --trace-dir ../data-reeval-multi/traces \
    --output traces/scicode_honey_MERGED_UPLOAD.json
```

### Extract Weave Logs (SciCode/CoreBench/SAB)

```bash
python scripts/extract_weave_traces.py \
    --trace-file traces/scicode_honey_MERGED_UPLOAD.json \
    --run-prefix "scicode_honey_openai_gpt-4_1"
```

### Extract ColBench Dialogues

```bash
python scripts/add_colbench_dialogues.py \
    traces/col_ivy_MERGED_UPLOAD.json \
    --results-dir results/colbench_backend_programming \
    --run-pattern "col_ivy_gpt-4_1_*" \
    --output traces/col_ivy_openai_gpt-4_1_WITH_DIALOGUES.json
```

## Rubric Template Format

All rubrics follow this structure:

```
# {Benchmark} Intrinsic Formation Error Detection Rubric

## Purpose
## Scoring (0 or 1)
## Two-Question Framework
  - Question 1: Does defect exist?
  - Question 2: Did defect cause failure?
## Deficiency Categories (benchmark-specific)
## CRITICAL EXCLUSIONS: Agent Capability Issues
## Evidence Requirements
## Response Format (JSON)
## Common Failure Patterns
```

**Grade 0**: No IFE - agent capability issue (agent could have succeeded)
**Grade 1**: IFE detected - benchmark defect (no agent could succeed)

## Model Configuration

Model configs in `configs/model_to_baseline_{benchmark}.json` specify:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1-2025-04-14",
    "short_name": "gpt-4.1",
    "baseline_trace": "scicode_agent_gpt4120250414_UPLOAD.json",
    "max_steps": 5
  }
}
```

## Supported Benchmarks

| Benchmark | Rubric | Fixer | Fix Runner |
|-----------|--------|-------|------------|
| SciCode | scicode.txt | claude_fixer_scicode.py | run_scicode_fixes.py |
| CoreBench | corebench.txt | claude_fixer_corebench.py | run_corebench_fixes.py |
| ScienceAgentBench | scienceagentbench.txt | claude_fixer_scienceagentbench.py | run_scienceagentbench_fixes.py |
| ColBench | colbench.txt | claude_fixer_colbench.py | run_colbench_fixes.py |
| SWE-bench | swebench.txt | - | - |
| USACO | usaco.txt | - | run_usaco_fixes.py |
| AssistantBench | assistantbench.txt | - | - |

## Quick Start Example

```bash
# 1. Setup
cd item-editor
pip install -r requirements.txt
git clone https://github.com/TransluceAI/docent.git
pip install -e docent/docent/ && pip install -e docent/

# 2. Download traces
huggingface-cli download ronanhansel/data-reeval-multi \
    --local-dir ../data-reeval-multi --repo-type dataset

# 3. Create output directories
mkdir -p rubrics_output/scicode judge_output fixes/scicode

# 4. Run rubric evaluation
python scripts/eval_rubric.py \
    --trace-file ../data-reeval-multi/traces/scicode_*.json \
    --rubric prompts/scicode.txt \
    --output-dir rubrics_output/scicode \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y

# 5. Aggregate verdicts
python scripts/judge.py \
    --pattern "*.csv" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 -y

# 6. Generate fixes
python scripts/claude_fixer_scicode.py \
    --all-ife \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv

# 7. Apply fixes (requires HAL harness)
python scripts/run_scicode_fixes.py --all --prefix honey_ --docker
```

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `docent not found` | Missing docent installation | `pip install -e docent/docent/ && pip install -e docent/` |
| `TRAPI permission denied` | Wrong API version | Set `TRAPI_API_VERSION=2025-03-01-preview` |
| `No accounts in MSAL cache` | Azure auth expired | Run `az login` to refresh |
| `rate limit exceeded` | Too many parallel requests | Reduce `--parallel` value |

### Debug Tips

```bash
# Check IFE tasks from verdicts
grep ",1," judge_output/*_verdict.csv | cut -d',' -f1 | sort -u

# View fix status
cat fixes/scicode/*/status.json

# Compare before/after
diff rubrics_output/scicode/ rubrics_output/scicode_after/
```

## References

- [HAL Harness](https://github.com/princeton-pli/hal-harness) - Princeton PLI's Holistic Agent Leaderboard evaluation framework
- [Docent](https://github.com/TransluceAI/docent) - TransluceAI's agent analysis platform for rubric-based evaluation
- [HAL Leaderboard](https://hal.cs.princeton.edu/) - Official HAL leaderboard website
