# HAL Agent Debug Pipeline

Automated Item Fixing Pipeline for benchmark evaluation. The core innovation is **item-level fixing** - modifying agent configurations, prompt templates, and runtime parameters to eliminate false-positive benchmark defects without changing the benchmark source code itself.

## Key Features

- **Rubric-based IFE Detection**: LLM evaluates failed tasks against benchmark-specific rubrics
- **Cross-model Consensus**: Aggregate verdicts across multiple models to reduce false positives
- **Automated Fix Generation**: Claude-powered fix generation for identified benchmark defects
- **Unified Runner**: Single script to run all benchmarks with configurable parallelism

## Supported Benchmarks

| Benchmark | Fixer Script | Status |
|-----------|-------------|--------|
| **SciCode** | `claude_fixer_scicode.py` | Active |
| **CoreBench** | `claude_fixer_corebench.py` | Active |
| **ScienceAgentBench** | `claude_fixer_scienceagentbench.py` | Active |
| **ColBench** | `claude_fixer_colbench.py` | Active |

**Runner**: All benchmarks use the unified `run_benchmark_fixes.py` script.

---

## Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.11 or 3.12 (via conda)
- **Docker**: Docker Engine 20.10+
- **Conda**: Miniconda or Anaconda

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/ronanhansel/agent-debug.git
cd agent-debug

# Create conda environment
conda create -n hal python=3.12 -y
conda activate hal

# Install dependencies
pip install -r requirements.txt
pip install gdown
pip install -e ./hal-harness
pip install -e ./docent
pip install -e ./docent/docent/

# Install Azure SDK (for TRAPI access)
pip install azure-identity msal

# Build Docker image (required for --docker mode)
cd hal-harness
docker build -t hal-agent-runner:latest -f hal/utils/docker/Dockerfile .
cd ..
```

### Environment Variables

Create `hal-harness/.env`:

```bash
# API Keys
OPENAI_API_KEY=sk-your-key-here
WANDB_API_KEY=your-wandb-key
HF_TOKEN=hf_your-token

# For Azure/TRAPI direct access (recommended)
USE_DIRECT_AZURE=true
TRAPI_ENDPOINT=https://trapi.research.microsoft.com/gcr/shared
TRAPI_API_VERSION=2025-03-01-preview
TRAPI_SCOPE=api://trapi/.default
```

### Azure Authentication

```bash
# Login to Azure (required for TRAPI access)
az login

# Verify authentication
az account show
az account get-access-token --resource api://trapi/.default
```

---

## Pipeline Overview

The complete pipeline has 5 stages:

```
1. Run Benchmarks    → Generate traces with hal-eval
2. Merge & Extract   → Combine traces, extract conversation logs
3. Rubric Evaluation → LLM grades failures (IFE detection)
4. Judge Aggregation → Cross-model consensus
5. Apply Fixes       → Re-run with fixes applied
```

---

## Running the Full Pipeline

### Option 1: Run All Benchmarks (Recommended)

```bash
# Run all benchmarks with all models
./bin/run_all_benchmarks.sh --prefix sun1_ --parallel-models 10 --parallel-tasks 50 --trace-mode local

# Options:
#   --prefix PREFIX     Run ID prefix (e.g., sun1_, moon2_)
#   --benchmarks LIST   Comma-separated benchmarks (scicode,corebench,scienceagentbench,colbench)
#   --parallel-models N Number of models to run concurrently
#   --parallel-tasks N  Number of tasks per model concurrently
#   --trace-mode MODE   Trace storage mode (local or weave)
#   --docker            Run in Docker containers (recommended)
#   --sample-tasks N    Run only N random tasks (for testing)
#   --fix-only          Run only tasks with fixes (skip --all-tasks)
#   --repeat N          Auto-increment prefix and repeat N times
```

### Option 2: Step-by-Step Pipeline

#### Step 1: Merge Traces + Extract Logs

```bash
# Set WANDB API key
export WANDB_API_KEY=your_key_here

# Process all benchmarks
./bin/FINAL_COMMANDS.sh

# Or specific benchmark
./bin/FINAL_COMMANDS.sh scicode
./bin/FINAL_COMMANDS.sh colbench
```

This script:
- Merges individual task traces into per-model files
- Extracts conversation logs from W&B Weave (SciCode, CoreBench, SAB)
- Extracts dialogue history from results directory (ColBench only)

#### Step 2: Run Rubric Evaluation

```bash
./bin/RUBRIC_COMMANDS.sh scicode

# Or manually:
python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y
```

#### Step 3: Aggregate Verdicts

```bash
./bin/JUDGE_COMMANDS.sh scicode

# Or manually:
python scripts/judge.py \
    --pattern "scicode_*" \
    --rubric-dir rubrics_output/scicode \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

#### Step 4: Generate Fixes

```bash
python scripts/claude_fixer_scicode.py \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv \
    --ife-only \
    --tasks-per-batch 5
```

#### Step 5: Apply Fixes and Re-run

```bash
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --all-configs \
    --prefix fixed_ \
    --docker \
    --parallel-tasks 10
```

---

## Benchmark-Specific Commands

### SciCode

```bash
# Run evaluation
./bin/run_all_benchmarks.sh --prefix scicode_sun1_ --benchmarks scicode --parallel-models 5 --parallel-tasks 10 --docker

# Process traces
./bin/FINAL_COMMANDS.sh scicode
./bin/RUBRIC_COMMANDS.sh scicode
./bin/JUDGE_COMMANDS.sh scicode

# Generate and apply fixes
python scripts/claude_fixer_scicode.py --rubric-dir rubrics_output/scicode --judge-csv judge_output/scicode_verdict.csv --ife-only
python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs --prefix fixed_ --docker
```

### CoreBench

```bash
# Decrypt test data first
gpg --output hal-harness/hal/benchmarks/corebench/core_test.json \
    --decrypt hal-harness/hal/benchmarks/corebench/core_test.json.gpg
# Password: reproducibility

# Run evaluation
./bin/run_all_benchmarks.sh --prefix cb_sun1_ --benchmarks corebench --parallel-models 5 --parallel-tasks 10 --docker

# Process traces
./bin/FINAL_COMMANDS.sh corebench
./bin/RUBRIC_COMMANDS.sh corebench
./bin/JUDGE_COMMANDS.sh corebench
```

### ScienceAgentBench

```bash
# Download datasets (required)
cd hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/benchmark/
pip install gdown
gdown 1IYVRVK0TSZXRVKiSc2D0wxV1LXoXhpfA
unzip -P scienceagentbench benchmark.zip
mv benchmark/* . && rmdir benchmark
cd -

# Run evaluation
./bin/run_all_benchmarks.sh --prefix sab_sun1_ --benchmarks scienceagentbench --parallel-models 5 --parallel-tasks 10 --docker

# Process traces
./bin/FINAL_COMMANDS.sh sab
./bin/RUBRIC_COMMANDS.sh sab
./bin/JUDGE_COMMANDS.sh sab
```

### ColBench

ColBench uses a **different workflow** - dialogue history comes from `results/` directory, NOT Weave:

```bash
# Run evaluation
./bin/run_all_benchmarks.sh --prefix col_sun1_ --benchmarks colbench --parallel-models 10 --parallel-tasks 50 --docker

# Process traces (automatically handles dialogues)
./bin/FINAL_COMMANDS.sh colbench
./bin/RUBRIC_COMMANDS.sh colbench
./bin/JUDGE_COMMANDS.sh colbench
```

---

## Directory Structure

```
agent-debug/
├── README.md                    # This file
├── INSTRUCTIONS_NEW_BENCHMARK.md # Guide for adding new benchmarks
├── requirements.txt             # Python dependencies
│
├── bin/                         # Shell scripts
│   ├── run_all_benchmarks.sh    # Main benchmark runner
│   ├── FINAL_COMMANDS.sh        # Trace merge + extraction
│   ├── RUBRIC_COMMANDS.sh       # Rubric evaluation
│   ├── JUDGE_COMMANDS.sh        # Judge aggregation
│   ├── cleanup_docker.sh        # Docker cleanup
│   └── kill_all.sh              # Kill hung processes
│
├── scripts/                     # Python scripts
│   ├── eval_rubric.py           # Rubric evaluation
│   ├── judge.py                 # Verdict aggregation
│   ├── merge_traces.py          # Trace merging
│   ├── extract_weave_traces.py  # Weave extraction
│   ├── add_colbench_dialogues.py # ColBench dialogue extraction
│   ├── claude_fixer_*.py        # Fix generation (per-benchmark)
│   ├── run_benchmark_fixes.py   # Unified fix runner
│   ├── build_response_matrix.py # Results analysis
│   └── find_failed_tasks.py     # Debug utility
│
├── model_configs/               # Model configuration files
│   └── model_to_baseline_*.json
│
├── rubric_templates/            # Rubric prompts for LLM graders
│   ├── scicode.txt
│   ├── corebench.txt
│   ├── scienceagentbench.txt
│   ├── colbench.txt
│   └── rubric.schema.json
│
├── fixes/                       # Generated fix packages
│   ├── scicode/
│   ├── corebench_hard/
│   ├── scienceagentbench/
│   └── colbench_backend_programming/
│
├── traces/                      # Agent execution traces
├── rubrics_output/              # Rubric evaluation results
├── judge_output/                # Aggregated verdicts
│
├── hal-harness/                 # HAL evaluation harness (submodule)
└── docent/                      # Rubric evaluation library (submodule)
```

---

## Configuration Files

### Model Configuration

Each benchmark has a config file in `model_configs/model_to_baseline_<benchmark>.json`:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1-2025-04-14",
    "short_name": "gpt-4.1",
    "max_steps": 5
  },
  "openai/o3-2025-04-16": {
    "model_id": "openai/o3-2025-04-16",
    "short_name": "o3",
    "reasoning_effort": "medium",
    "max_steps": 5
  }
}
```

### Fix Package Structure

Fixes are created in `fixes/<benchmark>/<task_id>/`:

```
fixes/<benchmark>/<task_id>/
├── README.md                   # Human-readable explanation
├── env_override.json           # Environment/dependencies
├── evaluation_override.json    # Criteria adjustments
├── instruction_override.json   # Clarifications
└── status.json                 # Fix metadata
```

---

## CLI Reference

### eval_rubric.py

```bash
python scripts/eval_rubric.py \
    --trace-file traces/*.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only \
    --max-batch-messages 1000 \
    -y
```

| Option | Description |
|--------|-------------|
| `--trace-file` | Path to trace JSON file (can repeat) |
| `--rubric` | Rubric template file |
| `--rubric-model` | Model as provider:model (e.g., `openai:gpt-5.2`) |
| `--failed-only` | Only evaluate failed tasks |
| `--max-batch-messages` | Max messages per batch |
| `-y` | Skip confirmation prompt |

### judge.py

```bash
python scripts/judge.py \
    --pattern "scicode_*" \
    --rubric-dir rubrics_output/scicode \
    --model openai:gpt-5.2 \
    --parallel 10 \
    -y
```

| Option | Description |
|--------|-------------|
| `--pattern` | Pattern to match rubric CSVs |
| `--rubric-dir` | Directory containing rubric CSVs |
| `--model` | Model for aggregation |
| `--parallel` | Number of parallel evaluations |
| `--output` | Output verdict CSV path |

### run_benchmark_fixes.py

```bash
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --all-configs \
    --all-tasks \
    --prefix test_ \
    --docker \
    --parallel-models 5 \
    --parallel-tasks 10
```

| Option | Description |
|--------|-------------|
| `--benchmark` | Benchmark to run |
| `--all-configs` | Run all model configurations |
| `--all-tasks` | Run all tasks (not just those with fixes) |
| `--prefix` | Prefix for run IDs |
| `--docker` | Run in Docker containers |
| `--parallel-models` | Concurrent model configs |
| `--parallel-tasks` | Concurrent tasks per model |
| `--resume` | Resume interrupted run |
| `--no-fix` | Run entire benchmark WITHOUT applying any fixes |
| `--timeout` | Set per-task timeout in seconds (e.g. 7500) |

---

## Timeout Configuration

The pipeline handles timeouts at two levels to balance reliability and completeness:

- **Per-Task Timeout**: Controlled via the `--timeout` flag (default is benchmark-specific, usually 7500s). This limits how long a single agent can spend on one problem. If an agent hangs, the task is killed, but the runner continues to the next task.
- **Run-Level Timeout**: There is **no overall timeout** for the benchmark execution. The scripts (`run_benchmark_fixes.py` and `run_all_benchmarks.sh`) will wait indefinitely for all parallel tasks to complete. This ensures that large-scale runs on 1000+ items can finish successfully without being prematurely terminated.
- **Retry Logic**: If a run fails due to transient API errors (rate limits, timeouts, connection issues), the runner automatically retries up to 3 times with exponential backoff before marking the configuration as failed.

---

## Troubleshooting

### Azure Authentication Issues

```bash
# Re-authenticate
az logout
az login

# Verify token acquisition
az account get-access-token --resource api://trapi/.default
```

### Docker Issues

```bash
# Clean up Docker resources
./bin/cleanup_docker.sh

# Force rebuild Docker images
docker images | grep agent-env | awk '{print $3}' | xargs -r docker rmi -f
docker builder prune -f
```

### Evaluation Hanging

If evaluations hang after "Logged in as Weights & Biases user":

```bash
# Check which .env is being used
cat hal-harness/.env | grep USE_DIRECT_AZURE

# Verify Azure login
az account show

# Kill hung processes
./bin/kill_all.sh
```

### Package Conflicts

```bash
# Recreate environment
conda deactivate
conda env remove -n hal
conda create -n hal python=3.12 -y
conda activate hal
pip install -r requirements.txt
pip install -e ./hal-harness
pip install -e ./docent
```

---

## Testing the Pipeline (Reproducibility)

After installation, verify your setup with these dry-run tests:

### 1. Verify Scripts Load Correctly

```bash
# Test core scripts import
python scripts/merge_traces.py --help
python scripts/eval_rubric.py --help
python scripts/judge.py --help
python scripts/run_benchmark_fixes.py --list-benchmarks
```

### 2. Test Fixer (Dry Run)

```bash
# Preview what the fixer would do without making API calls
python scripts/claude_fixer_scicode.py \
    --rubric-dir rubrics_output/scicode \
    --dry-run \
    --max-tasks 2
```

### 3. Test Runner (Dry Run)

```bash
# Preview what the runner would execute
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --dry-run
```

### 4. Verify Existing Data

```bash
# Check traces exist
ls traces/*.json | head -5

# Check rubric outputs exist
ls rubrics_output/scicode/*.csv | head -5

# Check judge verdicts exist
head -5 judge_output/scicode_verdict.csv

# Check fixes exist
ls fixes/scicode/
```

### 5. Full Pipeline Test (Requires API Access)

To run a minimal end-to-end test:

```bash
# Step 1: Merge existing traces (no API calls)
python scripts/merge_traces.py \
    --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' \
    --output /tmp/test_merge.json \
    --force

# Step 2: Run rubric evaluation (requires API)
python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only \
    --max-tasks 1 \
    -y

# Step 3: Run judge aggregation (requires API)
python scripts/judge.py \
    --pattern "scicode_honey_*" \
    --rubric-dir rubrics_output/scicode \
    --output /tmp/test_verdict.csv \
    --model openai:gpt-5.2 \
    --max-tasks 1 \
    -y

# Step 4: Apply fixes and re-run (requires Docker + API)
python scripts/run_benchmark_fixes.py \
    --benchmark scicode \
    --config gpt-4.1_scicode \
    --docker \
    --prefix test_
```

---

## Key Principles

**Make evaluation FAIR, not EASY.**

- Fix missing dependencies that ALL agents need
- Clarify ambiguous output format requirements
- Adjust overly strict numerical tolerances
- Do NOT give hints about solutions
- Do NOT simplify the problem
- Do NOT pre-compute partial results

---

## Scoring

- **Score 0**: Agent capability issue (a better agent could have succeeded)
- **Score 1**: Benchmark defect / Intrinsic Formation Error (no agent could succeed)

---

## Adding New Benchmarks

See [INSTRUCTIONS_NEW_BENCHMARK.md](INSTRUCTIONS_NEW_BENCHMARK.md) for detailed guide on:
1. Exploring benchmark structure
2. Researching known issues
3. Creating rubric templates
4. Creating fixer scripts
5. Creating fix runners

---

## License

[Add license information]

---

## Citation

If you use this pipeline in your research:

```bibtex
@misc{hal-agent-debug,
  title = {HAL Agent Debug Pipeline},
  author = {Your Name},
  year = {2026},
  howpublished = {\url{https://github.com/YOUR_ORG/agent-debug}}
}
```
