# reeval-multi

Research code for "Reliable and Efficient Amortized Model-based Evaluation for AI Agents" (ICML 2026). Implements amortized evaluation methods combining Item Response Theory (IRT), PCA, and Sparse Autoencoders (SAE) to predict AI model performance across benchmarks.

## Environment Setup

```bash
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n reeval python=3.10 -y
conda activate reeval
pip install -r requirements.txt

# For item-editor pipeline, also install docent:
git clone https://github.com/TransluceAI/docent.git
pip install -e docent/docent/
pip install -e docent/
```

To install latex-related packages (linux):

```bash
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra cm-super dvipng fonts-liberation
```

If you have problems with jupyter notebook not rendering tqdm correctly on Azure Notebooks:

```bash
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## Download Data

```bash
huggingface-cli download ronanhansel/data-reeval-multi \
    --local-dir ./data-reeval-multi \
    --repo-type dataset
```

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `helm/` | Amortised model on HELM dataset using Llama-3.1-8B-Instruct embeddings |
| `hal/` | Amortised model on ColBench from HAL with Qwen3-Embedding-8B + SAE |
| `item-editor/` | Item-Level Fixing Pipeline for detecting/fixing benchmark defects (IFEs) |
| `data-processing/` | Scripts to convert raw benchmarks into response matrices |

## Environment Variables

```bash
# For SAE feature interpretation (optional)
export OPENAI_KEY_SAE="your-openai-key"

# For item-editor rubric evaluation and judging
export OPENAI_API_KEY="your-openai-key"

# For Azure/TRAPI (Microsoft internal)
export TRAPI_ENDPOINT="https://trapi.research.microsoft.com/gcr/shared"
export TRAPI_API_VERSION="2025-03-01-preview"

# For Claude fixer (item-editor)
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Weave trace extraction (optional)
export WANDB_API_KEY="your-wandb-key"
```

---

## HAL Pipeline

The HAL directory implements the amortized IRT evaluation for ColBench:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1. EMBEDDING GENERATION                          │
│                    (generate_embeddings.py)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Questions ──► Qwen3-Embedding-8B ──► 4096-dim raw embeddings      │
│                        │                                            │
│              ┌────────┴────────┐                                   │
│              ▼                 ▼                                   │
│        PCA (48-dim)      SAE (48-dim, K=4)                         │
│              │                 │                                   │
│              └────────┬────────┘                                   │
│                       ▼                                            │
│               Push to HuggingFace                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    2. AMORTIZED IRT MODEL                           │
│                      (amortized_irt.py)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Inputs:                                                            │
│    • Response matrices (models × questions)                        │
│    • Pre-computed embeddings (raw / PCA / SAE)                     │
│                                                                     │
│  Models Compared:                                                   │
│    • Global Mean    - predict mean response (baseline)             │
│    • Rasch-IRT      - classic IRT: ability - difficulty            │
│    • Amortized IRT  - embedding-based item loadings (ours)         │
│                                                                     │
│  Output: result/amortized_irt_{embedding_type}.csv                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       3. PLOTTING                                   │
│                       (plotting.py)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: result/amortized_irt_*.csv                                 │
│                                                                     │
│  Output PDFs:                                                       │
│    • rmse_comparison_{type}.pdf   - bar (n=1 vs n=max)            │
│    • auc_comparison_{type}.pdf    - bar (n=1 vs n=max)            │
│    • rmse_convergence_{type}.pdf  - line (RMSE vs n)              │
│    • auc_convergence_{type}.pdf   - line (AUC vs n)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
cd hal

# 1. Generate embeddings (downloads raw from HuggingFace if missing)
python generate_embeddings.py                      # PCA + SAE from existing raw embeddings
python generate_embeddings.py --from-text          # Full pipeline: text → raw → PCA → SAE
python generate_embeddings.py --interpret          # Also interpret SAE features with GPT-4o
python generate_embeddings.py --push-to-hf         # Upload to HuggingFace

# Custom dimensions
python generate_embeddings.py --pca-dim 64 --sae-features 64 --sae-k 4

# 2. Run amortized IRT experiment
python amortized_irt.py --embedding-type pca                # full (n=1..22)
python amortized_irt.py --embedding-type pca --n-samples 22 # quick test

# 3. Generate plots
python plotting.py
```

### Key Files

| File | Purpose |
|------|---------|
| `generate_embeddings.py` | Generate raw (Qwen3), PCA, SAE embeddings; optional interpretation; push to HF |
| `amortized_irt.py` | Train IRT models, evaluate RMSE/AUC → CSV |
| `plotting.py` | Load CSV → generate PDF figures |
| `utils.py` | Shared utilities (compute_rmse, evaluate_auc) |

### Embedding Types

| Type | Dim | Description |
|------|-----|-------------|
| `raw` | 4096 | Direct Qwen3-Embedding-8B output |
| `pca` | 48 | PCA reduction |
| `sae` | 48 | Sparse Autoencoder (K=4 sparsity) |

---

## Item Editor Pipeline

The item-editor pipeline automatically detects and fixes **Intrinsic Formation Errors (IFEs)** in AI agent benchmarks without modifying the benchmark source code.

```
Traces → Rubric Evaluation → Verdict Aggregation → Fix Generation → Fix Application
```

### Directory Structure

```
item-editor/
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

### Quick Start

```bash
cd item-editor

# Create output directories
mkdir -p rubrics_output/scicode judge_output fixes/scicode

# 1. Run rubric evaluation
python scripts/eval_rubric.py \
    --trace-file ../data-reeval-multi/traces/scicode_*.json \
    --rubric prompts/scicode.txt \
    --output-dir rubrics_output/scicode \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y

# 2. Aggregate verdicts
python scripts/judge.py \
    --pattern "*.csv" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 -y

# 3. Generate fixes (requires Claude API key)
python scripts/claude_fixer_scicode.py \
    --all-ife \
    --rubric-dir rubrics_output/scicode \
    --judge-csv judge_output/scicode_verdict.csv

# 4. Apply fixes (requires HAL harness)
python scripts/run_scicode_fixes.py --all --prefix honey_ --docker
```

### Step 1: Rubric Evaluation

Evaluate failed tasks against benchmark-specific rubrics.

```bash
python scripts/eval_rubric.py \
    --trace-file ../data-reeval-multi/traces/scicode_*.json \
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

**Output:** CSV files with columns: `task_id`, `criteria`, `grade` (0=no IFE, 1=IFE), `explanation`, `model_run`

### Step 2: Verdict Aggregation

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

**Output:** Verdict CSV with columns: `task_id`, `final_grade`, `satisfies_rubric`, `num_evaluations`, `model_runs`, `reasoning`

### Step 3: Fix Generation

Generate fix packages for IFE tasks using Claude.

```bash
# List IFE tasks (Grade=1)
python scripts/claude_fixer_scicode.py --list-ife-tasks

# Fix a specific task
python scripts/claude_fixer_scicode.py --task-id 12

# Fix all IFE tasks
python scripts/claude_fixer_scicode.py --all-ife --skip-existing

# Batch mode (more efficient)
python scripts/claude_fixer_scicode.py --all-ife --batch --tasks-per-batch 3
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
├── env_override.json           # Environment fixes
├── evaluation_override.json    # Evaluation criteria adjustments
├── instruction_override.json   # Task clarifications
└── status.json                 # Application status
```

### Step 4: Fix Application

Apply fixes and re-run HAL evaluations.

```bash
# List available fixes
python scripts/run_scicode_fixes.py --list-fixes

# Dry run (preview)
python scripts/run_scicode_fixes.py --task-id 12 --prefix fixed_ --dry-run

# Run fixes for specific tasks
python scripts/run_scicode_fixes.py --task-id 12 --task-id 35 --prefix iter1_ --docker

# Run all available fixes
python scripts/run_scicode_fixes.py --all --prefix honey_ --docker --parallel 3
```

**Available Fix Runners:**

| Benchmark | Script |
|-----------|--------|
| SciCode | `run_scicode_fixes.py` |
| CoreBench | `run_corebench_fixes.py` |
| ScienceAgentBench | `run_scienceagentbench_fixes.py` |
| ColBench | `run_colbench_fixes.py` |
| USACO | `run_usaco_fixes.py` |

### Step 5: Before/After Comparison

Re-run rubric evaluation on fixed traces to measure improvement.

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

### Trace Utilities

```bash
# Merge individual traces
python scripts/merge_traces.py \
    --pattern "scicode_honey__*.json" \
    --trace-dir ../data-reeval-multi/traces \
    --output traces/scicode_honey_MERGED_UPLOAD.json

# Extract Weave logs
python scripts/extract_weave_traces.py \
    --trace-file traces/scicode_honey_MERGED_UPLOAD.json \
    --run-prefix "scicode_honey_openai_gpt-4_1"

# Extract ColBench dialogues
python scripts/add_colbench_dialogues.py \
    traces/col_ivy_MERGED_UPLOAD.json \
    --results-dir results/colbench_backend_programming \
    --output traces/col_ivy_WITH_DIALOGUES.json
```

### Supported Benchmarks

| Benchmark | Rubric | Fixer | Fix Runner |
|-----------|--------|-------|------------|
| SciCode | scicode.txt | claude_fixer_scicode.py | run_scicode_fixes.py |
| CoreBench | corebench.txt | claude_fixer_corebench.py | run_corebench_fixes.py |
| ScienceAgentBench | scienceagentbench.txt | claude_fixer_scienceagentbench.py | run_scienceagentbench_fixes.py |
| ColBench | colbench.txt | claude_fixer_colbench.py | run_colbench_fixes.py |
| SWE-bench | swebench.txt | - | - |
| USACO | usaco.txt | - | run_usaco_fixes.py |
| AssistantBench | assistantbench.txt | - | - |

### Rubric Grading

- **Grade 0**: No IFE - agent capability issue (agent could have succeeded)
- **Grade 1**: IFE detected - benchmark defect (no agent could succeed)

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `docent not found` | Missing docent installation | `pip install -e docent/docent/ && pip install -e docent/` |
| `TRAPI permission denied` | Wrong API version | Set `TRAPI_API_VERSION=2025-03-01-preview` |
| `No accounts in MSAL cache` | Azure auth expired | Run `az login` to refresh |
| `rate limit exceeded` | Too many parallel requests | Reduce `--parallel` value |

## References

- [HAL Harness](https://github.com/princeton-pli/hal-harness) - Princeton PLI's Holistic Agent Leaderboard evaluation framework
- [Docent](https://github.com/TransluceAI/docent) - TransluceAI's agent analysis platform for rubric-based evaluation
- [HAL Leaderboard](https://hal.cs.princeton.edu/) - Official HAL leaderboard website
