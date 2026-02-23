# Reliable and Efficient Amortized Model-based Evaluation for AI Agents (ICML 2026)

Research code for the paper: "Reliable and Efficient Amortized Model-based Evaluation for AI Agents". This repository implements amortized evaluation methods combining Item Response Theory (IRT), PCA, and Sparse Autoencoders (SAE) to predict AI model performance across benchmarks, alongside an automated item-editor pipeline for benchmark defect remediation.

---

## 1. Environment Setup

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.10+ (for reeval analysis) or 3.11-3.12 (for item-editor pipeline)
- **Docker**: Docker Engine 20.10+ (required for running benchmarks in isolated sandboxes)
- **Conda**: Miniconda or Anaconda
- **Git**: For cloning repository and submodules

### Basic Installation

```bash
# Clone repository with submodules
git clone --recursive https://github.com/aims-foundation/agent-eval.git
cd agent-eval

# If you already cloned without --recursive, initialize submodules:
git submodule update --init --recursive

# Create conda environment
CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -n reeval python=3.10 -y
conda activate reeval

# Install base requirements
pip install -r requirements.txt
```

### Item-Editor Pipeline Setup & Patching

The item-editor pipeline requires additional setup for the `docent` and `hal-harness` submodules.

**CRITICAL**: You must apply the patches to inject the latest fixes and ensure the execution system is in a consistent state.

```bash
# Navigate to item-editor
cd item-editor

# Ensure submodules are initialized (hal-harness and docent)
git submodule update --init --recursive

# Apply patches (Contains critical Docent JSON enhancements and HAL harness fixes)
bash patch/apply_patches.sh
```

Then, set up the virtual environment for the item-editor:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install hal-harness (HAL evaluation framework)
pip install -e ./hal-harness

# Install docent (rubric evaluation library)
pip install -e ./docent/docent/
pip install -e ./docent/

# Return to project root
cd ..
```

### Environment Variables

For the item-editor pipeline, copy the template and fill in your API keys in `item-editor/.env`:

```bash
cp item-editor/.env.template item-editor/.env
```

Ensure the following variables are configured in `.env`:
```env
# API Keys
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
WANDB_API_KEY=your-wandb-key
HF_TOKEN=hf_your-token

# For Azure/TRAPI direct access (recommended)
USE_DIRECT_AZURE=true
TRAPI_ENDPOINT=https://trapi.research.microsoft.com/gcr/shared
TRAPI_API_VERSION=2025-03-01-preview
TRAPI_SCOPE=api://trapi/.default
```

### Azure Authentication (for TRAPI access)

```bash
# Login to Azure
az login

# Verify authentication
az account show
az account get-access-token --resource api://trapi/.default
```

### Optional Dependencies

**LaTeX Installation (Linux)** (For generating plots and figures):
```bash
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra cm-super dvipng fonts-liberation
```

**Jupyter Notebook Widgets (Azure)** (If `tqdm` does not render correctly):
```bash
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

---

## 2. Global Directory Structure

The repository is modularly structured into the following components:

- `helm/`: Amortised model execution on the entire HELM dataset using `embed_meta-llama_Llama-3.1-8B-Instruct`.
- `hal/`: Amortised model execution on ColBench from HAL with `Qwen3-Embedding-8B` alongside Sparse Autoencoder (SAE) interpretations. Contains `pca_aggregate_survey.ipynb` (held out response matrices) and `sae_beta_irt.ipynb` ($N=22$).
- `item-editor/`: **Automated Item-Level Fixing Pipeline**. Detects and fixes Intrinsic Formation Errors (IFEs) in AI agent benchmarks non-destructively.
- `model/`: Configurations for LLM evaluation.
- `data-collection/`: Scripts for downloading existing benchmarks and traces (e.g., `hal.py`).
- `test/`: Backend scripts and testing utilities.

---

## 3. The Item-Editor Pipeline

The item-editor pipeline automates the diagnosis and remediation of Intrinsic Formation Errors (IFEs) without modifying the static benchmark source code. It injects non-destructive patches (environment overrides, instruction headers, evaluation shims) dynamically at runtime.

### Project Structure & Data Layout

Navigate to `item-editor/` to access the pipeline tools. All persistent outputs are centralized in the `result/` directory.

```
item-editor/
├── docent/                     # Docent evaluation library (TransluceAI)
├── hal-harness/                # Core evaluation harness (Princeton PLI)
├── eval_traces/                # Working directory for trace consolidation
├── eval_response_matrix/       # Generated binary result matrices
├── result/                     # Permanent results and metadata
│   ├── fixes/                  # Item-level patches (env, instructions, etc.)
│   └── .hal_data/              # Centralized data store (logs, results, cache)
├── config/                     # Static configurations
│   ├── model/                  # Model-to-baseline benchmark maps
│   └── rubric/                 # Rubric templates (.txt)
└── script/                     # Automation engine
    ├── eval/                   # qualitative evaluation & consensus
    ├── trace/                  # Trace management & merges
    ├── fix/                    # Diagnosis & patching tools
    ├── summary/                # Statistics counting and analysis utilities
    └── utils/                  # System utilities & reporting
```

### Sequential Roadmap (End-to-End)

To run the pipeline from scratch for a benchmark like `scicode`:

```bash
cd item-editor

# 1. Prep: Build Docker execution sandboxes
python script/utils/prebuild_all_images.py scicode

# 2. Baseline: Run the benchmark without fixes
python script/fix/runtime_fixes.py --benchmark scicode --prefix base_ --docker

# 3. Process: Consolidate the distributed traces
python script/trace/collect_upload_traces.py --prefix base_ --output eval_traces
python script/trace/merge_traces.py --input 'eval_traces/traces/*base_*' --output result/.hal_data/traces/merged_base.json

# 4. Grade: Execute Docent rubric evaluations and Meta-Judge consensus
python script/eval/eval_rubric.py --trace-file result/.hal_data/traces/merged_base.json --rubric config/rubric/scicode.txt --failed-only -y
python script/eval/judge.py --pattern "base_*" --rubric-dir result/.hal_data/rubrics_output/scicode --model openai:gpt-4o -y

# 5. Fix: Synthesize latent overlays using Claude 
python script/fix/claude_fixer.py --benchmark scicode --ife-only --judge-csv result/.hal_data/judge_output/scicode_verdict.csv

# 6. Verify: Re-run the benchmark *with* the newly generated fixes
python script/fix/runtime_fixes.py --benchmark scicode --prefix fixed_ --docker --fix-only

# 7. Final: Extract binary evaluation matrices for IRT
python script/utils/build_response_matrix.py --prefix fixed_ --traces-dir eval_traces --extract-subscores
```

### CLI Reference

#### Infrastructure & Maintenance

*   **Prebuild Docker Images**:
    ```bash
    python script/utils/prebuild_all_images.py [benchmarks] --force
    ```
    Builds the necessary agent execution environments for benchmarks (`scicode`, `corebench`, `colbench`, `scienceagentbench`).

*   **Unified Cleanup**:
    ```bash
    python script/utils/cleanup.py --aggressive --images
    ```
    Kills evaluation processes and prunes Docker resources to free up space.

#### Execution Engine

*   **Unified Benchmark Runner**:
    ```bash
    python script/fix/runtime_fixes.py --benchmark scicode --prefix myrun_ --docker
    ```
    *   `--prefix`: Tag for the Run IDs and traces (e.g. `run1_`).
    *   `--fix-only`: Runs only items with existing fixes in `result/fixes/`.
    *   `--no-fix`: Baseline mode. Ignores all generated fixes.
    *   `--parallel-tasks`: Concurrent tasks running per model config (default: 10).

*   **Real-time Log Watcher**:
    ```bash
    python script/utils/watch_all.py --prefix myrun_
    ```
    Tails logs for an active prefix run with color-coding.

#### Diagnosis & Consensus

*   **Rubric Evaluator**:
    ```bash
    python script/eval/eval_rubric.py --trace-file [PATH] --rubric config/rubric/scicode.txt --failed-only
    ```
    Grades agent failure traces. Supports `openai:gpt-4o` and specific `--reasoning-effort` flags for models like `o3-mini`.

*   **Judge Aggregator**:
    ```bash
    python script/eval/judge.py --pattern "run1_*" --rubric-dir result/.hal_data/rubrics_output/scicode 
    ```
    Produces binary IFE verdicts (`0` = Agent Flaw, `1` = Structural Defect) by aggregating multiple model evaluations.

#### Fixing & Reporting

*   **Claude Fixer**:
    ```bash
    python script/fix/claude_fixer.py --benchmark scicode --ife-only 
    ```
    Generates JSON fix packages (`env_override.json`, `instruction_override.json`, etc.) in `result/fixes/`.

*   **Count Fix Statistics**:
    ```bash
    python script/summary/count_stats.py
    ```
    Outputs the aggregated fix counts and verifiable defect proportions.

*   **Response Matrix Generator**:
    ```bash
    python script/utils/build_response_matrix.py --prefix run1_ --extract-subscores
    ```
    Generates the final binary result matrix and detailed metrics needed for Amortized Factor Model IRT analysis.

---

## 4. Latest Updates & Patch Details

The system includes crucial patches that solve execution blockades:
1.  **Smolagents Security Policy**: Allows safe filesystem ops and subprocess execution.
2.  **Scientific Operator Support**: Authorizes `@` (matrix multiplication) in the agent interpreter.
3.  **Docent Structured Outputs**: Implements rigorous JSON format enforcement.

## 5. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `docent not found` | Submodule not installed | Run `pip install -e docent/docent/ && pip install -e docent/` inside `item-editor/` |
| `TRAPI permission denied` | Wrong API version | Set `TRAPI_API_VERSION=2025-03-01-preview` in `.env` |
| `No accounts in MSAL cache` | Azure auth expired | Run `az login` to refresh |
| `rate limit exceeded` | Too many parallel requests | Reduce `--parallel-tasks` value in runner |

## License & Citation

Licensed under MIT.

If you use this pipeline in your research:
```bibtex
@misc{agent-eval,
  title = {Reliable and Efficient Amortized Model-based Evaluation for AI Agents},
  year = {2026},
  howpublished = {\url{https://github.com/aims-foundation/agent-eval}}
}
```
