# HAL Agent Debug Pipeline

Automated Item Fixing Pipeline for benchmark evaluation. The core innovation is **item-level fixing** - modifying agent configurations, prompt templates, and runtime parameters to eliminate false-positive benchmark defects without changing the benchmark source code itself.

---

## 1. Project Structure & Data Layout

The project is organized into a functional hierarchy. All persistent data and outputs are centralized to prevent workspace clutter.

```
item-editor/
├── docent/                     # Docent evaluation library
├── hal-harness/                # Core evaluation harness
├── eval_traces/                # Working directory for trace consolidation
├── eval_response_matrix/       # Generated result matrices
├── result/                     # Permanent results and metadata
│   ├── fixes/                  # Item-level patches (env, instructions, etc.)
│   └── .hal_data/              # Centralized data store (logs, results, cache)
├── config/                     # Static configurations
│   ├── model/                  # Model-to-baseline benchmark maps
│   └── rubric/                 # Rubric templates (.txt)
└── script/                     # Reorganized automation engine
    ├── eval/                   # qualitative evaluation & consensus
    ├── trace/                  # Trace management & merges
    ├── fix/                    # Diagnosis & patching tools
    └── utils/                  # System utilities & reporting
```

---

## 2. Comprehensive CLI Reference

### A. Infrastructure & Maintenance

#### Prebuild Docker Images
Ensures all required environments and datasets are ready.
```bash
python script/utils/prebuild_all_images.py [benchmarks] --force
```
*   **`benchmarks`** (Positional, Variadic):
    *   **Accepted**: Space-separated benchmark names (e.g., `scicode corebench colbench scienceagentbench`).
    *   **Default**: All known benchmarks found in `config/model/`.
*   **`--force`**:
    *   **Type**: Flag (No argument).
    *   **Effect**: Forces rebuild of images even if they already exist in the local Docker daemon.

#### Unified Cleanup
Kills evaluation processes and prunes Docker resources.
```bash
python script/utils/cleanup.py --aggressive --images
```
*   **`--aggressive`**:
    *   **Type**: Flag.
    *   **Effect**: Kills and removes **ALL** containers on the system. If unset, only benchmark-specific containers are targeted.
*   **`--images`**:
    *   **Type**: Flag.
    *   **Effect**: Removes ephemeral `agent-env-*` images and prunes the Docker build cache.
*   **`--only-docker`**:
    *   **Type**: Flag.
    *   **Effect**: Performs Docker cleanup only; local processes are left running.
*   **`--only-processes`**:
    *   **Type**: Flag.
    *   **Effect**: Kills local Python/Shell processes only; Docker state is preserved.

---

### B. Execution Engine

#### Unified Benchmark Runner
The main engine for executing agent evaluations.
```bash
python script/fix/runtime_fixes.py --benchmark scicode --prefix myrun_ --docker
```
*   **`--benchmark, -b`**:
    *   **Accepted**: Single benchmark name (e.g., `scicode`).
*   **`--benchmarks`**:
    *   **Accepted**: Comma-separated list (e.g., `scicode,corebench`).
*   **`--config, -c`**:
    *   **Accepted**: Config keys from `config/model/` (can repeat).
    *   **Default**: Runs all configurations in the benchmark's JSON file.
*   **`--prefix`**:
    *   **Accepted**: String (e.g., `run1_`).
    *   **Default**: `moon1_`. Used for tagging Run IDs and traces.
*   **`--parallel-tasks`**:
    *   **Accepted**: Integer.
    *   **Default**: `10`. Concurrent tasks running per model config.
*   **`--docker`**:
    *   **Type**: Flag.
    *   **Effect**: Mandatory for most runs. Isolates agent execution in containers.
*   **`--fix-only`**:
    *   **Type**: Flag.
    *   **Effect**: Only runs items that have an existing fix directory in `result/fixes/`.
*   **`--no-fix`**:
    *   **Type**: Flag.
    *   **Effect**: Baseline mode. Ignores all generated fixes and runs original benchmark items.
*   **`--until`**:
    *   **Accepted**: Integer (N).
    *   **Effect**: Loop mode. Increments the prefix number and repeats until N is reached.

#### Real-time Watcher
Tails logs for an active prefix run with color-coding.
```bash
python script/utils/watch_all.py --prefix run1_
```
*   **`--prefix`**:
    *   **Accepted**: String. (Required). Automatically locates logs in `result/.hal_data/logs`.

---

### C. Trace Consolidation

#### Trace Collector
Gathers distributed evaluation artifacts into a central working directory.
```bash
python script/trace/collect_upload_traces.py --prefix run1_ --output eval_traces --benchmark scicode
```
*   **`--prefix`**:
    *   **Accepted**: Regex or String. (Required).
*   **`--output`**:
    *   **Accepted**: Directory path. (Required). Recommended: `eval_traces`.
*   **`--benchmark`**:
    *   **Accepted**: Benchmark name (can repeat).
*   **`--run-root`**:
    *   **Accepted**: Path. Override for custom data roots.

#### Merge Traces
Consolidates task-level traces into a single agent-level trace file.
```bash
python script/trace/merge_traces.py --input 'eval_traces/traces/*run1_*' --output result/.hal_data/traces/merged_run1.json --force
```
*   **`--input`**:
    *   **Accepted**: Glob pattern (can repeat). (Required).
*   **`--output`**:
    *   **Accepted**: File path. (Required).
*   **`--run-id`**:
    *   **Accepted**: String. Overrides the `run_id` field in the resulting JSON.
*   **`--force`**:
    *   **Type**: Flag. Overwrites output if it exists.

---

### D. Diagnosis & Consensus

#### Rubric Evaluator
Grades agent failure traces using LLM rubrics.
```bash
python script/eval/eval_rubric.py --trace-file [PATH] --rubric config/rubric/scicode.txt --failed-only -y
```
*   **`--trace-file`**:
    *   **Accepted**: Path to trace JSON (can repeat).
*   **`--rubric`**:
    *   **Accepted**: Path to `.txt` template.
*   **`--rubric-model`**:
    *   **Accepted**: `provider:model` (e.g., `openai:gpt-4o`).
    *   **Default**: `gpt-4o` (from config).
*   **`--failed-only`**:
    *   **Type**: Flag. Only processes items marked as failures in the trace.
*   **`--max-batch-messages`**:
    *   **Accepted**: Integer.
    *   **Default**: `1000`. Dynamically groups tasks into batches based on message count.
*   **`--no-cache`**:
    *   **Type**: Flag. Bypasses SQLite cache and forces fresh API calls.

#### Judge Aggregator
Produces binary IFE verdicts by aggregating multiple model evaluations.
```bash
python script/eval/judge.py --pattern "run1_*" --rubric-dir result/.hal_data/rubrics_output/scicode --model openai:gpt-4o -y
```
*   **`--pattern`**:
    *   **Accepted**: Glob pattern for rubric CSVs.
*   **`--rubric-dir`**:
    *   **Accepted**: Path. (Required).
*   **`--model`**:
    *   **Accepted**: Judge model ID (Required).
*   **`--priority-override`**:
    *   **Accepted**: Two strings `[HIGH_PFX] [LOW_PFX]`.
    *   **Effect**: If a task has an eval from `HIGH_PFX`, discard any from `LOW_PFX`.
*   **`--common-only`**:
    *   **Type**: Flag. Only judges tasks that appear in every CSV matched by the pattern.

---

### E. Fixing & Reporting

#### Claude Fixer
Diagnoses and generates fix packages (`env`, `instruction`, `evaluation` overrides).
```bash
python script/fix/claude_fixer.py --benchmark scicode --ife-only --judge-csv result/.hal_data/judge_output/scicode_verdict.csv
```
*   **`--benchmark, -b`**:
    *   **Accepted**: `scicode`, `corebench`, `colbench`, `scienceagentbench`. (Required).
*   **`--ife-only`**:
    *   **Type**: Flag. Only processes tasks where the judge verdict was `1` (IFE confirmed).
*   **`--min-grade`**:
    *   **Accepted**: Float.
    *   **Default**: `0.5`. Minimum rubric grade to consider for fixing if judge CSV is missing.
*   **`--tasks-per-batch`**:
    *   **Accepted**: Integer.
    *   **Default**: `5`. Number of tasks processed in a single Claude Code session.

#### Response Matrix Generator
Generates the final binary result matrix and detailed metrics.
```bash
python script/utils/build_response_matrix.py --prefix run1_ --traces-dir eval_traces --extract-subscores
```
*   **`--prefix`**:
    *   **Accepted**: String/Regex. (Required).
*   **`--extract-subscores`**:
    *   **Type**: Flag. Saves detailed metrics (success_rate, codebert, etc.) into separate CSVs.
*   **`--reeval`**:
    *   **Type**: Flag. Re-runs the benchmark evaluation harness on raw agent submissions.

---

## 3. Sequential Roadmap (End-to-End)

1.  **Prep**: `python script/utils/prebuild_all_images.py`
2.  **Baseline**: `python script/fix/runtime_fixes.py --prefix base_ --docker`
3.  **Process**: `python script/trace/collect_upload_traces.py --prefix base_ --output eval_traces`
4.  **Grade**: `python script/eval/eval_rubric.py` then `python script/eval/judge.py`
5.  **Fix**: `python script/fix/claude_fixer.py --benchmark scicode --ife-only`
6.  **Verify**: `python script/fix/runtime_fixes.py --prefix fixed_ --docker --fix-only`
7.  **Final**: `python script/utils/build_response_matrix.py --prefix fixed_`

---

## 4. Latest Updates & Patch Summary

The system has been updated with several critical improvements to stability, performance, and agent reliability. These changes are delivered via sub-module patches.

### A. Functional Changes Summary

#### 1. Infrastructure & Execution
*   **Sequential Configuration Runner**: Refactored the unified benchmark runner to process model configurations sequentially while maintaining parallel task execution, ensuring better stability and easier debugging.
*   **Enhanced Docker Environment**: Updated the base `Dockerfile` with missing scientific and utility packages (`PyPDF2`, `xgboost`, `ddgs`, `r-rmarkdown`, `pathlib`) to ensure baseline compatibility for all benchmarks.
*   **Native Fix Application Support**: Patched the HAL harness to support loading external fix-enriched datasets, enabling the automated application of `env`, `instruction`, and `evaluation` overrides.

#### 2. Agent Runtime & Security
*   **Smolagents Security Policy**: Patched the `smolagents` executor to authorize safe file system operations (`posixpath`, `glob`) and process management (`subprocess`), which are required for complex engineering tasks.
*   **Scientific Operator Support**: Added support for the `@` matrix multiplication operator in the agent's Python interpreter, resolving failures in numerical tasks.

#### 3. Evaluation & Metrics
*   **Structured Output Support**: Added `response_format` support to the `docent` evaluation library, enabling reliable JSON extraction across all supported LLM providers.
*   **Robust Rubric Grading**: Improved the rubric evaluation pipeline with enhanced JSON parsing and malformed output recovery.

### B. Patch File Contents
The updated system logic is maintained via two comprehensive patch files which **must be applied** after initializing submodules:
*   `patch/docent/docent.patch`: Contains Docent library enhancements.
*   `patch/hal-harness/hal-harness.patch`: Contains core harness, agent runner, and environment updates.

**Note**: A minor fix was also applied directly to `script/utils/prebuild_all_images.py` in the root repository to resolve a missing import (`NameError: name 'List' is not defined`).

---

## 5. Step-by-Step Execution Guide (Fresh Install)

Follow these steps to set up and run the evaluation pipeline from a newly cloned repository.

### Step 1: Clone and Initialize
```bash
# Clone the repository
git clone <repo_url> item-editor
cd item-editor

# Initialize submodules (docent and hal-harness)
git submodule update --init --recursive
```

### Step 2: Apply Patches
**CRITICAL**: You must apply the patches to inject the latest fixes and ensure the system is in a consistent state.
```bash
# Run the provided patch script
bash patch/apply_patches.sh
```
*Verify output*: Ensure you see "Successfully applied hal-harness patch" and "Successfully applied docent patch".

### Step 3: Environment Setup
1.  **Create Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables**:
    Copy the template and fill in your API keys.
    ```bash
    cp .env.template .env
    # Edit .env and add your OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    ```

### Step 4: Prebuild Docker Images
Build the agent execution environments.
```bash
python script/utils/prebuild_all_images.py --force
```

### Step 5: Run Evaluation
Execute a benchmark run. This example runs `scicode` with Docker isolation.
```bash
python script/fix/runtime_fixes.py \
  --benchmark scicode \
  --prefix test_run_ \
  --docker
```

### Step 6: Monitor and Analyze
*   **Watch Logs**:
    ```bash
    python script/utils/watch_all.py --prefix test_run_
    ```
*   **Check Results**:
    Results will be generated in `result/.hal_data/results`.

---

## License & Citation

[Add license info]

If you use this pipeline in your research:
```bibtex
@misc{agent-eval,
  title = {Agent Eval},
  year = {2026},
  howpublished = {\url{https://github.com/aims-foundation/agent-eval}}
}
```
