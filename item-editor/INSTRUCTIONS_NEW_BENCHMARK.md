# Instructions: Adding a New Benchmark to the Item Fixing Pipeline

This document provides comprehensive step-by-step instructions for Claude Code to set up a new benchmark for the Item Fixing Pipeline, including rubric templates, Claude fixer scripts, fix runners, and model configurations.

---

## Overview

When adding a new benchmark, you need to create:
1. **Rubric template** - For LLM-based IFE detection (`rubric_templates/<benchmark>.txt`)
2. **Claude fixer script** - For automated fix generation (`scripts/claude_fixer_<benchmark>.py`)
3. **Model configuration** - Maps models to baseline traces (`model_configs/model_to_baseline_<benchmark>.json`)
4. **Fixes directory** - `fixes/<benchmark>/` to store generated fixes
5. **Documentation updates** - README.md entries

**Note**: The unified `run_benchmark_fixes.py` handles all benchmarks - no separate runner script needed.

---

## PHASE 1: Deep Understanding of the Benchmark

### Step 1.1: Explore the Benchmark Structure

**First, understand the complete harness structure:**

```bash
# Find the benchmark implementation
cat hal-harness/hal/benchmarks/<benchmark_name>.py

# Find the agent implementation
ls -la hal-harness/agents/ | grep -i <benchmark_name>
cat hal-harness/agents/<agent_dir>/main.py

# Check requirements and dependencies
cat hal-harness/agents/<agent_dir>/requirements.txt

# Check if there's a submodule with additional code
ls -la hal-harness/hal/benchmarks/<benchmark_name>/
```

**Answer these critical questions:**

| Question | Why It Matters |
|----------|---------------|
| What is the agent's entry point? | Needed for `--agent_function` in hal-eval |
| How does the agent framework work? | smolagents, LangChain, custom? Affects sandbox behavior |
| Does it require Docker? | Affects how files are copied and execution environment |
| How is success/failure determined? | Metrics, test cases, GPT-4 judge, exact match? |
| What is the evaluation flow? | Does it run agent code in sandbox or extract code? |
| What are the task data structures? | Keys like `task_inst`, `problem_statement`, etc. |
| What packages do gold programs use? | Must be in requirements.txt |

### Step 1.2: Understand the Evaluation Flow

**This is CRITICAL - different benchmarks evaluate differently:**

```python
# Example: ScienceAgentBench evaluation flow
# 1. Agent runs in Docker container with smolagents CodeAgent
# 2. Agent generates code wrapped in ```python blocks
# 3. recover_pred_from_log.py EXTRACTS code via regex: r"```python(.*?)```"
# 4. Extracted code runs in SEPARATE Docker container for evaluation
# 5. Results compared to gold program output

# Key insight: Sandbox "Forbidden function" errors don't matter!
# The agent just needs to OUTPUT valid code, not EXECUTE it successfully
```

**Read the evaluation code to understand:**
```bash
# Check how results are processed
grep -r "evaluate\|eval_" hal-harness/hal/benchmarks/<benchmark>*.py
cat hal-harness/hal/benchmarks/<benchmark_name>/<benchmark_name>/recover_pred_from_log.py
cat hal-harness/hal/benchmarks/<benchmark_name>/<benchmark_name>/run_eval.py
```

### Step 1.3: Check Docker/Container Setup

**Understand the Docker runner behavior:**

```bash
# Check docker_runner.py for:
# - Working directory setup (/workspace vs /workspace/environment)
# - File copying destinations
# - Environment variable handling
# - Prepared image caching (requirements hash)

grep -A20 "environment" hal-harness/hal/utils/docker_runner.py
grep -A10 "_requirements_hash" hal-harness/hal/utils/docker_runner.py
```

**Key Docker insights from ScienceAgentBench debugging:**
- Prepared images are cached based on hash of: `requirements.txt + base_image_id + template_version`
- Files copied via `benchmark['files']` dict go to specified destination
- Working directory may differ from file destination (e.g., `/workspace/environment/` vs `/workspace/`)
- `run_agent.py` may `chdir()` to a subdirectory before running agent

### Step 1.4: Analyze Gold Programs for Dependencies

**Find ALL packages used by gold/reference programs:**

```bash
# Extract imports from gold programs
grep -rh "^import\|^from" hal-harness/hal/benchmarks/<benchmark>/<submodule>/benchmark/gold_programs/*.py \
    | sed 's/import /\nimport /g; s/from /\nfrom /g' \
    | grep -E "^import|^from" \
    | sed 's/import \([a-zA-Z0-9_]*\).*/\1/; s/from \([a-zA-Z0-9_]*\).*/\1/' \
    | sort | uniq -c | sort -rn | head -40
```

**Compare with requirements.txt:**
```bash
# Check what's already in requirements
cat hal-harness/agents/<agent_dir>/requirements.txt

# Check what's in AUTHORIZED_IMPORTS (for smolagents)
grep -A200 "AUTHORIZED_IMPORTS" hal-harness/agents/<agent_dir>/main.py | head -250
```

---

## PHASE 2: Research Known Issues

### Step 2.1: Search for Research-Documented Issues

Use WebSearch to find:
- Academic papers analyzing the benchmark
- GitHub issues about broken/flaky tasks
- Community discussions about evaluation problems
- Leaderboard submissions with notes

**Search queries to try:**
- `"<benchmark_name>" benchmark issues limitations`
- `"<benchmark_name>" evaluation problems`
- `"<benchmark_name>" github issues`
- `"<benchmark_name>" paper analysis`

### Step 2.2: Analyze Existing Traces (If Available)

```bash
# List available traces
ls -la traces/<benchmark>*.json

# Extract failure patterns
python -c "
import json
from pathlib import Path
from collections import Counter

errors = Counter()
for f in Path('traces').glob('<benchmark>*.json'):
    data = json.loads(f.read_text())
    for result in data.get('raw_logging_results', []):
        if 'error' in str(result).lower():
            # Extract error type
            pass
print(errors.most_common(20))
"
```

### Step 2.3: Cross-Model Failure Analysis

**Key indicator of IFE: Same task fails across ALL models**

```bash
# Count tasks that fail across all models
# This suggests benchmark issue, not agent capability
```

---

## PHASE 3: Create the Rubric Template

Create `rubric_templates/<benchmark_name>.txt` with this comprehensive structure:

```markdown
# <Benchmark Name> Intrinsic Formation Error Detection Rubric

## Purpose
This rubric evaluates whether a task failure on the <Benchmark Name> benchmark stems from an **Intrinsic Formation Error (IFE)** in the benchmark itself, rather than agent capability limitations.

## Scoring
- **Score 1**: An intrinsic benchmark defect exists AND caused the failure (no agent could succeed)
- **Score 0**: The failure is due to agent capability issues (a better agent could have succeeded)

## Two-Question Framework

### Question 1: Does an Intrinsic Formation Defect Exist?
Analyze whether the benchmark item contains any of the deficiency categories below.

### Question 2: Did the Defect Cause This Failure?
Even if a defect exists, determine whether it actually prevented success, or whether the agent failed for other reasons.

---

## Deficiency Categories for <Benchmark Name>

### 1. Environment/Dependency Issues
Missing or misconfigured packages that block ALL agents:
- Required package not installed in Docker container
- Package version conflicts causing import failures
- Missing system libraries (C headers, CUDA, etc.)
- Conda environment not properly activated

### 2. Data/Input Issues
Problems with task data that prevent any solution:
- Data files missing or corrupted
- Data format doesn't match documentation
- File paths incorrect or inaccessible
- Encoding issues preventing file reading

### 3. Task Specification Issues
Ambiguous or impossible requirements:
- Output format requirements unclear or contradictory
- Task requires information not provided
- Conflicting instructions between task and evaluation
- Missing domain knowledge critical for solution

### 4. Evaluation Issues
Problems with how success is determined:
- Evaluation script has bugs
- Numerical tolerance too strict
- GPT-4 judge penalizes valid alternatives
- Format-sensitive comparison rejects correct answers

### 5. Gold Program/Reference Issues
Problems with the reference implementation:
- Gold program uses unavailable libraries
- Gold program has hardcoded paths
- Multiple valid approaches rejected
- Gold program doesn't match task requirements

[Add benchmark-specific categories based on research]

---

## CRITICAL EXCLUSIONS: Agent Capability Issues (Score 0)

Do NOT classify the following as benchmark deficiencies:

### 1. Algorithm/Logic Errors
- Agent chose wrong approach
- Agent made calculation mistakes
- Agent misunderstood the problem
- These are capability issues, not benchmark defects

### 2. Sandbox Limitations (If Code is Extracted)
- "Forbidden function" errors in smolagents sandbox
- Import restrictions during agent execution
- **IF the benchmark extracts code via regex and runs separately, sandbox errors don't matter**
- Check the evaluation flow before classifying these as IFEs

### 3. API/Network Errors
- Rate limiting from LLM provider
- Timeout errors from API calls
- These are infrastructure issues, not benchmark defects

### 4. Resource Exhaustion
- Agent ran out of steps
- Token limit exceeded
- These indicate agent inefficiency, not benchmark problems

[Add benchmark-specific exclusions]

---

## Evidence Requirements

For Score 1, you MUST provide:
1. **Specific defect identification**: Quote the exact issue
2. **Impossibility proof**: Explain why NO agent could overcome this
3. **Exclusion of alternatives**: Show no workaround exists

For Score 0, explain:
1. What the agent did wrong
2. What a capable agent could have done instead

---

## Response Format

Respond with a JSON object:

```json
{
  "score": 0 or 1,
  "deficiency_exists": true/false,
  "deficiency_caused_failure": true/false,
  "deficiency_type": "category name or 'none'",
  "existence_reasoning": "analysis of whether a benchmark defect exists",
  "causation_reasoning": "analysis of whether the defect caused failure",
  "evidence": "specific quotes from transcript supporting conclusion"
}
```

---

## Cross-Run Analysis Guidelines

When evaluating IFEs, use evidence from multiple model runs:

### Strong IFE Indicators (Lean toward Score 1):
1. **Same error across ALL models** - Universal failure suggests benchmark issue
2. **Valid output rejected** - Evaluation rejects correct answer
3. **Environment blocks ALL models identically** - Setup issue affects everyone
4. **Impossible requirements** - No amount of capability could satisfy

### Weak IFE Indicators (Lean toward Score 0):
1. **Only one model fails** - Other models succeeded
2. **Different error types** - Not systematic
3. **Partial success** - Model got partway, could have done better
4. **Known solvable** - Other submissions succeeded on this task

---

## Exploratory Analysis: Discovering Hidden Benchmark Issues

### Areas to Investigate (Suggestions, Not Rules)

**1. Evaluation Fairness**
Consider whether the evaluation method unfairly penalizes valid solutions:
- Does GPT-4 judge have biases about formatting/style?
- Are numerical tolerances reasonable for the domain?
- Do alternative correct approaches get rejected?

**2. Environment Completeness**
Examine whether the execution environment is properly configured:
- Are all packages from gold programs available?
- Are system dependencies (C libraries, tools) present?
- Does the working directory match file locations?

**3. Task Clarity**
Assess whether task requirements are unambiguous:
- Are output format requirements explicit?
- Is domain knowledge properly included?
- Do instructions match evaluation criteria?

### Questions to Ask Yourself

1. **"If I were a perfect agent, could I solve this?"**
2. **"Is the evaluation measuring what the task asks for?"**
3. **"Would adding packages/fixing environment allow success?"**
4. **"Is this a fair test of agent capability?"**

### Known Problematic Task Patterns

[Add specific task IDs and patterns discovered during trace analysis]

---

## <Benchmark Name>-Specific Evaluation Notes

[Add notes about how this specific benchmark evaluates]

Example for ScienceAgentBench:
- Agent generates code in ```python blocks
- Code is EXTRACTED via regex, not executed in sandbox
- Evaluation runs extracted code in separate Docker container
- GPT-4 judges figure similarity (subjective)
- "Forbidden function" sandbox errors are IRRELEVANT to final score
```

---

## PHASE 4: Create the Claude Fixer Script

Create `scripts/claude_fixer_<benchmark_name>.py` based on existing templates.

**Key components to customize:**

### 4.1: Update Constants

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = REPO_ROOT / "traces"
FIXES_DIR = REPO_ROOT / "fixes"
BENCHMARK = "<benchmark_name>"
```

### 4.2: Adapt Trace Loading

Different benchmarks store task data differently:

```python
def load_task_conversations(trace_files: List[str], task_ids: Set[str]) -> Dict[str, str]:
    """Extract agent conversations from trace files.

    CUSTOMIZE THIS for your benchmark's trace structure.
    """
    conversations = {}

    for trace_file in trace_files:
        data = json.loads(Path(trace_file).read_text())

        # ScienceAgentBench format:
        for result in data.get("raw_logging_results", []):
            task_id = str(result.get("inputs", {}).get("task_id", ""))
            # Extract conversation...

        # SciCode format:
        for result in data.get("raw_logging_results", []):
            task_id = result.get("inputs", {}).get("task_id", "")
            # Extract conversation...

        # Other benchmarks may have different structures

    return conversations
```

### 4.3: Customize the Claude Prompt

The prompt should include:
1. Benchmark-specific harness files to read
2. Evaluation flow description
3. Known IFE patterns from trace analysis
4. Thorough error analysis checklist
5. Fix format appropriate for the benchmark

**See template in existing INSTRUCTIONS_NEW_BENCHMARK.md Step 5**

---

## PHASE 5: Create Model Configuration

Create `model_configs/model_to_baseline_<benchmark_name>.json`:

```json
{
  "openai/gpt-4.1-2025-04-14": {
    "model_id": "openai/gpt-4.1-2025-04-14",
    "short_name": "gpt-4.1",
    "baseline_trace": "<benchmark>_gpt41_UPLOAD.json",
    "max_steps": 5
  },
  "openai/o3-2025-04-16": {
    "model_id": "openai/o3-2025-04-16",
    "short_name": "o3",
    "baseline_trace": "<benchmark>_o3_UPLOAD.json",
    "reasoning_effort": "medium",
    "max_steps": 5
  },
  "openai/o4-mini-2025-04-16-low": {
    "model_id": "openai/o4-mini-2025-04-16",
    "short_name": "o4-mini-low",
    "baseline_trace": "<benchmark>_o4mini_low_UPLOAD.json",
    "reasoning_effort": "low",
    "max_steps": 5
  },
  "openai/o4-mini-2025-04-16-high": {
    "model_id": "openai/o4-mini-2025-04-16",
    "short_name": "o4-mini-high",
    "baseline_trace": "<benchmark>_o4mini_high_UPLOAD.json",
    "reasoning_effort": "high",
    "max_steps": 5
  }
}
```

**Fields explained:**
- `model_id`: Full model identifier for LiteLLM/API
- `short_name`: Human-readable name for logs/output
- `baseline_trace`: Original trace file for comparison
- `reasoning_effort`: For o-series models (low/medium/high)
- `max_steps`: Agent max iterations

---

## PHASE 6: Configure the Unified Fix Runner

The unified `run_benchmark_fixes.py` handles all benchmarks. You need to:

### 6.1: Create Fixes Directory

```bash
mkdir -p fixes/<benchmark_name>
```

### 6.2: Add Benchmark to Unified Runner (if needed)

The unified runner auto-detects benchmarks by scanning `fixes/` and `model_configs/model_configs/model_to_baseline_*.json`.
If your benchmark needs special handling, you may need to update `run_benchmark_fixes.py`:

```python
# In run_benchmark_fixes.py, add to BENCHMARK_CONFIG if needed:
"<benchmark_name>": {
    "hal_benchmark": "<hal_benchmark_name>",  # Name used by hal-eval
    "fixes_dir": "fixes/<benchmark_name>",
    "task_id_field": "task_id",  # Field name in dataset
}
```

### 6.3: Verify Configuration

```bash
# List available benchmarks with fixes
python scripts/run_benchmark_fixes.py --list-benchmarks

# List configs for your benchmark
python scripts/run_benchmark_fixes.py --benchmark <benchmark_name> --list-configs
```

---

## PHASE 7: Testing and Validation

### 7.1: Test Docker Image Build

```bash
# Delete old cached images to force rebuild
docker images | grep agent-env | awk '{print $3}' | xargs -r docker rmi -f

# Clear Docker build cache
docker builder prune -af

# Run hal-eval to trigger fresh build
hal-eval --benchmark <benchmark_name> \
    --agent_dir agents/<agent_dir>/ \
    --agent_function main.run \
    --agent_name "test" \
    --docker \
    --max_tasks 1 \
    -A model_name=gpt-4o
```

### 7.2: Verify File Paths in Container

```bash
# Check a running container to verify file locations
docker exec <container_id> bash -c "pwd && ls -la && ls -la benchmark/datasets/ 2>/dev/null || echo 'not found'"
```

### 7.3: Test Rubric Evaluation

```bash
python scripts/eval_rubric.py \
    --trace-file traces/<benchmark>_*.json \
    --rubric rubric_templates/<benchmark>.txt \
    --rubric-model openai:gpt-4o \
    --failed-only -y
```

### 7.4: Test Fix Runner

```bash
# List fixes for your benchmark
python scripts/run_benchmark_fixes.py --benchmark <benchmark> --list-fixes

# Dry run
python scripts/run_benchmark_fixes.py --benchmark <benchmark> --dry-run

# Actual run with a single config
python scripts/run_benchmark_fixes.py --benchmark <benchmark> --config <config_key> --prefix test_ --docker
```

---

## Common Issues and Solutions

### Issue: Docker Cache Not Invalidating

**Symptoms:** Same hash appears after changing requirements.txt

**Solution:**
```bash
# 1. Stop all running containers
docker ps -q | xargs -r docker stop
docker ps -aq | xargs -r docker rm -f

# 2. Delete prepared images
docker images | grep agent-env | awk '{print $3}' | xargs -r docker rmi -f

# 3. Clear build cache
docker builder prune -af

# 4. Re-run evaluation
```

### Issue: Files Not Found in Container

**Symptoms:** `FileNotFoundError: benchmark/datasets/...`

**Solution:** Check that file destination matches working directory:
```python
# In benchmark.py, ensure files go to correct location
# If run_agent.py does: os.chdir("/workspace/environment")
# Then files must be copied to: "environment/benchmark/datasets/"
# NOT: "benchmark/datasets/"
```

### Issue: API Timeouts / Rate Limiting

**Symptoms:** `openai.APITimeoutError: Request timed out`

**Solution:** Reduce parallelism:
```bash
# Instead of --parallel-tasks 20, use --parallel-tasks 5
python scripts/run_benchmark_fixes.py --benchmark <benchmark> --parallel-tasks 5
```

### Issue: Sandbox "Forbidden Function" Errors

**Symptoms:** `InterpreterError: Forbidden function evaluation: 'open'`

**First, check if it matters:**
- If evaluation EXTRACTS code via regex → sandbox errors are IRRELEVANT
- If evaluation RUNS code in sandbox → need to fix AUTHORIZED_IMPORTS

**For extraction-based evaluation:** Ignore these errors in rubric evaluation.

**For sandbox-based evaluation:** Add to AUTHORIZED_IMPORTS in main.py.

### Issue: Missing Packages

**Symptoms:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
1. Add to `agents/<agent_dir>/requirements.txt`
2. Add to `AUTHORIZED_IMPORTS` in `main.py` (for smolagents)
3. Delete Docker cache and rebuild

---

## Checklist

### Phase 1: Understanding
- [ ] Read benchmark implementation (`hal/benchmarks/<benchmark>.py`)
- [ ] Read agent implementation (`agents/<agent_dir>/main.py`)
- [ ] Understand evaluation flow (sandbox vs extraction)
- [ ] Understand Docker/file path setup
- [ ] Extract all imports from gold programs
- [ ] Compare with requirements.txt and AUTHORIZED_IMPORTS

### Phase 2: Research
- [ ] Search for known benchmark issues (papers, GitHub)
- [ ] Analyze existing traces for failure patterns
- [ ] Identify cross-model failure patterns

### Phase 3: Rubric
- [ ] Create `rubric_templates/<benchmark>.txt`
- [ ] Include benchmark-specific deficiency categories
- [ ] Include proper exclusions (sandbox errors if extraction-based)
- [ ] Add cross-run analysis guidelines
- [ ] Add known problematic task patterns

### Phase 4: Fixer Script
- [ ] Create `scripts/claude_fixer_<benchmark>.py`
- [ ] Adapt trace loading for benchmark structure
- [ ] Customize prompt with benchmark specifics
- [ ] Include thorough error analysis checklist

### Phase 5: Model Config
- [ ] Create `model_configs/model_to_baseline_<benchmark>.json`
- [ ] Include all tested models
- [ ] Include reasoning_effort for o-series
- [ ] Reference correct baseline trace files

### Phase 6: Fix Runner
- [ ] Create `fixes/<benchmark>/` directory
- [ ] Verify benchmark is detected by unified runner
- [ ] Test with `--list-fixes` and `--dry-run`

### Phase 7: Documentation
- [ ] Update CLAUDE.md: Supported Benchmarks table
- [ ] Update CLAUDE.md: Benchmark-Specific Details section
- [ ] Update CLAUDE.md: Fixer Scripts section
- [ ] Create output directories

### Directories
- [ ] Create `rubrics_output/<benchmark>/`
- [ ] Create `fixes/<benchmark>/`

---

## Reference: Existing Implementations

| Benchmark | Rubric | Fixer | Model Config | Key Notes |
|-----------|--------|-------|--------------|-----------|
| SciCode | `scicode.txt` | `claude_fixer_scicode.py` | `model_configs/model_to_baseline_scicode.json` | Sandbox execution, import restrictions |
| ScienceAgentBench | `scienceagentbench.txt` | `claude_fixer_scienceagentbench.py` | `model_configs/model_to_baseline_scienceagentbench.json` | Code extraction via regex |
| CoreBench | `corebench.txt` | `claude_fixer_corebench.py` | `model_configs/model_to_baseline_corebench.json` | Docker container issues |
| ColBench | `colbench.txt` | `claude_fixer_colbench.py` | `model_configs/model_to_baseline_colbench.json` | Backend programming |

**Note**: All benchmarks use the unified `run_benchmark_fixes.py` runner.

---

## Key Principles

1. **Understand before fixing** - Deep dive into benchmark code before writing rubrics
2. **Check evaluation flow** - Sandbox errors may be irrelevant if code is extracted
3. **Cross-model analysis** - Universal failures indicate IFE, single-model failures don't
4. **Make fair, not easy** - Fixes should enable success, not guarantee it
5. **Test Docker setup** - File paths and working directories are common issues
6. **Document everything** - Future agents need to understand your reasoning
