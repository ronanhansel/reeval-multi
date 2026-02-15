# ColBench Fix Packages

This directory contains fix packages for ColBench benchmark tasks that have been identified as having **Intrinsic Formation Errors (IFE)** - benchmark defects that prevent any agent from succeeding.

## Directory Structure

Each task has its own subdirectory named by task index (0-99 for frontend, 0-999 for backend):

```
fixes/colbench/
├── README.md                    # This file
├── <task_id>/
│   ├── README.md                # Analysis of the issue and fix
│   ├── evaluation_override.json    # Document evaluation issues (CLIP, test cases)
│   ├── instruction_override.json   # Clarify task requirements
│   └── simulated_user_override.json # Document simulated user issues
└── ...
```

## Fix File Formats

### evaluation_override.json

Document evaluation-related issues:

```json
{
  "issue_type": "parser_regex_mismatch",
  "severity": "critical",
  "description": "Frontend HTML task uses Python-only code fence regex parser",
  "evidence": {
    "parser_error": "regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found",
    "task_requirement": "raw html code wrapped in <html> tag",
    "contradiction": "Parser rejects valid HTML submissions that follow task instructions"
  },
  "required_fix": {
    "location": "benchmark harness/parser",
    "action": "Accept html fenced blocks or raw HTML for frontend tasks"
  },
  "notes": "This is an infrastructure-level IFE affecting all frontend tasks."
}
```

### instruction_override.json

Clarify ambiguous task requirements:

```json
{
  "task_id": "15",
  "description": "Clarify output format requirement",
  "clarifications": [
    "Output should be wrapped in ```html code fence",
    "Include complete HTML document with DOCTYPE"
  ]
}
```

### simulated_user_override.json

Document simulated user response issues:

```json
{
  "task_id": "8",
  "description": "Simulated user provides contradictory feedback",
  "evidence": {
    "turn_1_response": "User says design should be minimalist",
    "turn_3_response": "User says design lacks visual elements",
    "contradiction": "Minimalist requirement conflicts with 'more visual elements' feedback"
  },
  "notes": "Agent cannot satisfy both pieces of feedback simultaneously."
}
```

## Common IFE Types in ColBench

### 1. Parser Regex Mismatch
The benchmark harness uses a Python-only regex (`(?:py|python)?\\s*\\n(.*?)\\n`) that rejects valid HTML submissions for frontend tasks.

### 2. Spec/Ground Truth Mismatch
Task specification describes one design but ground truth image shows a completely different design. Agent cannot match both.

### 3. Simulated User Issues
- Contradictory feedback across turns
- Irrelevant responses unrelated to the design task
- Hidden information never revealed through dialogue

### 4. CLIP Similarity Evaluation
- Valid alternative designs penalized
- Color scheme differences over-weighted
- Layout variations scored as failures

## Usage

After creating fix packages, run the fixing pipeline:

```bash
# List available fixes
python scripts/run_colbench_fixes.py --list-fixes

# Dry run to see what would execute
python scripts/run_colbench_fixes.py --all --all-models --dry-run

# Run fixes for a specific task with all models (with Docker)
python scripts/run_colbench_fixes.py --task-id 1 --all-models --prefix fixed_ --docker

# Run all fixes with all models in parallel (with Docker)
python scripts/run_colbench_fixes.py --all --all-models --parallel 3 --prefix iter1_ --docker

# Run specific task with specific model (with Docker)
python scripts/run_colbench_fixes.py --task-id 1 --model gpt-4.1-2025-04-14 --prefix test_ --docker
```

## Fix Runner Behavior

The fix runner (`scripts/run_colbench_fixes.py`) has the following characteristics:

- **Default benchmark**: `colbench_frontend_design` (100 tasks with CLIP similarity evaluation)
- **Model config**: Uses `model_to_baseline_colbench.json` with 3 models:
  - `gpt-4.1-2025-04-14` (short_name: gpt-4.1)
  - `o3-2025-04-16` (short_name: o3-low, reasoning_effort: low)
  - `o4-mini-2025-04-16` (short_name: o4-mini-high, reasoning_effort: high)
- **`--all-models` flag**: Creates jobs for every (task × model) combination
- **Parallel execution**: Use `--parallel N` to run N jobs concurrently

## Creating New Fixes

1. Identify the task failure from rubric evaluation (Grade=1)
2. Create a directory with the task index as name
3. Add a README.md documenting the issue
4. Create appropriate override JSON files
5. Test with `--dry-run` before running

## Score Guidelines

Per the ColBench rubric (`rubric_templates/colbench.txt`):

- **Score 1 (Benchmark Defect)**: Simulated user issues, hidden info accessibility problems, test case errors, CLIP evaluation issues, parser mismatches
- **Score 0 (Agent Issue)**: Dialogue strategy errors, code generation bugs, design misinterpretation, turn limit issues due to poor questioning

Only create fix packages for Score 1 issues.

## Current Fix Statistics

As of the initial analysis:
- **Total fixes**: 32 tasks
- **With evaluation_override.json**: 17 tasks (infrastructure issues)
- **README-only**: 15 tasks (documented but no actionable fix)

Most fixes are `evaluation_override.json` documenting:
1. Parser regex mismatch (HTML rejected by Python regex)
2. Spec/ground truth mismatch (task describes full website, ground truth is minimal design)
