# USACO Fix Packages

This directory contains fix packages for USACO benchmark tasks that have been identified as having **Intrinsic Formation Errors (IFE)** - benchmark defects that prevent any agent from succeeding.

## Directory Structure

Each task has its own subdirectory named by task_id:

```
fixes/usaco/
├── README.md                    # This file
├── <task_id>/
│   ├── README.md                # Analysis of the issue and fix
│   ├── instruction_override.json   # Fix problem description/constraints
│   ├── evaluation_override.json    # Fix judge/time limits/precision
│   ├── input_override.json         # Add clarifications
│   ├── test_case_override.json     # Fix specific test cases
│   └── problem_statement.txt       # Raw text additions (optional)
└── ...
```

## Fix File Formats

### instruction_override.json

Fix problem statement issues:

```json
{
  "task_id": "usaco_standin-bronze-2023-jan",
  "description": "Fix ambiguous constraint specification",
  "clarifications": [
    "Clarification 1 about input format",
    "Clarification 2 about edge cases"
  ],
  "corrected_description": "Full corrected problem statement (optional)",
  "corrected_constraints": "Fixed constraint text (optional)",
  "corrected_input_format": "Fixed input format (optional)",
  "corrected_output_format": "Fixed output format (optional)"
}
```

### evaluation_override.json

Fix judge/evaluation issues:

```json
{
  "task_id": "usaco_standin-bronze-2023-jan",
  "description": "Fix time limit issue",
  "notes": "Description of the judge issue",
  "time_limit_override": 5.0,
  "precision_override": 1e-6
}
```

### input_override.json

Add additional clarifications:

```json
{
  "task_id": "usaco_standin-bronze-2023-jan",
  "clarifications": [
    "Important constraint not mentioned in original",
    "Expected output format specification"
  ]
}
```

### test_case_override.json

Fix specific test case errors:

```json
{
  "task_id": "usaco_standin-bronze-2023-jan",
  "description": "Fix incorrect expected output in test case 3",
  "fixes": {
    "3": {
      "expected_output": "5\n",
      "input": "1 2 3\n"
    }
  }
}
```

### problem_statement.txt

Raw text to append to the problem description:

```
Additional context or clarifications in plain text format.
This will be added under "## ADDITIONAL CONTEXT" heading.
```

## Usage

After creating fix packages, run the fixing pipeline:

```bash
# List available fixes
python scripts/run_usaco_fixes.py --list-fixes

# Verify fixes work correctly (dry run)
python scripts/run_usaco_fixes.py --verify-fixes

# Run fixes for specific tasks
python scripts/run_usaco_fixes.py \
    --prefix usaco_lime_ \
    --task-id usaco_standin-bronze-2023-jan \
    --docker

# Run all fixes with specific model
python scripts/run_usaco_fixes.py \
    --prefix usaco_lime_ \
    --model openai/gpt-4.1-2025-04-14 \
    --docker
```

## Creating New Fixes

1. Identify the task failure from rubric evaluation (Grade=1)
2. Create a directory with the task_id name
3. Add a README.md documenting the issue
4. Create appropriate override JSON files
5. Test with `--verify-fixes` before running

## Score Guidelines

Per the USACO rubric (`rubric_templates/usaco.txt`):

- **Score 1 (Benchmark Defect)**: Problem statement errors, test case errors, judge system defects, resource issues, execution environment issues
- **Score 0 (Agent Issue)**: Algorithm selection errors, implementation bugs, time/space complexity issues, I/O handling errors

Only create fix packages for Score 1 issues.
