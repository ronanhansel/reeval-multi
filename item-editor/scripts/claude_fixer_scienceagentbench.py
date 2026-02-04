#!/usr/bin/env python3
"""
Claude Code CLI-based ScienceAgentBench Fixer

Uses Claude Code CLI (claude -p) to diagnose and fix Intrinsic Formation Errors (IFEs)
in ScienceAgentBench tasks WITHOUT nerfing the scientific problems.

This script:
1. Gathers context about the task (rubric results, model conversations, task details)
2. Invokes Claude Code CLI with a detailed prompt
3. Claude Code analyzes and creates targeted fixes

Key principle: Fix INTRINSIC FORMATION ERRORS only, never simplify the science.

Usage:
    # Dry run - preview what would be processed
    python scripts/claude_fixer_scienceagentbench.py \
        --rubric-dir rubrics_output/scienceagentbench \
        --judge-csv judge_output/scienceagentbench_verdict.csv \
        --ife-only \
        --dry-run

    # Fix all IFE tasks with traces for context
    python scripts/claude_fixer_scienceagentbench.py \
        --rubric-dir rubrics_output/scienceagentbench \
        --judge-csv judge_output/scienceagentbench_verdict.csv \
        --trace-files traces/scienceagentbench_*.json \
        --ife-only \
        --tasks-per-batch 5

    # Parallel processing
    python scripts/claude_fixer_scienceagentbench.py \
        --rubric-dir rubrics_output/scienceagentbench \
        --judge-csv judge_output/scienceagentbench_verdict.csv \
        --trace-files traces/scienceagentbench_*.json \
        --ife-only \
        --tasks-per-batch 5 \
        --parallel 4

    # Fix specific tasks
    python scripts/claude_fixer_scienceagentbench.py \
        --rubric-dir rubrics_output/scienceagentbench \
        --task-ids 74 43 2

    # Skip existing fixes
    python scripts/claude_fixer_scienceagentbench.py \
        --rubric-dir rubrics_output/scienceagentbench \
        --judge-csv judge_output/scienceagentbench_verdict.csv \
        --ife-only \
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXES_DIR = REPO_ROOT / "fixes"
BENCHMARK = "scienceagentbench"

# Thread-safe progress tracking
_progress_lock = Lock()
_completed_count = 0
_total_count = 0


def log(msg: str, prefix: str = "") -> None:
    with _progress_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        tag = f"[{prefix}] " if prefix else ""
        print(f"[{ts}] {tag}{msg}", flush=True)


def log_progress(task_id: str, status: str, prefix: str = "") -> None:
    """Thread-safe progress logging for parallel mode."""
    global _completed_count
    with _progress_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        if status == "completed":
            _completed_count += 1
            print(f"[{ts}] ✓ {task_id} DONE ({_completed_count}/{_total_count})", flush=True)
        elif status == "started":
            print(f"[{ts}] → {task_id} STARTED ({_completed_count}/{_total_count} done)", flush=True)
        elif status == "skipped":
            _completed_count += 1
            print(f"[{ts}] ⊘ {task_id} SKIPPED (already has fix) ({_completed_count}/{_total_count})", flush=True)
        elif status == "failed":
            _completed_count += 1
            print(f"[{ts}] ✗ {task_id} FAILED ({_completed_count}/{_total_count})", flush=True)


def has_existing_fix(task_id: str) -> bool:
    """Check if a task already has a fix."""
    fix_dir = FIXES_DIR / BENCHMARK / task_id
    if not fix_dir.exists():
        return False
    fix_files = ["env_override.json", "evaluation_override.json", "README.md"]
    return any((fix_dir / f).exists() for f in fix_files)



def build_claude_prompt_single(
    task_id: str,
    evaluations: List[Dict[str, Any]],
    judge_verdict: Optional[Dict[str, Any]],
    conversations: Dict[str, str],
) -> str:
    """Build prompt section for a single task."""

    # Format evaluations section
    eval_sections = []
    for i, ev in enumerate(evaluations, 1):
        eval_sections.append(f"""
### Evaluation {i} (from {ev.get('model_run', 'unknown')})
- **Grade**: {ev.get('grade', 'N/A')} (1.0 = IFE detected, 0.0 = no IFE)
- **Explanation**: {ev.get('explanation', 'N/A')[:2000]}
""")
    evaluations_text = "\n".join(eval_sections) if eval_sections else "No evaluations available."

    # Format judge verdict
    judge_text = "No judge verdict available."
    if judge_verdict:
        judge_text = f"""
**Final Grade**: {judge_verdict.get('final_grade', 'N/A')}
**Satisfies Rubric (has IFE)**: {judge_verdict.get('satisfies_rubric', False)}
**Judge Reasoning**: {judge_verdict.get('reasoning', 'N/A')}
"""

    # Format conversations (model results)
    conv_sections = []
    for model, conv in list(conversations.items())[:4]:
        conv_sections.append(f"#### {model}\n```\n{conv[:3000]}\n```")
    conversations_text = "\n".join(conv_sections) if conv_sections else "No logs."

    return f"""
---
## TASK: {task_id}
---

### Rubric Evaluations
{evaluations_text}

### Judge Verdict
{judge_text}

### Model Execution Results
{conversations_text}
"""


def build_claude_prompt_batch(tasks_data: List[Dict[str, Any]]) -> str:
    """Build the prompt for Claude Code CLI with multiple tasks."""

    task_ids = [t['task_id'] for t in tasks_data]
    task_sections = []
    for t in tasks_data:
        section = build_claude_prompt_single(
            task_id=t['task_id'],
            evaluations=t['evaluations'],
            judge_verdict=t['judge_verdict'],
            conversations=t['conversations'],
        )
        task_sections.append(section)

    tasks_text = "\n".join(task_sections)

    prompt = f'''You are diagnosing and fixing Intrinsic Formation Errors (IFEs) in ScienceAgentBench tasks.

**YOU HAVE {len(task_ids)} TASKS TO PROCESS: {", ".join(task_ids)}**

Process each task sequentially, creating fixes as needed. Be THOROUGH in your analysis.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the scientific problem easier
2. **PRESERVE SCIENTIFIC RIGOR** - The task should remain as challenging as intended
3. **NO NERFING** - Do not simplify scientific concepts, give hints, reduce precision, or pre-compute results
4. **VALID FIXES**: Environment packages, Docker config, evaluation tolerance, ambiguous instructions
5. **INVALID FIXES**: Solution hints, simplified science, pre-importing specialized modules

## SCIENCEAGENTBENCH HARNESS STRUCTURE

**First, read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/scienceagentbench.py` - Main benchmark class
- `hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/` - Evaluation harness

**How Evaluation Works:**
1. Agent produces Python code for a scientific task
2. Code is executed in Docker container with `evaluation/harness.py`
3. Metrics collected: Valid Execution Rate, Success Rate (task-specific criteria), CodeBERTScore
4. For visualization tasks: GPT-4 compares generated figures to gold standard

**To inspect a specific task, run:**
```python
from datasets import load_dataset
ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
task = ds[int(TASK_ID) - 1]  # 1-indexed
print(task['task_inst'])  # Task instruction
print(task['dataset_folder_tree'])  # Data files available
```

## THOROUGH ERROR ANALYSIS CHECKLIST

For EACH task, systematically check ALL of these potential error sources:

### 1. Environment/Dependency Issues
- [ ] Missing Python packages in Docker container (oggm, mne, mastml, biopsykit)
- [ ] Package version conflicts between scientific libraries
- [ ] Missing system libraries (GDAL, MPI, etc.)
- [ ] Conda vs pip installation conflicts for scientific packages
- [ ] GPU/CUDA requirements for deep learning tasks
- [ ] Memory/timeout limits too restrictive for large datasets

### 2. Data/Input Issues
- [ ] Missing or corrupted data files in dataset
- [ ] Data format differs from task documentation
- [ ] Column names don't match task description
- [ ] Encoding issues with scientific data files
- [ ] File path mismatches between task description and actual paths

### 3. Task Specification Issues
- [ ] Ambiguous output format requirements
- [ ] Unclear success criteria (what makes output "correct"?)
- [ ] Missing domain knowledge in instructions (formulas, methods)
- [ ] Conflicting requirements between task steps
- [ ] Unstated assumptions from source scientific papers

### 4. Evaluation Script Issues
- [ ] Numerical tolerance too strict for scientific precision
- [ ] Format-sensitive comparison (whitespace, column ordering)
- [ ] Evaluation crashes on valid but alternative outputs
- [ ] Metrics don't match task description expectations
- [ ] GPT-4 judge subjectivity for figure/visualization tasks

### 5. Gold Program/Reference Issues
- [ ] Gold program has hardcoded paths
- [ ] Gold program uses unavailable or outdated libraries
- [ ] Multiple scientifically valid approaches rejected
- [ ] Gold program doesn't match task requirements exactly

### 6. Cross-Model Failure Patterns
- [ ] Same error across ALL models → likely IFE
- [ ] Valid output rejected by evaluation → evaluation issue
- [ ] Environment blocks ALL models identically → setup issue
- [ ] Figure evaluation fails for functionally equivalent plots

## KNOWN IFE PATTERNS (from trace analysis)

**Environment Issues (High Priority):**
- Task 74: `oggm` / `oggm.core.distribute_2d` (glacier modeling) not available
- Task 43: `mne` (neuroimaging) not available
- Task 2: `mastml.features` (materials science ML) not available
- Other domain-specific packages: `biopsykit`, specialized GIS libraries

**Figure Evaluation Issues:**
- GPT-4 judge penalizes functionally equivalent but stylistically different visualizations
- Color scheme differences cause failures despite correct scientific content
- Axis label formatting differences cause failures
- Tasks 8, 25, 28, 50, 59, 68, 84, 91, 93 have figure evaluation issues

**Key Statistics:**
- 67/102 tasks failed across ALL 4 models (GPT-4.1, O3, O4-mini-high, O4-mini-low)
- 39 tasks have runtime errors in 2+ models
- 13 tasks have figure comparison failures in 2+ models

## FIX OUTPUT FORMAT

For each task that needs a fix, create: `fixes/scienceagentbench/TASK_ID/`

**Environment Fixes** (`env_override.json`):
```json
{{
  "HAL_CONDA_CHANNELS": "conda-forge",
  "HAL_CONDA_PACKAGES": "mne oggm",
  "HAL_PIP_PACKAGES": "biopsykit mastml",
  "HAL_APT_PACKAGES": "libfoo-dev",
  "HAL_TIMEOUT_SECONDS": 600,
  "notes": "Justification for these environment changes"
}}
```

**Evaluation Fixes** (`evaluation_override.json`):
```json
{{
  "figure_tolerance": "relaxed",
  "numerical_tolerance": 1e-4,
  "accept_alternative_formats": true,
  "skip_style_check": true,
  "notes": "Why this adjustment is fair, not a nerf"
}}
```

**Instruction Clarifications** (`instruction_override.json`):
```json
{{
  "clarifications": [
    "Output file must be named exactly 'output.csv'",
    "Use specific library version X"
  ],
  "additional_context": "Any missing domain knowledge needed"
}}
```

**Documentation** (`README.md`):
- Root cause analysis of the IFE
- What fix was applied and why
- Why this preserves task difficulty
- Expected outcome after fix

If NO fix needed (capability issue, not IFE), create README.md explaining why.

## FIX RUNNER SCRIPT

After creating fixes, also check/update the fix runner script:
`scripts/run_scienceagentbench_fixes.py`

The fix runner must:
1. Load fixes from `fixes/scienceagentbench/<task_id>/`
2. Apply environment overrides before Docker evaluation
3. Inject instruction clarifications into task prompts
4. Adjust evaluation parameters as specified
5. Run HAL evaluation with fixes applied
6. Output new traces with configurable prefix

**Reference implementation**: See `scripts/run_scicode_fixes.py` for the pattern.

## TASKS TO PROCESS

{tasks_text}

## BEGIN - SYSTEMATIC APPROACH

For EACH task:

1. **Read the benchmark code** to understand evaluation pipeline
2. **Load the specific task** from HuggingFace dataset
3. **Analyze ALL error messages** from model execution logs
4. **Check EACH item** in the error analysis checklist above
5. **Cross-reference with other models** - same error = likely IFE
6. **Create fix OR document why no fix needed**
7. **Verify fix doesn't nerf the scientific problem**

After processing all tasks:
8. **Update fix runner script** if new fix types were used
9. **Test that fixes can be applied** without errors

Remember: Make evaluation FAIR, not EASY. Be THOROUGH in diagnosis.
Scientific rigor must be preserved - we fix infrastructure, not science.
'''

    return prompt


def format_stream_json(line: str, task_id: str) -> None:
    """Format and print a JSON stream line nicely."""
    try:
        data = json.loads(line)
        msg_type = data.get("type", "unknown")

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"

        ts = datetime.now().strftime("%H:%M:%S")

        if msg_type == "assistant":
            content = data.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        if "file_path" in tool_input:
                            path = tool_input.get("file_path", "")
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {path}")
                        else:
                            print(f"{DIM}[{ts}]{RESET} {YELLOW}[TOOL: {tool_name}]{RESET} {json.dumps(tool_input)[:200]}")
            elif isinstance(content, str) and content:
                print(f"{DIM}[{ts}]{RESET} {CYAN}[ASSISTANT]{RESET} {content[:500]}")

        elif msg_type == "user":
            content = data.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "")[:8]
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            preview = result_content[:300].replace("\n", " ")
                            print(f"{DIM}[{ts}]{RESET} {GREEN}[RESULT {tool_id}]{RESET} {preview}...")

        elif msg_type == "result":
            cost = data.get("cost_usd", 0)
            duration = data.get("duration_ms", 0) / 1000
            print(f"\n{BOLD}{GREEN}[COMPLETED]{RESET} Task: {task_id}")
            print(f"  Cost: ${cost:.4f} | Duration: {duration:.1f}s")

        elif msg_type == "error":
            error = data.get("error", {})
            print(f"{DIM}[{ts}]{RESET} {RED}[ERROR]{RESET} {error.get('message', str(error))}")

        elif msg_type == "system":
            msg = data.get("message", "")
            if msg:
                print(f"{DIM}[{ts}]{RESET} {BLUE}[SYSTEM]{RESET} {msg}")

    except json.JSONDecodeError:
        if line.strip():
            print(line.strip())


def run_claude_code(
    prompt: str,
    task_id: str,
    working_dir: Path,
    fix_dir: Path,
    quiet: bool = False,
) -> int:
    """Run Claude Code CLI with the given prompt."""

    base_cmd = [
        "claude",
        "--dangerously-skip-permissions",
    ]

    if not quiet:
        base_cmd.extend(["--verbose", "--output-format", "stream-json"])
    else:
        base_cmd.extend(["--output-format", "json"])

    base_cmd.extend(["-p", prompt])

    log_path = fix_dir / "claude_session.jsonl"

    if not quiet:
        log(f"Running Claude Code CLI for {task_id}...")
        log(f"Working directory: {working_dir}")
        print(f"\n{'='*60}")
        print(f"CLAUDE CODE SESSION: {task_id}")
        print(f"{'='*60}\n")

        process = subprocess.Popen(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            with log_path.open("w") as log_file:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    format_stream_json(line, task_id)
        except KeyboardInterrupt:
            process.terminate()
            log(f"Interrupted by user")
            return 130

        process.wait()
        log(f"Session log saved to: {log_path}")
        return process.returncode

    else:
        # Quiet mode
        result = subprocess.run(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            capture_output=True,
            text=True,
        )
        with log_path.open("w") as log_file:
            log_file.write(result.stdout)
        return result.returncode



def process_task_batch(
    task_ids: List[str],
    rubric_dir: Path,
    judge_verdicts: Dict[str, Dict[str, Any]],
    trace_files: List[Path],
    benchmark: str,
    batch_id: int = 0,
    quiet: bool = False,
) -> List[Tuple[str, bool, str]]:
    """Process a batch of tasks in a single Claude session. Returns list of (task_id, success, message)."""

    try:
        # Gather data for all tasks in batch
        tasks_data = []
        for task_id in task_ids:
            evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
            judge_verdict = judge_verdicts.get(task_id)
            conversations = load_task_conversations_from_files(trace_files, task_id)

            tasks_data.append({
                'task_id': task_id,
                'evaluations': evaluations,
                'judge_verdict': judge_verdict,
                'conversations': conversations,
            })

            # Create fix directory for each task
            fix_dir = FIXES_DIR / benchmark / task_id
            fix_dir.mkdir(parents=True, exist_ok=True)

        # Build batch prompt
        prompt = build_claude_prompt_batch(tasks_data)

        # Save prompt to first task's directory
        batch_fix_dir = FIXES_DIR / benchmark / task_ids[0]
        (batch_fix_dir / "claude_prompt_batch.txt").write_text(prompt)

        if not quiet:
            log(f"Batch {batch_id}: Starting {len(task_ids)} tasks: {', '.join(task_ids)}")

        # Run Claude Code for entire batch
        exit_code = run_claude_code(
            prompt=prompt,
            task_id=f"batch_{batch_id}_{'-'.join(task_ids[:3])}",
            working_dir=REPO_ROOT,
            fix_dir=batch_fix_dir,
            quiet=quiet,
        )

        if exit_code == 0:
            if not quiet:
                log(f"Batch {batch_id}: Completed all {len(task_ids)} tasks")
            return [(tid, True, "Batch completed successfully") for tid in task_ids]
        else:
            if not quiet:
                log(f"Batch {batch_id}: Failed with exit code {exit_code}")
            return [(tid, False, f"Batch failed with code {exit_code}") for tid in task_ids]

    except Exception as e:
        log(f"Batch {batch_id}: Exception - {e}")
        return [(tid, False, str(e)) for tid in task_ids]


def load_all_rubric_evaluations(rubric_dir: Path, task_id: str) -> List[Dict[str, Any]]:
    """Load all rubric evaluations for a task from all CSV files in the rubric directory."""
    evaluations = []
    for csv_file in rubric_dir.glob("*.csv"):
        with csv_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("task_id") == task_id:
                    evaluations.append({
                        "source": csv_file.stem,
                        "grade": row.get("grade", ""),
                        "explanation": row.get("explanation", ""),
                        "model_run": row.get("model_run", ""),
                    })
    return evaluations


def load_task_conversations_from_files(trace_files: List[Path], task_id: str) -> Dict[str, str]:
    """Load conversations for a task from provided trace files."""
    conversations = {}

    for trace_path in trace_files:
        if not trace_path.exists():
            continue

        try:
            data = json.loads(trace_path.read_text())
        except:
            continue

        # Extract model name
        model_name = data.get("config", {}).get("agent_args", {}).get("model_name", trace_path.stem[:30])
        model_name = model_name.replace("openai/", "").replace("-2025", "")

        # Get eval results for this task
        raw_eval = data.get("raw_eval_results", {}).get("eval_result", {})
        task_result = raw_eval.get(task_id, {})

        if task_result:
            # Format the result
            lines = []
            if isinstance(task_result, dict):
                for key, value in task_result.items():
                    if isinstance(value, str) and len(value) > 500:
                        value = value[:500] + "..."
                    lines.append(f"{key}: {value}")
            conversations[model_name] = "\n".join(lines)

    return conversations


def main():
    global _total_count, _completed_count

    parser = argparse.ArgumentParser(
        description="Use Claude Code to diagnose and fix ScienceAgentBench IFEs"
    )

    parser.add_argument(
        "--rubric-dir",
        type=str,
        required=True,
        help="Directory containing rubric CSV outputs (e.g., rubrics_output/scienceagentbench)",
    )
    parser.add_argument(
        "--judge-csv",
        type=str,
        help="Path to judge verdict CSV (optional)",
    )
    parser.add_argument(
        "--trace-files",
        type=str,
        nargs="+",
        default=[],
        help="Trace files to extract conversations from (optional, provides additional context)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="scienceagentbench",
        help="Benchmark name (default: scienceagentbench)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="Specific task IDs to process (default: all with IFEs)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--min-grade",
        type=float,
        default=0.5,
        help="Minimum rubric grade to consider as IFE (default: 0.5)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have fixes",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel Claude Code sessions (default: 1)",
    )
    parser.add_argument(
        "--tasks-per-batch",
        type=int,
        default=5,
        help="Number of tasks per Claude session (default: 5). Each Claude instance processes this many tasks sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without running Claude Code",
    )
    parser.add_argument(
        "--ife-only",
        action="store_true",
        help="Only process tasks with judge verdict = 1 (confirmed IFEs). Requires --judge-csv.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="inspect",
        help="Prefix for logging output (default: inspect)",
    )

    args = parser.parse_args()
    prefix = args.prefix

    # Color codes for terminal
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}ScienceAgentBench IFE Fixer - Claude Code CLI{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

    # Resolve paths
    rubric_dir = Path(args.rubric_dir)
    if not rubric_dir.is_absolute():
        rubric_dir = REPO_ROOT / rubric_dir

    log(f"Rubric directory: {rubric_dir}", prefix)
    log(f"Benchmark: {args.benchmark}", prefix)

    trace_files = [Path(f) if Path(f).is_absolute() else REPO_ROOT / f for f in args.trace_files] if args.trace_files else []
    if trace_files:
        log(f"Using {len(trace_files)} trace files for conversation context", prefix)
        for tf in trace_files[:3]:
            log(f"  - {tf.name}", prefix)
        if len(trace_files) > 3:
            log(f"  ... and {len(trace_files) - 3} more", prefix)
    else:
        log("No trace files provided - using rubric evaluations only", prefix)
        log(f"  Hint: Trace files are in traces/ with names like scienceagentbench_*.json", prefix)

    # Load judge verdicts if provided
    judge_verdicts = {}
    if args.judge_csv:
        judge_path = Path(args.judge_csv)
        if not judge_path.is_absolute():
            judge_path = REPO_ROOT / judge_path
        judge_verdicts = load_judge_verdicts_from_file(judge_path)
        log(f"Loaded {len(judge_verdicts)} judge verdicts from {judge_path.name}", prefix)

    # Validate --ife-only requires --judge-csv
    if args.ife_only and not args.judge_csv:
        print(f"{RED}Error: --ife-only requires --judge-csv to be specified{RESET}")
        return

    # Determine tasks to process
    if args.task_ids:
        task_ids = args.task_ids
        log(f"Processing {len(task_ids)} specified tasks: {', '.join(task_ids)}", prefix)
    else:
        # Find all tasks with IFEs based on rubric grades
        task_ids = set()
        csv_count = 0
        for csv_file in rubric_dir.glob("*.csv"):
            csv_count += 1
            with csv_file.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        grade = float(row.get("grade", 0) or 0)
                    except (ValueError, TypeError):
                        grade = 0
                    if grade >= args.min_grade:
                        task_ids.add(row.get("task_id", ""))
        task_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else x)
        log(f"Scanned {csv_count} CSV files in {rubric_dir.name}", prefix)
        log(f"Found {len(task_ids)} tasks with grade >= {args.min_grade} (potential IFEs)", prefix)

    # Filter to confirmed IFEs only (judge verdict = 1)
    if args.ife_only:
        original_count = len(task_ids)
        ife_task_ids = []
        for tid in task_ids:
            verdict = judge_verdicts.get(tid, {})
            if verdict.get("final_grade", 0) == 1:
                ife_task_ids.append(tid)
        task_ids = ife_task_ids
        log(f"Filtered to {len(task_ids)} confirmed IFEs (verdict=1) from {original_count} candidates", prefix)

    # Filter existing fixes
    skipped_tasks = []
    if args.skip_existing:
        original_count = len(task_ids)
        for tid in task_ids:
            if has_existing_fix(tid):
                skipped_tasks.append(tid)
        task_ids = [t for t in task_ids if t not in skipped_tasks]
        log(f"Skipping {len(skipped_tasks)} tasks with existing fixes, {len(task_ids)} remaining", prefix)

    # Limit tasks
    if args.max_tasks and len(task_ids) > args.max_tasks:
        log(f"Limiting to first {args.max_tasks} tasks (use --max-tasks to change)", prefix)
        task_ids = task_ids[:args.max_tasks]

    if not task_ids:
        log(f"{YELLOW}No tasks to process{RESET}", prefix)
        return

    # Set totals for progress tracking
    _total_count = len(task_ids) + len(skipped_tasks)
    _completed_count = len(skipped_tasks)

    # Create batches
    tasks_per_batch = args.tasks_per_batch
    batches = [task_ids[i:i + tasks_per_batch] for i in range(0, len(task_ids), tasks_per_batch)]
    num_batches = len(batches)

    print(f"\n{BOLD}Configuration:{RESET}")
    print(f"  Tasks to process: {len(task_ids)}")
    print(f"  Tasks per batch: {tasks_per_batch}")
    print(f"  Total batches: {num_batches}")
    print(f"  Parallel sessions: {args.parallel}")
    print(f"  Fixes output: fixes/{args.benchmark}/<task_id>/")
    print()

    if args.dry_run:
        print(f"\n{BOLD}{YELLOW}{'='*60}{RESET}")
        print(f"{BOLD}{YELLOW}DRY RUN MODE - Previewing batch prompts{RESET}")
        print(f"{BOLD}{YELLOW}{'='*60}{RESET}\n")

        # Preview first batch
        preview_batch = batches[0] if batches else []
        tasks_data = []
        for task_id in preview_batch:
            evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
            judge_verdict = judge_verdicts.get(task_id)
            conversations = load_task_conversations_from_files(trace_files, task_id)
            tasks_data.append({
                'task_id': task_id,
                'evaluations': evaluations,
                'judge_verdict': judge_verdict,
                'conversations': conversations,
            })
            # Show task details
            log(f"Task {task_id}:", prefix)
            log(f"  - Evaluations: {len(evaluations)}", prefix)
            log(f"  - Judge verdict: {'Yes' if judge_verdict else 'No'}", prefix)
            log(f"  - Conversations: {len(conversations)} models", prefix)

        prompt = build_claude_prompt_batch(tasks_data)

        print(f"\n{BOLD}BATCH 1 of {num_batches}{RESET}")
        print(f"Tasks: {', '.join(preview_batch)}")
        print(f"Prompt length: {len(prompt):,} chars")
        print(f"\n{DIM}{'='*60}{RESET}")
        print(prompt[:5000] + "..." if len(prompt) > 5000 else prompt)

        print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
        print(f"{BOLD}DRY RUN SUMMARY{RESET}")
        print(f"{'='*60}")
        print(f"  Total tasks: {len(task_ids)}")
        print(f"  Skipped (existing): {len(skipped_tasks)}")
        print(f"  Batches: {num_batches}")
        print(f"  Tasks per batch: {tasks_per_batch}")
        print(f"  Parallel sessions: {args.parallel}")
        print(f"  Total Claude sessions: {num_batches}")
        print(f"{'='*60}\n")
        return

    # Process batches
    all_results = []

    if args.parallel > 1:
        log(f"\n{BOLD}Running {args.parallel} parallel Claude sessions, {tasks_per_batch} tasks each{RESET}", prefix)
        print(f"{'='*60}\n")

        def run_batch(batch_tuple):
            batch_idx, batch = batch_tuple
            log_progress(f"batch_{batch_idx}", "started", prefix)
            results = process_task_batch(
                task_ids=batch,
                rubric_dir=rubric_dir,
                judge_verdicts=judge_verdicts,
                trace_files=trace_files,
                benchmark=args.benchmark,
                batch_id=batch_idx,
                quiet=True,
            )
            for tid, success, msg in results:
                if success:
                    log_progress(tid, "completed", prefix)
                else:
                    log_progress(tid, "failed", prefix)
            return results

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_batch, (idx, batch)): idx for idx, batch in enumerate(batches)}

            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)

    else:
        # Sequential batch processing with detailed logging
        log(f"\nProcessing {len(task_ids)} tasks in {num_batches} batches (sequential mode)", prefix)
        print(f"{'='*60}\n")

        for batch_idx, batch in enumerate(batches):
            print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
            print(f"{BOLD}BATCH {batch_idx + 1}/{num_batches}{RESET}")
            print(f"Tasks: {', '.join(batch)}")
            print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

            # Show task details before processing
            for task_id in batch:
                evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
                judge_verdict = judge_verdicts.get(task_id)
                conversations = load_task_conversations_from_files(trace_files, task_id)

                log(f"Task {task_id}:", prefix)
                log(f"  - Rubric evaluations: {len(evaluations)}", prefix)
                for ev in evaluations[:2]:
                    grade_str = ev.get('grade', 'N/A')
                    log(f"    * {ev.get('source', 'unknown')}: grade={grade_str}", prefix)
                if len(evaluations) > 2:
                    log(f"    ... and {len(evaluations) - 2} more", prefix)
                log(f"  - Judge verdict: {judge_verdict.get('final_grade', 'N/A') if judge_verdict else 'None'}", prefix)
                log(f"  - Conversation logs: {len(conversations)} models", prefix)

            batch_results = process_task_batch(
                task_ids=batch,
                rubric_dir=rubric_dir,
                judge_verdicts=judge_verdicts,
                trace_files=trace_files,
                benchmark=args.benchmark,
                batch_id=batch_idx,
                quiet=False,
            )
            all_results.extend(batch_results)

            # Batch summary
            batch_success = sum(1 for _, s, _ in batch_results if s)
            print(f"\n{BOLD}Batch {batch_idx + 1} Result:{RESET} {batch_success}/{len(batch)} tasks completed")

    # Final summary
    successful = [r for r in all_results if r[1]]
    failed = [r for r in all_results if not r[1]]

    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}FINAL SUMMARY{RESET}")
    print(f"{'='*60}")
    print(f"\n{GREEN}Succeeded: {len(successful)}/{len(all_results)}{RESET}")
    for tid, _, msg in successful[:10]:
        print(f"  {GREEN}✓{RESET} {tid}")
    if len(successful) > 10:
        print(f"  ... and {len(successful) - 10} more")

    if failed:
        print(f"\n{RED}Failed: {len(failed)}/{len(all_results)}{RESET}")
        for tid, _, msg in failed:
            print(f"  {RED}✗{RESET} {tid}: {msg}")

    if skipped_tasks:
        print(f"\n{YELLOW}Skipped (existing fixes): {len(skipped_tasks)}{RESET}")
        for tid in skipped_tasks[:5]:
            print(f"  {YELLOW}⊘{RESET} {tid}")
        if len(skipped_tasks) > 5:
            print(f"  ... and {len(skipped_tasks) - 5} more")

    print(f"\n{BOLD}Fixes saved to:{RESET} fixes/{args.benchmark}/<task_id>/")
    print(f"{'='*60}\n")


def load_judge_verdicts_from_file(verdict_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load judge verdicts from a CSV file."""
    results = {}
    if not verdict_csv.exists():
        return results
    with verdict_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "")
            results[task_id] = {
                "final_grade": float(row.get("final_grade", 0)),
                "satisfies_rubric": row.get("satisfies_rubric", "0") == "1",
                "reasoning": row.get("reasoning", ""),
                "num_evaluations": int(row.get("num_evaluations", 0)),
                "model_runs": row.get("model_runs", "").split(";"),
            }
    return results


if __name__ == "__main__":
    main()
