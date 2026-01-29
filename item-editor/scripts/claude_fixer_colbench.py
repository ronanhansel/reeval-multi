#!/usr/bin/env python3
"""
Claude Code CLI-based ColBench Fixer

Uses Claude Code CLI (claude -p) to diagnose and fix Intrinsic Formation Errors (IFEs)
in ColBench benchmark tasks.

This script:
1. Gathers context about the task (rubric results, model conversations, task details)
2. Invokes Claude Code CLI with a detailed prompt
3. Claude Code analyzes and creates targeted fixes

Key principle: Fix INTRINSIC FORMATION ERRORS only (simulated user issues, evaluation issues),
never simplify the collaborative task itself.
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
TRACES_DIR = REPO_ROOT / "traces"
FIXES_DIR = REPO_ROOT / "fixes"

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


def has_existing_fix(benchmark: str, task_id: str) -> bool:
    """Check if a task already has a fix."""
    fix_dir = FIXES_DIR / benchmark / task_id
    if not fix_dir.exists():
        return False
    fix_files = ["env_override.json", "instruction_override.json", "evaluation_override.json", "README.md"]
    return any((fix_dir / f).exists() for f in fix_files)


def load_rubric_results(rubric_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load rubric evaluation results by task_id."""
    results = {}
    with rubric_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "")
            results[task_id] = {
                "score": float(row.get("grade", 0)),
                "explanation": row.get("explanation", ""),
                "model_run": row.get("model_run", ""),
            }
    return results


def load_judge_verdicts(verdict_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load judge verdicts by task_id."""
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


def load_task_conversations(trace_files: List[Path], task_id: str) -> Dict[str, str]:
    """Load conversations for a task from multiple trace files."""
    conversations = {}

    for trace_path in trace_files:
        if not trace_path.exists():
            continue

        try:
            data = json.loads(trace_path.read_text())
        except:
            continue

        # Extract model name and benchmark type
        config = data.get("config", {})
        model_name = config.get("agent_args", {}).get("model_name", trace_path.stem[:30])
        benchmark_name = config.get("benchmark_name", "colbench")
        model_name = model_name.replace("openai/", "").replace("-2025", "")

        # Add benchmark type to model name for clarity
        if "frontend" in benchmark_name:
            model_name = f"{model_name}_frontend"
        else:
            model_name = f"{model_name}_backend"

        # Find conversation for this task
        raw_entries = data.get("raw_logging_results", [])
        task_entries = []

        for entry in raw_entries:
            entry_task = (
                entry.get("attributes", {}).get("weave_task_id")
                or entry.get("weave_task_id")
                or ""
            )
            if entry_task == task_id:
                task_entries.append(entry)

        if task_entries:
            # Format conversation
            lines = []
            for entry in task_entries[:30]:  # Limit entries
                messages = entry.get("inputs", {}).get("messages", [])
                for msg in messages[-5:]:  # Last 5 messages per entry (more for colbench dialogues)
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    if content:
                        lines.append(f"[{role}]: {content[:3000]}")

                # Assistant output
                output = entry.get("output", {})
                choices = output.get("choices", [])
                if choices:
                    out_content = choices[0].get("message", {}).get("content", "")
                    if isinstance(out_content, list):
                        out_content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in out_content
                        )
                    if out_content:
                        lines.append(f"[assistant]: {out_content[:3000]}")

            conversations[model_name] = "\n\n".join(lines[-60:])  # Last 60 exchanges for dialogues

    return conversations


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
- **Explanation**: {ev.get('explanation', 'N/A')[:3000]}
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

    # Format conversations (abbreviated)
    conv_sections = []
    for model, conv in list(conversations.items())[:4]:  # Max 4 models per task (backend/frontend)
        conv_sections.append(f"#### {model}\n```\n{conv[:8000]}\n```")
    conversations_text = "\n".join(conv_sections) if conv_sections else "No logs."

    return f"""
---
## TASK: {task_id}
---

### Rubric Evaluations
{evaluations_text}

### Judge Verdict
{judge_text}

### Model Conversation Logs (abbreviated)
{conversations_text}
"""


def build_claude_prompt_batch(
    tasks_data: List[Dict[str, Any]],
    benchmark: str,
) -> str:
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

    prompt = f'''You are diagnosing and fixing Intrinsic Formation Errors (IFEs) in ColBench benchmark tasks.

**YOU HAVE {len(task_ids)} TASKS TO PROCESS: {", ".join(task_ids)}**

Process each task sequentially, creating fixes as needed. You only need to read the HAL harness files ONCE at the start.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the collaborative task easier
2. **PRESERVE TASK DIFFICULTY** - The task should remain as challenging as intended
3. **NO NERFING** - Do not give hints, simplify requirements, or reveal hidden information

## ColBench Overview

ColBench evaluates collaborative agents through multi-turn dialogue with a simulated human user:
- **Backend Programming**: Agent helps user write Python code (1000 tasks)
- **Frontend Design**: Agent helps user create HTML/CSS designs (100 tasks)

Key architecture:
1. Agent receives problem description
2. Agent asks questions (up to 10 rounds)
3. Simulated user (GPT-4o) responds based on hidden information
4. Agent provides final answer (code or HTML)
5. Evaluation: test cases (backend) or CLIP similarity (frontend)

## HAL HARNESS STRUCTURE - READ ONCE AT START

**FIRST, read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/colbench.py` - Main benchmark class
- `hal-harness/agents/colbench_example_agent/main.py` - Example agent

**Key Components:**
- Task data files: `hal-harness/hal/benchmarks/colbench/data/backend_test.jsonl` and `frontend_test.jsonl`
- Each task has: `problem_description`, `hidden_information`, `test_cases` (backend) or `ground_truth` (frontend)

**Simulated User Prompts:**
```
CODE_USER_PROMPT (for backend):
- "You should make use of the following hidden information to answer the LLM agent"
- "SAY YOU DON'T KNOW IF THE ANSWER CAN NOT BE FOUND IN THE HIDDEN INFORMATION"

HTML_USER_PROMPT (for frontend):
- "Describe briefly how is the image made by the agent is mainly different from the image that the human user wants"
```

## VALID IFE CATEGORIES (can be fixed)

1. **Simulated User Response Issues**: User provides contradictory or incorrect information
2. **Hidden Information Design Issues**: Hidden info contains undiscoverable implementation details
3. **Test Case Issues** (Backend): Test cases verify behavior not specified in task
4. **CLIP Evaluation Issues** (Frontend): Valid designs scored incorrectly
5. **Task Specification Ambiguity**: Problem description is unclear or contradictory

## INVALID FIXES (agent capability issues, NOT IFEs)

- Agent asked vague questions
- Agent wrote buggy code
- Agent misunderstood feedback
- Agent ran out of turns due to inefficient dialogue
- Agent produced wrong output format

## FIX OUTPUT FORMAT

For each task that needs a fix, create: `fixes/{benchmark}/TASK_ID/`
- `instruction_override.json` - Clarified problem description additions
- `evaluation_override.json` - Evaluation harness fixes (tolerances, alternative accepted outputs)
- `simulated_user_override.json` - Fixes to simulated user prompt/behavior
- `README.md` - Explanation of the fix

If NO fix needed (capability issue, not IFE), create README.md explaining why.

## Fix JSON Formats

**instruction_override.json:**
```json
{{
  "clarifications": [
    "Additional instruction text to append to problem_description"
  ],
  "notes": "Why this clarification is needed"
}}
```

**evaluation_override.json:**
```json
{{
  "tolerance": 0.01,
  "accept_alternatives": true,
  "notes": "Why evaluation needs adjustment"
}}
```

**simulated_user_override.json:**
```json
{{
  "additional_prompt": "Additional context for simulated user",
  "allow_explicit_values": true,
  "notes": "Why simulated user behavior needs fixing"
}}
```

## TASKS TO PROCESS

{tasks_text}

## BEGIN

1. First, read `hal-harness/hal/benchmarks/colbench.py` to understand evaluation
2. For each task above:
   a. Analyze the rubric evaluations and conversation logs
   b. Identify if the simulated user gave bad/contradictory feedback
   c. Check if test cases are testing undocumented behavior
   d. Determine if IFE exists or if it's a capability issue
   e. Create appropriate fix (or document why no fix needed)
3. Create fixes in `fixes/{benchmark}/TASK_ID/` directories

Remember: Make evaluation FAIR, not EASY. The collaborative challenge must remain intact.
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
    """Run Claude Code CLI with the given prompt.

    Uses stdin to pass the prompt to avoid 'Argument list too long' errors
    when the prompt contains large conversation logs.
    """

    base_cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", "-",  # Read prompt from stdin
    ]

    if not quiet:
        base_cmd.extend(["--verbose", "--output-format", "stream-json"])
    else:
        base_cmd.extend(["--output-format", "json"])

    log_path = fix_dir / "claude_session.jsonl"

    if not quiet:
        log(f"Running Claude Code CLI for {task_id}...")
        log(f"Working directory: {working_dir}")
        log(f"Prompt size: {len(prompt):,} chars")
        print(f"\n{'='*60}")
        print(f"CLAUDE CODE SESSION: {task_id}")
        print(f"{'='*60}\n")

        process = subprocess.Popen(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Send prompt via stdin and close it
        process.stdin.write(prompt)
        process.stdin.close()

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
        result = subprocess.run(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            input=prompt,  # Pass prompt via stdin
            capture_output=True,
            text=True,
        )

        with log_path.open("w") as log_file:
            log_file.write(result.stdout)
            if result.stderr:
                log_file.write(f"\n--- STDERR ---\n{result.stderr}")

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
            conversations = load_task_conversations(trace_files, task_id)

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
        prompt = build_claude_prompt_batch(
            tasks_data=tasks_data,
            benchmark=benchmark,
        )

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


def process_single_task(
    task_id: str,
    rubric_dir: Path,
    judge_verdicts: Dict[str, Dict[str, Any]],
    trace_files: List[Path],
    benchmark: str,
    quiet: bool = False,
) -> Tuple[str, bool, str]:
    """Process a single task (legacy, wraps batch function)."""
    results = process_task_batch(
        task_ids=[task_id],
        rubric_dir=rubric_dir,
        judge_verdicts=judge_verdicts,
        trace_files=trace_files,
        benchmark=benchmark,
        batch_id=0,
        quiet=quiet,
    )
    return results[0] if results else (task_id, False, "No result")


def main():
    global _total_count, _completed_count

    parser = argparse.ArgumentParser(
        description="Use Claude Code to diagnose and fix ColBench IFEs"
    )

    parser.add_argument(
        "--rubric-dir",
        type=str,
        required=True,
        help="Directory containing rubric CSV outputs (e.g., rubrics_output/colbench)",
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
        default="colbench",
        help="Benchmark name (default: colbench)",
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
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}ColBench IFE Fixer - Claude Code CLI{RESET}")
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
        log(f"  Hint: Trace files are in traces/ with names like colbench_*.json", prefix)

    # Load judge verdicts if provided
    judge_verdicts = {}
    if args.judge_csv:
        judge_path = Path(args.judge_csv)
        if not judge_path.is_absolute():
            judge_path = REPO_ROOT / judge_path
        judge_verdicts = load_judge_verdicts(judge_path)
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
        task_ids = sorted(task_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
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
            if has_existing_fix(args.benchmark, tid):
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
            conversations = load_task_conversations(trace_files, task_id)
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

        prompt = build_claude_prompt_batch(
            tasks_data=tasks_data,
            benchmark=args.benchmark,
        )

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

    # Actually run
    print(f"\n{BOLD}{GREEN}Starting processing...{RESET}\n")

    results = []
    if args.parallel <= 1:
        # Sequential processing
        for batch_id, batch_tasks in enumerate(batches):
            batch_results = process_task_batch(
                task_ids=batch_tasks,
                rubric_dir=rubric_dir,
                judge_verdicts=judge_verdicts,
                trace_files=trace_files,
                benchmark=args.benchmark,
                batch_id=batch_id,
                quiet=False,
            )
            results.extend(batch_results)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_task_batch,
                    batch_tasks,
                    rubric_dir,
                    judge_verdicts,
                    trace_files,
                    args.benchmark,
                    batch_id,
                    True,  # quiet mode for parallel
                ): batch_id
                for batch_id, batch_tasks in enumerate(batches)
            }

            for future in as_completed(futures):
                batch_id = futures[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    for task_id, success, _ in batch_results:
                        log_progress(task_id, "completed" if success else "failed")
                except Exception as e:
                    log(f"Batch {batch_id} exception: {e}")

    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}FINAL SUMMARY{RESET}")
    print(f"{'='*60}")
    print(f"  Processed: {len(results)} tasks")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {len(skipped_tasks)}")
    print(f"  Fixes in: fixes/{args.benchmark}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
