#!/usr/bin/env python3
"""
Unified Claude Code CLI-based Benchmark Fixer

Uses Claude Code CLI (claude -p) to diagnose and fix Intrinsic Formation Errors (IFEs)
across multiple benchmarks.

Supported benchmarks:
- scienceagentbench
- scicode
- corebench (corebench_hard)
- colbench (backend and frontend)

Usage:
    # Fix specific ScienceAgentBench tasks
    python scripts/claude_fixer.py --benchmark scienceagentbench --task-ids 11 74

    # Dry run for SciCode
    python scripts/claude_fixer.py --benchmark scicode --ife-only --dry-run

    # Parallel processing for CoreBench
    python scripts/claude_fixer.py --benchmark corebench --parallel 4 --tasks-per-batch 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration & Constants
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXES_DIR = REPO_ROOT / "result" / "fixes"
TRACES_DIR = REPO_ROOT / "result" / ".hal_data"

# Benchmark name mappings
BENCHMARK_MAP = {
    "scienceagentbench": "scienceagentbench",
    "scicode": "scicode",
    "corebench": "corebench_hard",
    "corebench_hard": "corebench_hard",
    "colbench": "colbench",
}

# Thread-safe progress tracking
_progress_lock = Lock()
_completed_count = 0
_total_count = 0

# =============================================================================
# Logging Utilities
# =============================================================================

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

# =============================================================================
# Data Loading Utilities
# =============================================================================

def has_existing_fix(benchmark: str, task_id: str) -> bool:
    """Check if a task already has a fix."""
    # Handle corebench naming
    bench_dir = BENCHMARK_MAP.get(benchmark, benchmark)
    fix_dir = FIXES_DIR / bench_dir / task_id
    if not fix_dir.exists():
        return False
    fix_files = ["env_override.json", "evaluation_override.json", "input_override.json", "instruction_override.json", "README.md"]
    return any((fix_dir / f).exists() for f in fix_files)


def load_judge_verdicts(verdict_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load judge verdicts from a CSV file."""
    results = {}
    if not verdict_csv.exists():
        return results
    try:
        with verdict_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get("task_id", "")
                try:
                    final_grade = float(row.get("final_grade", 0))
                except (ValueError, TypeError):
                    final_grade = 0
                results[task_id] = {
                    "final_grade": final_grade,
                    "satisfies_rubric": row.get("satisfies_rubric", "0") == "1",
                    "reasoning": row.get("reasoning", ""),
                    "num_evaluations": int(row.get("num_evaluations", 0) or 0),
                    "model_runs": row.get("model_runs", "").split(";"),
                }
    except Exception as e:
        log(f"Warning: Failed to load judge verdicts from {verdict_csv}: {e}")
    return results


def load_all_rubric_evaluations(rubric_dir: Path, task_id: str) -> List[Dict[str, Any]]:
    """Load all rubric evaluations for a task from all CSV files in the rubric directory."""
    evaluations = []
    if not rubric_dir.exists():
        return evaluations
        
    for csv_file in rubric_dir.glob("*.csv"):
        try:
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
        except Exception as e:
            log(f"Warning: Failed to read rubric file {csv_file}: {e}")
    return evaluations


def load_task_conversations(benchmark: str, trace_files: List[Path], task_id: str) -> Dict[str, str]:
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
        config = data.get("config", {})
        model_name = config.get("agent_args", {}).get("model_name", trace_path.stem[:30])
        model_name = model_name.replace("openai/", "").replace("-2025", "")

        # Handle colbench specific naming
        if benchmark == "colbench":
            bench_name = config.get("benchmark_name", "colbench")
            if "frontend" in bench_name:
                model_name = f"{model_name}_frontend"
            else:
                model_name = f"{model_name}_backend"

        lines = []

        # Strategy 1: Check raw_eval_results (common in ScienceAgentBench)
        raw_eval = data.get("raw_eval_results", {})
        if isinstance(raw_eval, dict):
            # Check eval_result sub-dict (some SAB traces)
            task_result = raw_eval.get("eval_result", {}).get(task_id) or raw_eval.get(task_id)
            if task_result:
                if isinstance(task_result, dict):
                    for key, value in task_result.items():
                        if isinstance(value, str) and len(value) > 500:
                            value = value[:500] + "..."
                        lines.append(f"{key}: {value}")
                else:
                    lines.append(str(task_result))

        # Strategy 2: Check raw_logging_results (common in SciCode, CoreBench, ColBench)
        raw_entries = data.get("raw_logging_results", [])
        task_entries = []

        for entry in raw_entries:
            # Check various task ID fields
            entry_task = (
                entry.get("attributes", {}).get("weave_task_id")
                or entry.get("weave_task_id")
                or entry.get("task_id")
                or entry.get("capsule_id")
                or ""
            )
            if str(entry_task) == str(task_id):
                task_entries.append(entry)

        if task_entries:
            # Format conversation entries
            for entry in task_entries[:30]:  # Limit entries
                messages = entry.get("inputs", {}).get("messages", [])
                for msg in messages[-5:]:  # Last few messages per entry
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

        if lines:
            # Join lines, respect per-benchmark limits
            limit = 80 if benchmark == "colbench" else 50
            conversations[model_name] = "\n\n".join(lines[-limit:])

    return conversations

# =============================================================================
# Prompt Building
# =============================================================================

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
    max_convs = 4 if "frontend" in str(conversations.keys()) else 3
    for model, conv in list(conversations.items())[:max_convs]:
        limit = 8000 if "colbench" in str(model) else 5000
        conv_sections.append(f"#### {model}\n```\n{conv[:limit]}\n```")
    conversations_text = "\n".join(conv_sections) if conv_sections else "No logs."

    return f"""
---
## TASK: {task_id}
---

### Rubric Evaluations
{evaluations_text}

### Judge Verdict
{judge_text}

### Model Execution Results / Conversation Logs
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

    # Base prompt common to all benchmarks
    prompt_header = f'''You are diagnosing and fixing Intrinsic Formation Errors (IFEs) in {benchmark} benchmark tasks.

**YOU HAVE {len(task_ids)} TASKS TO PROCESS: {", ".join(task_ids)}**

Process each task sequentially, creating fixes as needed. Be THOROUGH in your analysis.

## CRITICAL CONSTRAINTS - READ CAREFULLY

1. **FIX INTRINSIC FORMATION ERRORS ONLY** - Do NOT make the scientific problem easier
2. **PRESERVE SCIENTIFIC RIGOR** - The task should remain as challenging as intended
3. **NO NERFING** - Do not simplify concepts, give hints, reduce precision, or pre-compute results
4. **VALID FIXES**: Environment packages, Docker config, ambiguous instructions, evaluation tolerance, simulated user behavior
5. **INVALID FIXES**: Solution hints, simplified science, pre-importing specialized modules revealing solution
'''

    # Benchmark-specific context
    if benchmark == "scienceagentbench":
        benchmark_context = '''
## SCIENCEAGENTBENCH HARNESS STRUCTURE
**Read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/scienceagentbench.py` - Main benchmark class
- `hal-harness/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/` - Evaluation harness

**To inspect a specific task, run:**
```python
from datasets import load_dataset
ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
task = ds[int(TASK_ID) - 1]
print(task['task_inst'])
```
'''
    elif benchmark == "scicode":
        benchmark_context = '''
## SCICODE HARNESS STRUCTURE
**Read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/scicode.py` - Main benchmark class
- `hal-harness/hal/benchmarks/SciCode/` - Evaluation utilities

**To inspect a specific task, run:**
```python
from datasets import load_dataset
dataset = load_dataset("SciCode1/SciCode", split="test")
task = [t for t in dataset if t['problem_id'] == 'TASK_ID'][0]
print(task['sub_steps'])
```
'''
    elif benchmark in ("corebench", "corebench_hard"):
        benchmark_context = '''
## COREBENCH HARNESS STRUCTURE
**Read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/corebench.py` - Main benchmark class
- `hal-harness/hal/benchmarks/corebench/core_test.json` - Task definitions
'''
    elif benchmark == "colbench":
        benchmark_context = '''
## COLBENCH HARNESS STRUCTURE
**Read these files to understand the benchmark:**
- `hal-harness/hal/benchmarks/colbench.py` - Main benchmark class
- `hal-harness/hal/benchmarks/colbench/data/` - Task data (backend_test.jsonl, frontend_test.jsonl)

**Simulated User Issues**: Look for cases where the simulated user (GPT-4o) provided contradictory or incorrect info based on `hidden_information`.
'''
    else:
        benchmark_context = ""

    prompt_footer = f'''
## FIX OUTPUT FORMAT

For each task that needs a fix, create: `fixes/{BENCHMARK_MAP.get(benchmark, benchmark)}/TASK_ID/`

- `env_override.json`: Conda/Pip/Apt packages or timeouts
- `instruction_override.json`: Clarifications to task prompt
- `evaluation_override.json`: Numerical tolerance, figure tolerance, alternative formats
- `README.md`: Root cause analysis and justification

If NO fix needed (capability issue, not IFE), create README.md explaining why.

## TASKS TO PROCESS

{tasks_text}

## BEGIN - SYSTEMATIC APPROACH

1. **Understand evaluation pipeline** by reading benchmark code
2. **Load specific task details** from dataset
3. **Analyze error messages** from model execution logs
4. **Cross-reference with other models** - same error = likely IFE
5. **Create fix OR document why no fix needed**
6. **Verify fix doesn't nerf the problem**

Remember: Make evaluation FAIR, not EASY. Preserve scientific rigor.
'''

    return prompt_header + benchmark_context + prompt_footer

# =============================================================================
# Execution
# =============================================================================

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
        "-p", "-",  # Read prompt from stdin to avoid arg length limits
    ]

    if not quiet:
        base_cmd.extend(["--verbose", "--output-format", "stream-json"])
    else:
        base_cmd.extend(["--output-format", "json"])

    log_path = fix_dir / "claude_session.jsonl"

    if not quiet:
        log(f"Running Claude Code CLI for {task_id}...")
        log(f"Working directory: {working_dir}")
        print(f"\n{"="*60}")
        print(f"CLAUDE CODE SESSION: {task_id}")
        print(f"{"="*60}\n")
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
        try:
            # Write prompt to stdin
            process.stdin.write(prompt)
            process.stdin.close()

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
        # Quiet mode
        result = subprocess.run(
            base_cmd,
            cwd=working_dir,
            env={**os.environ, "CLAUDE_CODE_TASK_ID": task_id},
            input=prompt,
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
    """Process a batch of tasks in a single Claude session."""

    bench_dir = BENCHMARK_MAP.get(benchmark, benchmark)
    try:
        # Gather data for all tasks in batch
        tasks_data = []
        for task_id in task_ids:
            evaluations = load_all_rubric_evaluations(rubric_dir, task_id)
            judge_verdict = judge_verdicts.get(task_id)
            conversations = load_task_conversations(benchmark, trace_files, task_id)

            tasks_data.append({
                'task_id': task_id,
                'evaluations': evaluations,
                'judge_verdict': judge_verdict,
                'conversations': conversations,
            })

            # Create fix directory for each task
            fix_dir = FIXES_DIR / bench_dir / task_id
            fix_dir.mkdir(parents=True, exist_ok=True)

        # Build batch prompt
        prompt = build_claude_prompt_batch(tasks_data, benchmark)

        # Save prompt to first task's directory
        batch_fix_dir = FIXES_DIR / bench_dir / task_ids[0]
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


def main():
    global _total_count, _completed_count

    parser = argparse.ArgumentParser(
        description="Unified Claude Code IFE Fixer"
    )

    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        required=True,
        choices=["scienceagentbench", "scicode", "corebench", "colbench"],
        help="Benchmark to process",
    )
    parser.add_argument(
        "--rubric-dir",
        type=str,
        help="Directory containing rubric CSV outputs",
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
        help="Trace files to extract conversations from (optional)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="Specific task IDs to process",
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
        help="Number of tasks per Claude session (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without running Claude Code",
    )
    parser.add_argument(
        "--ife-only",
        action="store_true",
        help="Only process tasks with judge verdict = 1. Requires --judge-csv.",
    )
    parser.add_argument(
        "--list-ife-tasks",
        action="store_true",
        help="List all tasks that have IFEs detected and exit",
    )

    args = parser.parse_args()
    benchmark = args.benchmark

    # Resolve default paths if not provided
    rubric_dir_str = args.rubric_dir or f"result/.hal_data/rubrics_output/{benchmark}"
    rubric_dir = Path(rubric_dir_str)
    if not rubric_dir.is_absolute():
        rubric_dir = REPO_ROOT / rubric_dir

    # Color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"\n{BOLD}{CYAN}{"="*60}{RESET}")
    print(f"{BOLD}{CYAN}Unified IFE Fixer - {benchmark}{RESET}")
    print(f"{BOLD}{CYAN}{"="*60}{RESET}\n")
    log(f"Rubric directory: {rubric_dir}")
    
    trace_files = [Path(f) if Path(f).is_absolute() else REPO_ROOT / f for f in args.trace_files]
    if not trace_files:
        # Try to find default traces
        trace_files = list(TRACES_DIR.glob(f"{benchmark}_*.json"))
        if not trace_files:
            log("No trace files found in traces/ - will use rubric evaluations only")
    
    if trace_files:
        log(f"Using {len(trace_files)} trace files")

    # Load judge verdicts
    judge_verdicts = {}
    if args.judge_csv:
        judge_path = Path(args.judge_csv)
        if not judge_path.is_absolute():
            judge_path = REPO_ROOT / judge_path
        judge_verdicts = load_judge_verdicts(judge_path)
        log(f"Loaded {len(judge_verdicts)} judge verdicts")

    # Find tasks from rubrics
    task_ids_from_rubric = set()
    if rubric_dir.exists():
        for csv_file in rubric_dir.glob("*.csv"):
            try:
                with csv_file.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            grade = float(row.get("grade", 0) or 0)
                        except: grade = 0
                        if grade >= args.min_grade:
                            task_ids_from_rubric.add(row.get("task_id", ""))
            except: pass
    
    # Sort task IDs (natural sort for digits)
    task_ids_from_rubric = sorted(task_ids_from_rubric, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    log(f"Found {len(task_ids_from_rubric)} potential IFEs in rubrics")

    if args.list_ife_tasks:
        print(f"\n{BOLD}IFE Tasks (grade >= {args.min_grade}):{RESET}")
        for tid in task_ids_from_rubric:
            has_fix = has_existing_fix(benchmark, tid)
            verdict = judge_verdicts.get(tid, {}).get("final_grade", "?")
            fix_status = f"{GREEN}[fix exists]{RESET}" if has_fix else ""
            print(f"  - {tid} (verdict: {verdict}) {fix_status}")
        return

    # Determine tasks to process
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = list(task_ids_from_rubric)

    # Filter to confirmed IFEs only
    if args.ife_only:
        if not args.judge_csv:
            print(f"{RED}Error: --ife-only requires --judge-csv{RESET}")
            return
        task_ids = [tid for tid in task_ids if judge_verdicts.get(tid, {}).get("final_grade", 0) == 1]
        log(f"Filtered to {len(task_ids)} confirmed IFEs (verdict=1)")

    # Filter existing fixes
    skipped_tasks = []
    if args.skip_existing:
        original_count = len(task_ids)
        skipped_tasks = [tid for tid in task_ids if has_existing_fix(benchmark, tid)]
        task_ids = [tid for tid in task_ids if tid not in skipped_tasks]
        log(f"Skipping {len(skipped_tasks)} tasks with existing fixes, {len(task_ids)} remaining")

    if args.max_tasks and len(task_ids) > args.max_tasks:
        task_ids = task_ids[:args.max_tasks]

    if not task_ids:
        log(f"{YELLOW}No tasks to process{RESET}")
        return

    # Create batches
    _total_count = len(task_ids) + len(skipped_tasks)
    _completed_count = len(skipped_tasks)
    batches = [task_ids[i:i + args.tasks_per_batch] for i in range(0, len(task_ids), args.tasks_per_batch)]

    if args.dry_run:
        log(f"{YELLOW}DRY RUN MODE - Previewing first batch{RESET}")
        preview_batch = batches[0]
        tasks_data = []
        for tid in preview_batch:
            tasks_data.append({
                'task_id': tid,
                'evaluations': load_all_rubric_evaluations(rubric_dir, tid),
                'judge_verdict': judge_verdicts.get(tid),
                'conversations': load_task_conversations(benchmark, trace_files, tid),
            })
        prompt = build_claude_prompt_batch(tasks_data, benchmark)
        print(f"\n{BOLD}PROMPT PREVIEW (Batch 1):{RESET}\n{"-"*40}\n{prompt[:2000]}...\n{"-"*40}\n")

    # Process batches
    all_results = []
    if args.parallel <= 1:
        for i, batch in enumerate(batches):
            results = process_task_batch(batch, rubric_dir, judge_verdicts, trace_files, benchmark, i, quiet=False)
            all_results.extend(results)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_task_batch, batch, rubric_dir, judge_verdicts, trace_files, benchmark, i, True): i for i, batch in enumerate(batches)}
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
                for tid, success, _ in results:
                    log_progress(tid, "completed" if success else "failed")

    # Final summary
    successful = [r for r in all_results if r[1]]
    failed = [r for r in all_results if not r[1]]

    print(f"\n{BOLD}{GREEN}{"="*60}{RESET}")
    print(f"{BOLD}FINAL SUMMARY{RESET}")
    print(f"{"="*60}")
    print(f"\n{GREEN}Succeeded: {len(successful)}/{len(all_results)}{RESET}")
    if failed:
        print(f"{RED}Failed: {len(failed)}/{len(all_results)}{RESET}")
        for tid, _, msg in failed:
            print(f"  {RED}✗{RESET} {tid}: {msg}")
    print(f"\n{BOLD}Fixes saved to:{RESET} fixes/{BENCHMARK_MAP.get(benchmark, benchmark)}/")
    print(f"{"="*60}\n")
if __name__ == "__main__":
    main()
