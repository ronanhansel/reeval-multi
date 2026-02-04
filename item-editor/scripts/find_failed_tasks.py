#!/usr/bin/env python3
"""
Find Failed Tasks Utility

This script analyzes benchmark runs to find:
1. Tasks that failed due to external errors (API timeout, Docker crash, auth errors)
2. Tasks that were never executed (missing from results)
3. Generate commands to rerun only the failed tasks

Usage:
    python scripts/find_failed_tasks.py --benchmark scicode --run-id <run_id>
    python scripts/find_failed_tasks.py --scan-results  # Scan all results
    python scripts/find_failed_tasks.py --scan-logs logs/benchmark_run_*  # Scan log files
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Patterns that indicate external failures (not agent capability issues)
EXTERNAL_ERROR_PATTERNS = [
    r"401.*unauthorized",
    r"invalid.*token",
    r"token.*expired",
    r"connection.*reset",
    r"connection.*refused",
    r"timeout|timed out",
    r"503|504|502|500",
    r"429.*rate.limit",
    r"docker.*error",
    r"container.*failed",
    r"oom|out.of.memory",
    r"killed",
    r"segmentation.fault",
    r"cannot.connect",
    r"network.*error",
    r"ssl.*error",
    r"certificate.*error",
]

EXTERNAL_ERROR_REGEX = re.compile("|".join(EXTERNAL_ERROR_PATTERNS), re.IGNORECASE)


def find_results_dirs(base_path: str = "results") -> Dict[str, List[Path]]:
    """Find all results directories organized by benchmark."""
    results = {}
    base = Path(base_path)
    if not base.exists():
        return results

    for benchmark_dir in base.iterdir():
        if benchmark_dir.is_dir():
            benchmark = benchmark_dir.name
            results[benchmark] = list(benchmark_dir.iterdir())

    return results


def analyze_trace_file(trace_path: Path) -> Dict:
    """Analyze a trace file for failures."""
    with open(trace_path) as f:
        data = json.load(f)

    analysis = {
        "path": str(trace_path),
        "config": data.get("config", {}),
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": [],
        "external_errors": [],
        "missing_tasks": [],
    }

    # Get expected tasks from config or raw_eval_results
    raw_eval = data.get("raw_eval_results", {})

    if isinstance(raw_eval, dict):
        analysis["total_tasks"] = len(raw_eval)
        analysis["completed_tasks"] = len(raw_eval)

        # Check for tasks with zero scores or errors
        for task_id, result in raw_eval.items():
            if isinstance(result, dict):
                # Check for error indicators
                if result.get("error") or result.get("exception"):
                    analysis["failed_tasks"].append({
                        "task_id": task_id,
                        "error": result.get("error") or result.get("exception"),
                    })

    # Check raw_logging_results for errors
    raw_logging = data.get("raw_logging_results", [])
    if isinstance(raw_logging, list):
        for entry in raw_logging:
            if isinstance(entry, dict):
                # Check for errors in the entry
                error_text = str(entry.get("error", "")) + str(entry.get("exception", ""))
                if error_text and EXTERNAL_ERROR_REGEX.search(error_text):
                    task_id = entry.get("task_id", "unknown")
                    analysis["external_errors"].append({
                        "task_id": task_id,
                        "error": error_text[:200],
                    })

    return analysis


def scan_log_file(log_path: Path) -> Dict:
    """Scan a log file for errors and failed tasks."""
    analysis = {
        "path": str(log_path),
        "external_errors": [],
        "failed_tasks": [],
        "task_status": {},
    }

    with open(log_path, errors="ignore") as f:
        content = f.read()

    # Find task starts and completions
    task_starts = re.findall(r"\[(\w+)\].*(?:Starting|Running).*task[:\s]+(\S+)", content, re.IGNORECASE)
    task_completes = re.findall(r"\[(\w+)\].*(?:SUCCESS|COMPLETED).*task[:\s]+(\S+)", content, re.IGNORECASE)
    task_failures = re.findall(r"\[(\w+)\].*(?:FAILED|ERROR).*task[:\s]+(\S+)", content, re.IGNORECASE)

    # Find external errors
    for match in EXTERNAL_ERROR_REGEX.finditer(content):
        # Get context around the match
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end].replace("\n", " ")

        # Try to extract task ID from context
        task_match = re.search(r"task[:\s_-]*(\S+)", context, re.IGNORECASE)
        task_id = task_match.group(1) if task_match else "unknown"

        analysis["external_errors"].append({
            "task_id": task_id,
            "error": match.group()[:100],
            "context": context[:200],
        })

    # Track task status
    for _, task_id in task_starts:
        analysis["task_status"][task_id] = "started"
    for _, task_id in task_completes:
        analysis["task_status"][task_id] = "completed"
    for _, task_id in task_failures:
        analysis["task_status"][task_id] = "failed"
        analysis["failed_tasks"].append(task_id)

    return analysis


def scan_results_directory(results_dir: Path) -> Dict:
    """Scan a results directory for task outputs and identify missing/failed tasks."""
    analysis = {
        "path": str(results_dir),
        "run_id": results_dir.name,
        "completed_tasks": [],
        "failed_tasks": [],
        "errors": [],
    }

    # Look for output files
    for item in results_dir.iterdir():
        if item.is_dir():
            # Task subdirectory
            task_id = item.name
            output_file = item / "output.json"
            if output_file.exists():
                try:
                    with open(output_file) as f:
                        output = json.load(f)
                    if output.get("error"):
                        analysis["failed_tasks"].append(task_id)
                        analysis["errors"].append({
                            "task_id": task_id,
                            "error": str(output.get("error"))[:200],
                        })
                    else:
                        analysis["completed_tasks"].append(task_id)
                except Exception as e:
                    analysis["failed_tasks"].append(task_id)
                    analysis["errors"].append({
                        "task_id": task_id,
                        "error": f"Failed to read output: {e}",
                    })
            else:
                # No output file - task didn't complete
                analysis["failed_tasks"].append(task_id)

    # Also check verbose logs
    for log_file in results_dir.glob("*_verbose.log"):
        with open(log_file, errors="ignore") as f:
            content = f.read()

        for match in EXTERNAL_ERROR_REGEX.finditer(content):
            analysis["errors"].append({
                "task_id": "from_log",
                "error": match.group()[:100],
            })

    return analysis


def generate_rerun_command(
    benchmark: str,
    task_ids: List[str],
    prefix: str = "rerun_",
    model: Optional[str] = None,
) -> str:
    """Generate command to rerun specific failed tasks."""

    # For HAL, we need to use --continue_run with the same run_id
    # Or create a filtered dataset with only the failed tasks

    cmd_parts = [
        "python scripts/run_benchmark_fixes.py",
        f"--benchmark {benchmark}",
        f"--prefix {prefix}",
        "--docker",
    ]

    if model:
        cmd_parts.append(f"--model {model}")

    # Add task IDs (if supported)
    if len(task_ids) <= 10:
        task_list = ",".join(task_ids)
        cmd_parts.append(f"# --task-ids {task_list}  # NOTE: May need custom filtering")
    else:
        cmd_parts.append(f"# {len(task_ids)} tasks to rerun - consider using --continue_run")

    return " \\\n    ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="Find failed tasks in benchmark runs")
    parser.add_argument("--benchmark", help="Specific benchmark to analyze")
    parser.add_argument("--run-id", help="Specific run ID to analyze")
    parser.add_argument("--scan-results", action="store_true", help="Scan all results directories")
    parser.add_argument("--scan-logs", nargs="*", help="Log directories/files to scan")
    parser.add_argument("--trace-file", help="Specific trace file to analyze")
    parser.add_argument("--results-dir", default="results", help="Results base directory")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--generate-rerun", action="store_true", help="Generate rerun commands")

    args = parser.parse_args()

    all_analyses = []

    # Analyze specific trace file
    if args.trace_file:
        trace_path = Path(args.trace_file)
        if trace_path.exists():
            print(f"\n{'='*60}")
            print(f"Analyzing trace: {trace_path}")
            print('='*60)
            analysis = analyze_trace_file(trace_path)
            all_analyses.append(analysis)

            print(f"Total tasks: {analysis['total_tasks']}")
            print(f"Completed: {analysis['completed_tasks']}")

            if analysis["external_errors"]:
                print(f"\nExternal errors found ({len(analysis['external_errors'])}):")
                for err in analysis["external_errors"][:10]:
                    print(f"  - Task {err['task_id']}: {err['error']}")

    # Scan results directories
    if args.scan_results or args.benchmark:
        results_dirs = find_results_dirs(args.results_dir)

        for benchmark, run_dirs in results_dirs.items():
            if args.benchmark and benchmark != args.benchmark:
                continue

            print(f"\n{'='*60}")
            print(f"Benchmark: {benchmark}")
            print('='*60)

            for run_dir in sorted(run_dirs)[-5:]:  # Last 5 runs
                if args.run_id and args.run_id not in run_dir.name:
                    continue

                print(f"\n  Run: {run_dir.name}")
                analysis = scan_results_directory(run_dir)
                all_analyses.append(analysis)

                print(f"    Completed: {len(analysis['completed_tasks'])}")
                print(f"    Failed: {len(analysis['failed_tasks'])}")

                if analysis["errors"]:
                    print(f"    Errors ({len(analysis['errors'])}):")
                    for err in analysis["errors"][:5]:
                        print(f"      - {err['task_id']}: {err['error'][:80]}")

                if args.generate_rerun and analysis["failed_tasks"]:
                    print(f"\n    Rerun command:")
                    cmd = generate_rerun_command(benchmark, analysis["failed_tasks"])
                    print(f"    {cmd}")

    # Scan log files
    if args.scan_logs:
        print(f"\n{'='*60}")
        print("Scanning log files")
        print('='*60)

        for log_path in args.scan_logs:
            log_path = Path(log_path)

            if log_path.is_dir():
                log_files = list(log_path.glob("*.log"))
            else:
                log_files = [log_path]

            for log_file in log_files:
                print(f"\n  Log: {log_file.name}")
                analysis = scan_log_file(log_file)
                all_analyses.append(analysis)

                if analysis["external_errors"]:
                    print(f"    External errors ({len(analysis['external_errors'])}):")
                    for err in analysis["external_errors"][:5]:
                        print(f"      - Task {err['task_id']}: {err['error']}")

                if analysis["failed_tasks"]:
                    print(f"    Failed tasks ({len(analysis['failed_tasks'])}): {analysis['failed_tasks'][:10]}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    total_external_errors = sum(len(a.get("external_errors", [])) for a in all_analyses)
    total_failed = sum(len(a.get("failed_tasks", [])) for a in all_analyses)

    print(f"Total external errors found: {total_external_errors}")
    print(f"Total failed tasks: {total_failed}")

    if total_external_errors > 0:
        print("\nThese failures are likely recoverable by rerunning.")
        print("Use --generate-rerun to get rerun commands.")

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_analyses, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
