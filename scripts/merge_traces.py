#!/usr/bin/env python3
"""
Merge multiple HAL trace JSON files into a single trace that can be re-graded.

Supports both corebench and scicode trace formats.

Example (corebench):
    python scripts/merge_traces.py \
        --input 'traces/capsule-*20260113*_UPLOAD.json' \
        --output traces/corebench_hard_hal_generalist_agentgpt41_20260113_FIXED.json

Example (scicode):
    python scripts/merge_traces.py \
        --input 'traces/reeval_scicode_tomato_*gpt-4*_UPLOAD.json' \
        --output traces/scicode_tomato_gpt4_MERGED_UPLOAD.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge HAL trace JSON files.")
    parser.add_argument(
        "--input",
        dest="patterns",
        action="append",
        required=True,
        help="Glob pattern for trace files (can be repeated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file for the merged trace.",
    )
    parser.add_argument(
        "--run-id",
        help="Override the run_id stored in the merged trace (default: output filename stem).",
    )
    parser.add_argument(
        "--agent-name",
        help="Override the agent_name stored in the merged trace.",
    )
    parser.add_argument(
        "--date",
        help="Override the date stored in the merged trace (default: first trace date).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def expand_patterns(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matched = [Path(p).resolve() for p in glob.glob(pattern)]
        files.extend(sorted(matched))
    unique_files: List[Path] = []
    seen = set()
    for file_path in files:
        if file_path not in seen:
            unique_files.append(file_path)
            seen.add(file_path)
    return unique_files


def load_trace(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_scicode_trace(data: Dict[str, Any]) -> bool:
    """Detect if a trace is from scicode benchmark."""
    # Check for scicode-specific structure
    raw_eval = data.get("raw_eval_results") or {}
    if "details" in raw_eval:
        return True
    # Check for subtask_accuracy in results
    results = data.get("results") or {}
    if "subtask_accuracy" in results:
        return True
    # Check benchmark_name in config
    config = data.get("config") or {}
    if config.get("benchmark_name", "").startswith("scicode"):
        return True
    return False


def is_colbench_trace(data: Dict[str, Any]) -> bool:
    """Detect if a trace is from colbench benchmark."""
    config = data.get("config") or {}
    benchmark_name = config.get("benchmark_name", "")
    return benchmark_name.startswith("colbench")


def extract_colbench_task_id(run_id: str) -> Optional[str]:
    """Extract task ID from ColBench run_id.

    Examples:
        col_tommy_gpt-4_1-2025-04-14_1_colbench_backend_programming -> '1'
        col_tommy_o3-2025-04-16_low_292_colbench_backend_programming -> '292'
    """
    import re
    # Pattern: prefix_model_taskid_colbench_...
    # Task ID is the number right before 'colbench'
    match = re.search(r'_(\d+)_colbench_', run_id)
    if match:
        return match.group(1)
    return None


def merge_scicode_traces(
    paths: List[Path],
    *,
    run_id: Optional[str],
    agent_name: Optional[str],
    run_date: Optional[str],
) -> Dict[str, Any]:
    """Merge scicode-specific trace files."""
    first_trace = load_trace(paths[0])
    config = json.loads(json.dumps(first_trace.get("config", {})))
    if run_id:
        config["run_id"] = run_id
    else:
        config.setdefault("run_id", Path(paths[0]).stem)
    if agent_name:
        config["agent_name"] = agent_name
    if run_date:
        config["date"] = run_date
    config["source_traces"] = [Path(p).name for p in paths]

    # Use sets for deduplication
    successful_tasks: Set[str] = set()
    failed_tasks: Set[str] = set()
    merged_latencies: Dict[str, Any] = {}
    total_cost = 0.0
    total_usage: Dict[str, Dict[str, float]] = {}
    raw_logging_results: List[Any] = []

    # Scicode-specific: merge details dict
    merged_details: Dict[str, List[str]] = {}

    for path in paths:
        data = load_trace(path)
        results = data.get("results", {})

        # Add tasks to sets (automatic deduplication)
        for task in results.get("successful_tasks") or []:
            successful_tasks.add(str(task))
        for task in results.get("failed_tasks") or []:
            failed_tasks.add(str(task))

        # Merge latencies
        if results.get("latencies"):
            for key, value in results["latencies"].items():
                if isinstance(value, dict):
                    existing = merged_latencies.setdefault(
                        key,
                        {
                            "first_call_timestamp": value.get("first_call_timestamp"),
                            "last_call_timestamp": value.get("last_call_timestamp"),
                        },
                    )
                    first_ts = value.get("first_call_timestamp")
                    last_ts = value.get("last_call_timestamp")
                    if first_ts and (
                        not existing.get("first_call_timestamp")
                        or first_ts < existing["first_call_timestamp"]
                    ):
                        existing["first_call_timestamp"] = first_ts
                    if last_ts and (
                        not existing.get("last_call_timestamp")
                        or last_ts > existing["last_call_timestamp"]
                    ):
                        existing["last_call_timestamp"] = last_ts
                else:
                    merged_latencies[key] = merged_latencies.get(key, 0) + value

        # Merge logging results
        source_logging = data.get("raw_logging_results") or results.get("raw_logging_results")
        if source_logging:
            raw_logging_results.extend(source_logging)

        total_cost += results.get("total_cost") or data.get("total_cost") or 0.0

        # Merge usage
        usage = data.get("total_usage") or {}
        for model_name, usage_stats in usage.items():
            bucket = total_usage.setdefault(
                model_name, {"input_tokens": 0, "output_tokens": 0}
            )
            bucket["input_tokens"] += usage_stats.get("input_tokens", 0)
            bucket["output_tokens"] += usage_stats.get("output_tokens", 0)

        # Merge scicode details
        raw_eval = data.get("raw_eval_results") or {}
        details = raw_eval.get("details") or {}
        for task_id, subtasks in details.items():
            task_id_str = str(task_id)
            if task_id_str not in merged_details:
                merged_details[task_id_str] = []
            # Merge subtask lists, keeping unique subtasks
            existing_subtasks = set(merged_details[task_id_str])
            for subtask in subtasks:
                if subtask not in existing_subtasks:
                    merged_details[task_id_str].append(subtask)
                    existing_subtasks.add(subtask)

    # Remove successful tasks from failed tasks (in case of duplicates across traces)
    failed_tasks -= successful_tasks

    # Sort task lists for consistent output
    successful_list = sorted(successful_tasks, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
    failed_list = sorted(failed_tasks, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))

    # Calculate accuracies
    total_tasks = len(successful_list) + len(failed_list)
    accuracy = len(successful_list) / total_tasks if total_tasks else 0.0

    # Calculate subtask_accuracy from merged details
    total_subtasks = 0
    successful_subtasks = 0
    for task_id, subtasks in merged_details.items():
        successful_subtasks += len(subtasks)
        # Count expected subtasks based on task_id pattern
        # Subtasks are named like "58.1", "58.2", "58.3" for task 58
        # We need to determine total expected subtasks per task
        # For now, we'll track what we have and note this may need adjustment

    # To calculate subtask_accuracy properly, we need to know total expected subtasks
    # We can estimate from the benchmark or use successful + failed subtask count
    # For scicode, each task has varying number of subtasks
    # We'll calculate based on what's in the details
    for task_id in set(list(merged_details.keys()) + successful_list + failed_list):
        # Count subtasks for this task based on naming convention
        task_subtasks = merged_details.get(str(task_id), [])
        # Get max subtask number to estimate total
        max_subtask = 0
        for subtask in task_subtasks:
            if "." in subtask:
                try:
                    subtask_num = int(subtask.split(".")[-1])
                    max_subtask = max(max_subtask, subtask_num)
                except ValueError:
                    pass
        # If we found subtasks, use max as total; otherwise assume at least 1
        if max_subtask > 0:
            total_subtasks += max_subtask
        elif str(task_id) in merged_details:
            # Task exists in details but no subtasks passed
            total_subtasks += 1  # Assume at least 1 subtask

    # Recalculate with actual counts from details
    total_subtasks = 0
    successful_subtasks = 0
    for task_id, subtasks in merged_details.items():
        successful_subtasks += len(subtasks)

    # Get total subtasks from benchmark knowledge (scicode has 288 total subtasks across 80 tasks)
    # But for a subset, we estimate based on task range
    # A simpler approach: use the ratio if we have partial data
    # For merged traces, just report what we have
    subtask_accuracy = 0.0
    if total_subtasks > 0:
        subtask_accuracy = successful_subtasks / total_subtasks
    elif successful_subtasks > 0:
        # If we have successful subtasks but no total, report raw count
        # This happens when merging partial results
        # Use 288 as scicode total subtasks
        subtask_accuracy = successful_subtasks / 288

    merged_results: Dict[str, Any] = {
        "successful_tasks": successful_list,
        "failed_tasks": failed_list,
        "latencies": merged_latencies,
        "accuracy": accuracy,
        "subtask_accuracy": subtask_accuracy,
        "total_cost": total_cost,
    }

    merged: Dict[str, Any] = {
        "config": config,
        "results": merged_results,
        "raw_eval_results": {"details": merged_details},
        "raw_logging_results": raw_logging_results,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "git_info": first_trace.get("git_info"),
    }
    return merged


def merge_colbench_traces(
    paths: List[Path],
    *,
    run_id: Optional[str],
    agent_name: Optional[str],
    run_date: Optional[str],
) -> Dict[str, Any]:
    """Merge ColBench-specific trace files.

    ColBench traces have:
    - raw_eval_results: list of scores (one per task)
    - Task ID embedded in run_id
    - results.average_correctness and results.accuracy
    """
    first_trace = load_trace(paths[0])
    config = json.loads(json.dumps(first_trace.get("config", {})))
    if run_id:
        config["run_id"] = run_id
    else:
        config.setdefault("run_id", Path(paths[0]).stem)
    if agent_name:
        config["agent_name"] = agent_name
    if run_date:
        config["date"] = run_date
    config["source_traces"] = [Path(p).name for p in paths]

    # ColBench: Build task_id -> score mapping
    task_scores: Dict[str, float] = {}
    successful_tasks: Set[str] = set()
    failed_tasks: Set[str] = set()
    total_cost = 0.0
    total_usage: Dict[str, Dict[str, float]] = {}
    raw_logging_results: List[Any] = []
    merged_latencies: Dict[str, Any] = {}

    for path in paths:
        data = load_trace(path)
        trace_config = data.get("config", {})
        trace_run_id = trace_config.get("run_id", "")
        results = data.get("results", {})

        # Extract task ID from run_id
        task_id = extract_colbench_task_id(trace_run_id)

        # Get score from raw_eval_results (list with one element for single-task traces)
        raw_eval = data.get("raw_eval_results") or []
        if isinstance(raw_eval, list) and len(raw_eval) > 0:
            score = float(raw_eval[0])
            if task_id:
                task_scores[task_id] = score
                if score == 1.0:
                    successful_tasks.add(task_id)
                else:
                    failed_tasks.add(task_id)

        # Merge latencies
        if results.get("latencies"):
            for key, value in results["latencies"].items():
                if isinstance(value, dict):
                    existing = merged_latencies.setdefault(
                        key,
                        {
                            "first_call_timestamp": value.get("first_call_timestamp"),
                            "last_call_timestamp": value.get("last_call_timestamp"),
                        },
                    )
                    first_ts = value.get("first_call_timestamp")
                    last_ts = value.get("last_call_timestamp")
                    if first_ts and (
                        not existing.get("first_call_timestamp")
                        or first_ts < existing["first_call_timestamp"]
                    ):
                        existing["first_call_timestamp"] = first_ts
                    if last_ts and (
                        not existing.get("last_call_timestamp")
                        or last_ts > existing["last_call_timestamp"]
                    ):
                        existing["last_call_timestamp"] = last_ts
                else:
                    merged_latencies[key] = merged_latencies.get(key, 0) + value

        # Merge logging results
        source_logging = data.get("raw_logging_results") or results.get("raw_logging_results")
        if source_logging:
            raw_logging_results.extend(source_logging)

        total_cost += results.get("total_cost") or data.get("total_cost") or 0.0

        # Merge usage
        usage = data.get("total_usage") or {}
        for model_name, usage_stats in usage.items():
            bucket = total_usage.setdefault(
                model_name, {"input_tokens": 0, "output_tokens": 0}
            )
            bucket["input_tokens"] += usage_stats.get("input_tokens", 0)
            bucket["output_tokens"] += usage_stats.get("output_tokens", 0)

    # Remove successful tasks from failed tasks (in case of duplicates)
    failed_tasks -= successful_tasks

    # Sort task lists numerically
    successful_list = sorted(successful_tasks, key=lambda x: int(x) if x.isdigit() else float('inf'))
    failed_list = sorted(failed_tasks, key=lambda x: int(x) if x.isdigit() else float('inf'))

    # Calculate ColBench metrics
    all_scores = list(task_scores.values())
    total_tasks = len(all_scores)
    average_correctness = sum(all_scores) / total_tasks if total_tasks else 0.0
    accuracy = len(successful_list) / total_tasks if total_tasks else 0.0

    # Build raw_eval_results as dict (task_id -> score) for easier rubric grading
    # Also include as list for compatibility
    raw_eval_results_dict: Dict[str, float] = task_scores
    raw_eval_results_list: List[float] = [
        task_scores.get(str(i), 0.0) for i in range(max(int(k) for k in task_scores.keys()) + 1)
    ] if task_scores else []

    merged_results: Dict[str, Any] = {
        "successful_tasks": successful_list,
        "failed_tasks": failed_list,
        "latencies": merged_latencies,
        "average_correctness": average_correctness,
        "accuracy": accuracy,
        "total_cost": total_cost,
    }

    merged: Dict[str, Any] = {
        "config": config,
        "results": merged_results,
        "raw_eval_results": raw_eval_results_dict,  # Dict format for rubric grading
        "raw_eval_results_list": raw_eval_results_list,  # List format for compatibility
        "raw_logging_results": raw_logging_results,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "git_info": first_trace.get("git_info"),
    }
    return merged


def merge_corebench_traces(
    paths: List[Path],
    *,
    run_id: Optional[str],
    agent_name: Optional[str],
    run_date: Optional[str],
) -> Dict[str, Any]:
    """Merge corebench-specific trace files."""
    first_trace = load_trace(paths[0])
    config = json.loads(json.dumps(first_trace.get("config", {})))
    if run_id:
        config["run_id"] = run_id
    else:
        config.setdefault("run_id", Path(paths[0]).stem)
    if agent_name:
        config["agent_name"] = agent_name
    if run_date:
        config["date"] = run_date
    config["source_traces"] = [Path(p).name for p in paths]

    # Use sets for deduplication
    successful_tasks: Set[str] = set()
    failed_tasks: Set[str] = set()
    merged_latencies: Dict[str, Any] = {}
    total_cost = 0.0
    total_usage: Dict[str, Dict[str, float]] = {}
    raw_eval_results: Dict[str, Any] = {}
    raw_logging_results: List[Any] = []
    written_correct = vision_correct = 0
    written_total = vision_total = 0

    for path in paths:
        data = load_trace(path)
        results = data.get("results", {})

        # Add tasks to sets (automatic deduplication)
        for task in results.get("successful_tasks") or []:
            successful_tasks.add(str(task))
        for task in results.get("failed_tasks") or []:
            failed_tasks.add(str(task))

        if results.get("latencies"):
            for key, value in results["latencies"].items():
                if isinstance(value, dict):
                    existing = merged_latencies.setdefault(
                        key,
                        {
                            "first_call_timestamp": value.get("first_call_timestamp"),
                            "last_call_timestamp": value.get("last_call_timestamp"),
                        },
                    )
                    first_ts = value.get("first_call_timestamp")
                    last_ts = value.get("last_call_timestamp")
                    if first_ts and (
                        not existing.get("first_call_timestamp")
                        or first_ts < existing["first_call_timestamp"]
                    ):
                        existing["first_call_timestamp"] = first_ts
                    if last_ts and (
                        not existing.get("last_call_timestamp")
                        or last_ts > existing["last_call_timestamp"]
                    ):
                        existing["last_call_timestamp"] = last_ts
                else:
                    merged_latencies[key] = merged_latencies.get(key, 0) + value

        source_logging = data.get("raw_logging_results") or results.get("raw_logging_results")
        if source_logging:
            raw_logging_results.extend(source_logging)
        total_cost += results.get("total_cost") or data.get("total_cost") or 0.0

        usage = data.get("total_usage") or {}
        for model_name, usage_stats in usage.items():
            bucket = total_usage.setdefault(
                model_name, {"input_tokens": 0, "output_tokens": 0}
            )
            bucket["input_tokens"] += usage_stats.get("input_tokens", 0)
            bucket["output_tokens"] += usage_stats.get("output_tokens", 0)

        capsule_eval = data.get("raw_eval_results") or {}
        for task_id, stats in capsule_eval.items():
            raw_eval_results[task_id] = stats
            if isinstance(stats, dict):
                written_correct += stats.get("correct_written_answers", 0)
                vision_correct += stats.get("correct_vision_answers", 0)
                written_total += stats.get("total_written_questions", 0)
                vision_total += stats.get("total_vision_questions", 0)

    # Remove successful tasks from failed tasks
    failed_tasks -= successful_tasks

    # Sort task lists
    successful_list = sorted(successful_tasks)
    failed_list = sorted(failed_tasks)

    total_tasks = len(successful_list) + len(failed_list)

    merged_results: Dict[str, Any] = {
        "successful_tasks": successful_list,
        "failed_tasks": failed_list,
        "latencies": merged_latencies,
        "accuracy": len(successful_list) / total_tasks if total_tasks else 0.0,
        "written_accuracy": written_correct / written_total if written_total else 0.0,
        "vision_accuracy": vision_correct / vision_total if vision_total else 0.0,
        "total_cost": total_cost,
    }

    merged: Dict[str, Any] = {
        "config": config,
        "results": merged_results,
        "raw_eval_results": raw_eval_results,
        "raw_logging_results": raw_logging_results,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "git_info": first_trace.get("git_info"),
    }
    return merged


def merge_traces(
    paths: List[Path],
    *,
    run_id: Optional[str],
    agent_name: Optional[str],
    run_date: Optional[str],
) -> Dict[str, Any]:
    """Merge trace files, auto-detecting colbench vs scicode vs corebench format."""
    # Check first trace to determine type
    first_trace = load_trace(paths[0])

    if is_colbench_trace(first_trace):
        return merge_colbench_traces(
            paths, run_id=run_id, agent_name=agent_name, run_date=run_date
        )
    elif is_scicode_trace(first_trace):
        return merge_scicode_traces(
            paths, run_id=run_id, agent_name=agent_name, run_date=run_date
        )
    else:
        return merge_corebench_traces(
            paths, run_id=run_id, agent_name=agent_name, run_date=run_date
        )


def main() -> None:
    args = parse_args()
    input_files = expand_patterns(args.patterns)
    if not input_files:
        raise SystemExit("No trace files matched the provided --input pattern(s).")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output file {args.output} already exists. Use --force to overwrite.")

    run_id = args.run_id or args.output.stem
    merged = merge_traces(
        input_files,
        run_id=run_id,
        agent_name=args.agent_name,
        run_date=args.date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged {len(input_files)} traces into {args.output}")


if __name__ == "__main__":
    main()
