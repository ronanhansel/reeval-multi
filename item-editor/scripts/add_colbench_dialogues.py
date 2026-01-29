#!/usr/bin/env python3
"""
Add dialogue history from ColBench results to merged traces for rubric evaluation.

Usage:
    python scripts/add_colbench_dialogues.py \
        traces/col_tommy_openai_gpt-4_1_MERGED_UPLOAD.json \
        --results-dir results/colbench_backend_programming \
        --run-pattern "col_tommy_gpt-4_1-2025-04-14_*" \
        --output traces/col_tommy_openai_gpt-4_1_WITH_DIALOGUES.json
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, Any, List

def find_task_output(results_dir: Path, run_pattern: str, task_id: str) -> Path | None:
    """Find the output.json file for a specific task.

    ColBench single-task runs always use subdirectory "0" for the task data.
    The actual task ID is in the run directory name.
    Pattern: results_dir/run_pattern_TASKID_*/0/output.json
    """
    # Replace * in pattern with task_id
    # e.g., "col_tommy_gpt-4_1-2025-04-14_*" -> "col_tommy_gpt-4_1-2025-04-14_1_*"
    task_pattern = run_pattern.replace("_*", f"_{task_id}_*")
    pattern = str(results_dir / task_pattern / "0" / "output.json")
    matches = glob.glob(pattern)

    if matches:
        return Path(matches[0])
    return None

def load_dialogue_history(output_path: Path) -> Dict[str, Any] | None:
    """Load dialogue history and task data from output.json."""
    try:
        with open(output_path) as f:
            data = json.load(f)
        
        # ColBench output format: {"0": {"answer": ..., "dialogue_history": ..., "task": ...}}
        # The key is always "0" for single-task runs
        task_data = data.get("0", {})
        
        return {
            "answer": task_data.get("answer", ""),
            "dialogue_history": task_data.get("dialogue_history", []),
            "task": task_data.get("task", {})
        }
    except Exception as e:
        print(f"Warning: Failed to load {output_path}: {e}")
        return None

def add_dialogues_to_trace(
    trace_path: Path,
    results_dir: Path,
    run_pattern: str,
    output_path: Path
) -> None:
    """Add dialogue history to merged trace for rubric evaluation."""
    # Load merged trace
    with open(trace_path) as f:
        trace = json.load(f)
    
    results_dir = Path(results_dir)
    
    # Get raw_eval_results dict (task_id -> score)
    raw_eval = trace.get("raw_eval_results", {})
    if not isinstance(raw_eval, dict):
        print("Error: raw_eval_results is not a dict. Run merge_traces.py first.")
        return
    
    # Create raw_logging_results with dialogue history
    raw_logging_results = []
    
    print(f"Processing {len(raw_eval)} tasks...")
    found = 0
    missing = 0
    
    for task_id, score in sorted(raw_eval.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        # Find output.json for this task
        output_path_found = find_task_output(results_dir, run_pattern, task_id)
        
        if output_path_found:
            dialogue_data = load_dialogue_history(output_path_found)
            if dialogue_data:
                raw_logging_results.append({
                    "task_id": task_id,
                    "score": score,
                    "answer": dialogue_data["answer"],
                    "dialogue_history": dialogue_data["dialogue_history"],
                    "task": dialogue_data["task"]
                })
                found += 1
            else:
                missing += 1
        else:
            task_pattern = run_pattern.replace("_*", f"_{task_id}_*")
            print(f"  Warning: No output found for task {task_id} (pattern: {task_pattern}/0/output.json)")
            missing += 1
    
    print(f"\nFound dialogue history for {found}/{len(raw_eval)} tasks ({missing} missing)")
    
    # Update trace
    trace["raw_logging_results"] = raw_logging_results
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)
    
    print(f"Wrote {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Add dialogue history to ColBench merged traces")
    parser.add_argument("trace_file", type=Path, help="Merged trace file (from merge_traces.py)")
    parser.add_argument("--results-dir", type=Path, required=True, help="Results directory")
    parser.add_argument("--run-pattern", required=True, help="Run directory pattern (e.g., 'col_tommy_gpt-4_1-2025-04-14_*')")
    parser.add_argument("--output", type=Path, required=True, help="Output trace file with dialogues")
    
    args = parser.parse_args()
    
    add_dialogues_to_trace(
        args.trace_file,
        args.results_dir,
        args.run_pattern,
        args.output
    )

if __name__ == "__main__":
    main()
