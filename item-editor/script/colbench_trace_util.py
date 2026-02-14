#!/usr/bin/env python3
"""
Unified ColBench Trace Utility

Provides tools to:
1. Fix corrupted or incomplete ColBench traces by reconstructing from raw submissions.
2. Add dialogue history from result directories to merged traces.
"""

import argparse
import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# =============================================================================
# Trace Reconstruction (from fix_colbench_traces.py)
# =============================================================================

def reconstruct_trace(upload_path: Path, raw_path: Path) -> bool:
    """Reconstruct a trace file from raw submissions JSONL."""
    print(f"Processing {upload_path.name}...")
    try:
        with open(upload_path) as f:
            upload_data = json.load(f)
        
        reconstructed_logging = []
        with open(raw_path) as f:
            for line in f:
                task_obj = json.loads(line)
                task_id = next(iter(task_obj.keys()))
                task_data = task_obj[task_id]
                
                if isinstance(task_data, str):
                    # Robustness: attempt to extract JSON dictionary from string if it looks like one
                    if "{" in task_data:
                        try:
                            # Find first { and last }
                            start = task_data.find("{")
                            end = task_data.rfind("}") + 1
                            task_data = json.loads(task_data[start:end])
                        except:
                            continue
                    else:
                        continue

                reconstructed_logging.append({
                    "task_id": task_id,
                    "dialogue_history": task_data.get("dialogue_history", []),
                    "answer": task_data.get("answer", ""),
                    "task": task_data.get("task", {})
                })
        
        upload_data["raw_logging_results"] = reconstructed_logging
        
        with open(upload_path, "w") as f:
            json.dump(upload_data, f, indent=2)
        
        print(f"  ✅ Fixed {upload_path.name} with {len(reconstructed_logging)} entries.")
        return True
    except Exception as e:
        print(f"  ❌ Failed to fix {upload_path.name}: {e}")
        return False

# =============================================================================
# Dialogue Injection (from add_colbench_dialogues.py)
# =============================================================================

def find_task_output(results_dir: Path, run_pattern: str, task_id: str) -> Optional[Path]:
    """Find the output.json file for a specific task."""
    # Replace * in pattern with task_id
    task_pattern = run_pattern.replace("*", f"{task_id}_*")
    pattern = str(results_dir / task_pattern / "0" / "output.json")
    matches = glob.glob(pattern)

    if matches:
        return Path(matches[0])
    return None

def load_dialogue_history(output_path: Path) -> Optional[Dict[str, Any]]:
    """Load dialogue history and task data from output.json."""
    try:
        with open(output_path) as f:
            data = json.load(f)
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
    with open(trace_path) as f:
        trace = json.load(f)
    
    raw_eval = trace.get("raw_eval_results", {})
    if not isinstance(raw_eval, dict):
        print("Error: raw_eval_results is not a dict. Run merge_traces.py first.")
        return
    
    raw_logging_results = []
    found = 0
    missing = 0
    
    task_ids = sorted(raw_eval.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    print(f"Processing {len(task_ids)} tasks for dialogue injection...")
    
    for task_id in task_ids:
        score = raw_eval[task_id]
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
                continue
        
        missing += 1
    
    print(f"Injection complete: Found {found}, Missing {missing}")
    trace["raw_logging_results"] = raw_logging_results
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Wrote updated trace to {output_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified ColBench Trace Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: reconstruct
    recon_parser = subparsers.add_parser("reconstruct", help="Reconstruct traces from raw submissions")
    recon_parser.add_argument("--prefix", type=str, required=True, help="Prefix to match (e.g., moon18)")
    recon_parser.add_argument("--traces-dir", type=Path, default=Path("result/.hal_data"), help="Traces directory")
    recon_parser.add_argument("--raw-dir", type=Path, default=Path("result/.hal_data/raw_submission"), help="Raw submissions directory")

    # Command: inject-dialogues
    inject_parser = subparsers.add_parser("inject-dialogues", help="Inject dialogue history from results")
    inject_parser.add_argument("trace_file", type=Path, help="Merged trace file")
    inject_parser.add_argument("--results-dir", type=Path, required=True, help="Results directory")
    inject_parser.add_argument("--run-pattern", required=True, help="Run directory pattern (e.g., 'col_tommy_gpt-4_1-2025-04-14_*')")
    inject_parser.add_argument("--output", type=Path, required=True, help="Output trace file")

    args = parser.parse_args()

    if args.command == "reconstruct":
        upload_pattern = f"colbench_colbench_{args.prefix}_*_colbench_example_agent_*_UPLOAD.json"
        found_files = list(args.traces_dir.glob(upload_pattern))
        
        if not found_files:
            print(f"No files found matching prefix: {args.prefix}")
            return

        for upload_path in found_files:
            raw_filename = upload_path.name.replace("_UPLOAD.json", "_RAW_SUBMISSIONS.jsonl")
            raw_path = args.raw_dir / raw_filename
            if raw_path.exists():
                reconstruct_trace(upload_path, raw_path)
            else:
                print(f"  ⚠️  Missing raw submissions file for {upload_path.name}")

    elif args.command == "inject-dialogues":
        add_dialogues_to_trace(args.trace_file, args.results_dir, args.run_pattern, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
