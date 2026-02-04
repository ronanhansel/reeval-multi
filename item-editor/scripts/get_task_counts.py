#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the scripts directory to the path so we can import from run_benchmark_fixes
sys.path.append(os.path.dirname(__file__))
from run_benchmark_fixes import load_benchmark_dataset, get_task_ids_with_fixes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()

    benchmarks = args.benchmarks.split(",")
    total_tasks = 0
    results = []
    
    for b in benchmarks:
        if args.all_tasks:
            dataset = load_benchmark_dataset(b)
            count = len(dataset) if dataset else 0
        else:
            count = len(get_task_ids_with_fixes(b))
        
        results.append((b, count))
        total_tasks += count
        
    mode_str = "fixes only" if not args.all_tasks else "all tasks"
    print(f"\nMode: {mode_str} ({total_tasks} tasks total)")
    
    for b, count in results:
        print(f"  - {b}: {count} tasks")
    print()

if __name__ == "__main__":
    main()