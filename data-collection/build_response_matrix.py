#!/usr/bin/env python3
import os
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

# Configuration
TRACES_DIR = Path("traces")
OUTPUT_DIR = Path("output")

# Map of benchmark prefixes to friendly names (folder names)
# Order matters: longer prefixes first if overlaps exist
BENCHMARKS = [
    "scicode", 
    "colbench", 
    "swebench", 
    "gaia", 
    "assistantbench", 
    "usaco", 
    "seeact", 
    "corebench", 
    "taubench", 
    "scienceagentbench", 
    "browser-use"
]

def get_benchmark_name(filename):
    filename = filename.lower()
    for b in BENCHMARKS:
        if filename.startswith(b):
            return b
    return "other"

def safe_div(n, d):
    return n / d if d else 0.0

def main():
    if not TRACES_DIR.exists():
        print(f"Error: {TRACES_DIR} does not exist.")
        return

    # Data structures
    # benchmark -> set of all unique task_ids encountered
    all_task_ids = defaultdict(set)
    
    # benchmark -> list of agent_data
    # agent_data = {
    #   'agent': str (filename),
    #   'binary': dict[task_id, int],  # 1 or 0
    #   'subscores': dict[score_type, dict[task_id, float]]
    # }
    agent_results = defaultdict(list)

    files = sorted([f for f in TRACES_DIR.glob("*.json") if not f.name.endswith(".encrypted")])
    
    print(f"Found {len(files)} JSON files in {TRACES_DIR}...")

    for f in files:
        benchmark = get_benchmark_name(f.name)
        agent_name = f.stem.replace("_UPLOAD", "")
        
        print(f"Processing {benchmark}: {f.name}...")
        
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")
            continue

        binary_map = {}
        subscores_map = defaultdict(dict)
        
        # ---
        # Extraction Logic
        # ---
        
        # 1. Scicode
        if benchmark == "scicode":
            # Binary
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            
            # Record IDs
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0
            
            # Subscores: Complex to get without metadata, skipping for safety/accuracy
            
        # 2. Colbench
        elif benchmark == "colbench":
            # raw_eval_results is a list of scores [0.0, 1.0, ...]
            raw = data.get("raw_eval_results", [])
            if isinstance(raw, list):
                for idx, score in enumerate(raw):
                    tid = str(idx)
                    all_task_ids[benchmark].add(tid)
                    
                    try:
                        score_f = float(score)
                        binary_map[tid] = 1 if score_f >= 0.999 else 0
                        subscores_map["raw_score"][tid] = score_f
                    except:
                        pass

        # 3. ScienceAgentBench
        elif benchmark == "scienceagentbench":
            # raw_eval_results['eval_result'] is dict {task_id: {success_rate, ...}}
            raw = data.get("raw_eval_results", {})
            eval_res = raw.get("eval_result", {})
            if isinstance(eval_res, dict):
                for tid, metrics in eval_res.items():
                    tid = str(tid)
                    all_task_ids[benchmark].add(tid)
                    
                    sr = metrics.get("success_rate", 0)
                    binary_map[tid] = 1 if sr == 1 else 0
                    
                    subscores_map["success_rate"][tid] = sr
                    subscores_map["codebert_score"][tid] = metrics.get("codebert_score")
                    subscores_map["valid_program"][tid] = metrics.get("valid_program")

        # 4. Corebench
        elif benchmark == "corebench":
            # Binary
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0
            
            # Subscores from raw_eval_results
            # {task_id: {correct_written_answers, total_...}}
            raw = data.get("raw_eval_results", {})
            if isinstance(raw, dict):
                for tid, metrics in raw.items():
                    tid = str(tid)
                    if isinstance(metrics, dict):
                        w_corr = metrics.get("correct_written_answers", 0)
                        w_tot = metrics.get("total_written_questions", 0)
                        v_corr = metrics.get("correct_vision_answers", 0)
                        v_tot = metrics.get("total_vision_questions", 0)
                        
                        if w_tot > 0:
                            subscores_map["written_score"][tid] = w_corr / w_tot
                        if v_tot > 0:
                            subscores_map["vision_score"][tid] = v_corr / v_tot

        # 5. AssistantBench
        elif benchmark == "assistantbench":
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0
            # Skipping subscores due to ambiguous mapping

        # 6. Generic/Others (swebench, gaia, usaco, seeact, taubench, browser-use)
        else:
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            
            # Some benchmarks like 'gaia' might have level_X_accuracy but per-task score is usually binary
            
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0

        # Save extracted data
        agent_entry = {
            "agent": agent_name,
            "binary": binary_map,
            "subscores": subscores_map
        }
        agent_results[benchmark].append(agent_entry)

    # ---
    # Writing Output
    # ---
    
    for bench, agents in agent_results.items():
        if not agents:
            continue
            
        bench_dir = OUTPUT_DIR / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine column order (Sorted Task IDs)
        # Try to sort numerically if possible
        raw_ids = list(all_task_ids[bench])
        try:
            sorted_ids = sorted(raw_ids, key=lambda x: int(x))
        except:
            sorted_ids = sorted(raw_ids)
            
        # 1. Benchmark Binary CSV
        csv_path = bench_dir / "benchmark.csv"
        header = ["agent"] + sorted_ids
        
        print(f"Writing {csv_path} with {len(sorted_ids)} columns...")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            for ag in agents:
                row = [ag["agent"]]
                for tid in sorted_ids:
                    # Look up score
                    val = ag["binary"].get(tid)
                    row.append(str(val) if val is not None else "")
                writer.writerow(row)

        # 2. Subscore CSVs
        # Collect all subscore types for this benchmark
        all_subtypes = set()
        for ag in agents:
            all_subtypes.update(ag["subscores"].keys())
            
        for subtype in all_subtypes:
            sub_csv_path = bench_dir / f"{subtype}.csv"
            print(f"Writing {sub_csv_path}...")
            
            with open(sub_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                
                for ag in agents:
                    row = [ag["agent"]]
                    scores_for_type = ag["subscores"].get(subtype, {})
                    for tid in sorted_ids:
                        val = scores_for_type.get(tid)
                        row.append(str(val) if val is not None else "")
                    writer.writerow(row)

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
