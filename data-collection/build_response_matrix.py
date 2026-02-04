#!/usr/bin/env python3
import os
import json
import csv
import sys
import re
from pathlib import Path
from collections import defaultdict

# Configuration
TRACES_DIR = Path("traces")
OUTPUT_DIR = Path("output")

# Fallback prefixes if config is missing
# Order matters: longer prefixes first if overlaps exist
FALLBACK_PREFIXES = [
    "scicode", 
    "colbench_backend_programming",
    "colbench_frontend_design",
    "colbench", 
    "swebench_verified_mini",
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

def get_benchmark_name_from_filename(filename):
    filename = filename.lower()
    for b in FALLBACK_PREFIXES:
        if filename.startswith(b):
            return b
    return "other"

def normalize_text(text):
    if not text:
        return ""
    # Lowercase, strip
    text = text.lower().strip()
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s\-]+', '_', text)
    # Remove any other non-alphanumeric chars (except underscore and forward slash)
    text = re.sub(r'[^a-z0-9_/]', '', text)
    return text

def get_scaffold_name(raw_name):
    # 1. Remove content within parenthesis
    name = re.sub(r'\s*\(.*?\)', '', raw_name)
    
    # 2. Check for generalist keywords
    if any(k in name.lower() for k in ["general", "hal generalist", "generalist agent"]):
        return "hal_generalist"
    
    # 3. Normalize
    return normalize_text(name)

def get_model_info(config):
    model_name = ""
    reasoning = ""
    
    # Extract model name from top level
    for k in ["model", "model_name", "model_id"]:
        if k in config and config[k]:
            model_name = str(config[k])
            break
            
    # Extract from agent_args if not found or to find reasoning
    agent_args = config.get("agent_args", {})
    if isinstance(agent_args, dict):
        if not model_name:
            for k in ["model_name", "model", "model_id", "agent.model.name", "model_name"]:
                if k in agent_args and agent_args[k]:
                    model_name = str(agent_args[k])
                    break
            # Fallback search
            if not model_name:
                for k, v in agent_args.items():
                    if "model" in k and "name" in k and v:
                        model_name = str(v)
                        break
        
        # Extract reasoning effort
        # Look for keys ending in reasoning_effort or exactly reasoning_effort
        for k, v in agent_args.items():
            if k == "reasoning_effort" or k.endswith(".reasoning_effort"):
                if v:
                    reasoning = str(v)
                break
                
    return model_name, reasoning

def construct_agent_identifier(config, filename_stem):
    raw_agent_name = config.get("agent_name", filename_stem)
    scaffold = get_scaffold_name(raw_agent_name)
    
    model_name, reasoning = get_model_info(config)
    
    # Normalize model name (remove provider prefixes like openrouter/ etc if desired, 
    # but instruction says just normalize text)
    # Often model names have slashes 'openai/gpt-4', let's replace slashes with underscore too for safety in CSV
    clean_model = normalize_text(model_name)
    
    # Append reasoning if present
    if reasoning:
        clean_reasoning = normalize_text(reasoning)
        full_model_part = f"{clean_model}_{clean_reasoning}"
    else:
        full_model_part = clean_model
        
    if not full_model_part:
        full_model_part = "unknown_model"
        
    return f"{scaffold}__{full_model_part}"

def main():
    if not TRACES_DIR.exists():
        print(f"Error: {TRACES_DIR} does not exist.")
        return

    # Data structures
    all_task_ids = defaultdict(set)
    
    # benchmark -> unique_id -> results_data
    agent_results = defaultdict(lambda: defaultdict(lambda: {
        'binary': {},
        'subscores': defaultdict(dict)
    }))

    files = sorted([f for f in TRACES_DIR.glob("*.json") if not f.name.endswith(".encrypted")])
    
    print(f"Found {len(files)} JSON files in {TRACES_DIR}...")

    for f in files:
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")
            continue

        config = data.get("config", {})
        
        # Determine Benchmark Name
        benchmark = config.get("benchmark_name")
        if not benchmark:
            benchmark = get_benchmark_name_from_filename(f.name)
            print(f"  [Warning] No benchmark_name in config for {f.name}. Fallback to: {benchmark}")
        benchmark = benchmark.lower()
        
        # Construct Unique Agent Identifier
        unique_id = construct_agent_identifier(config, f.stem.replace("_UPLOAD", ""))
            
        print(f"Processing {benchmark}: {unique_id} ({f.name})")

        binary_map = {}
        subscores_map = defaultdict(dict)
        
        # --- Extraction Logic ---
        
        # 1. Scicode
        if "scicode" in benchmark:
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0
            
        # 2. Colbench
        elif "colbench" in benchmark:
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
            elif isinstance(raw, dict):
                for tid, score in raw.items():
                    all_task_ids[benchmark].add(tid)
                    try:
                        score_f = float(score)
                        binary_map[tid] = 1 if score_f >= 0.999 else 0
                        subscores_map["raw_score"][tid] = score_f
                    except:
                        pass

        # 3. ScienceAgentBench
        elif "scienceagentbench" in benchmark:
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
        elif "corebench" in benchmark:
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0
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

        # 5. Default / Generic
        else:
            results = data.get("results", {})
            succ = set(map(str, results.get("successful_tasks", [])))
            fail = set(map(str, results.get("failed_tasks", [])))
            tasks = succ | fail
            all_task_ids[benchmark].update(tasks)
            for t in tasks:
                binary_map[t] = 1 if t in succ else 0

        # Merge results into agent_results using unique identifier
        target_agent = agent_results[benchmark][unique_id]
        
        # Merge binary
        for tid, val in binary_map.items():
            if tid not in target_agent['binary'] or val > target_agent['binary'][tid]:
                target_agent['binary'][tid] = val
        
        # Merge subscores
        for subtype, scores in subscores_map.items():
            for tid, val in scores.items():
                if val is not None:
                    if tid not in target_agent['subscores'][subtype] or val > target_agent['subscores'][subtype][tid]:
                        target_agent['subscores'][subtype][tid] = val

    # --- Writing Output ---
    
    for bench, agents_dict in agent_results.items():
        if not agents_dict:
            continue
            
        bench_dir = OUTPUT_DIR / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine column order
        raw_ids = list(all_task_ids[bench])
        try:
            sorted_ids = sorted(raw_ids, key=lambda x: int(x))
        except:
            sorted_ids = sorted(raw_ids)
            
        header = ["agent"] + [f"{bench}.{tid}" for tid in sorted_ids]
        
        # 1. Benchmark Binary CSV
        csv_path = bench_dir / "benchmark.csv"
        print(f"Writing {csv_path} with {len(sorted_ids)} columns...")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for unique_id, ag_data in sorted(agents_dict.items()):
                row = [unique_id]
                for tid in sorted_ids:
                    val = ag_data["binary"].get(tid)
                    row.append(str(val) if val is not None else "")
                writer.writerow(row)

        # 2. Subscore CSVs
        all_subtypes = set()
        for ag_data in agents_dict.values():
            all_subtypes.update(ag_data["subscores"].keys())
            
        for subtype in all_subtypes:
            sub_csv_path = bench_dir / f"{subtype}.csv"
            print(f"Writing {sub_csv_path}...")
            with open(sub_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for unique_id, ag_data in sorted(agents_dict.items()):
                    row = [unique_id]
                    scores_for_type = ag_data["subscores"].get(subtype, {})
                    for tid in sorted_ids:
                        val = scores_for_type.get(tid)
                        row.append(str(val) if val is not None else "")
                    writer.writerow(row)

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()