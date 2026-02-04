import json
import os
from pathlib import Path
# moon18, sun17, sun18, sun19

def reconstruct_trace(upload_path, raw_path):
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
                            # If still a string or parsing failed, skip it
                            continue
                    else:
                        # Skip error strings
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
    except Exception as e:
        print(f"  ❌ Failed to fix {upload_path.name}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix ColBench traces for a specific prefix.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix to match (e.g., moon18, sun17)")
    args = parser.parse_args()
    
    traces_dir = Path("eval_traces/traces")
    raw_dir = Path("eval_traces/raw_submission")
    
    # Pattern to match UPLOAD files with the given prefix
    # Example: colbench_colbench_moon18_gpt-5_colbench_example_agent_..._UPLOAD.json
    upload_pattern = f"colbench_colbench_{args.prefix}_*_colbench_example_agent_*_UPLOAD.json"
    
    found_files = list(traces_dir.glob(upload_pattern))
    
    if not found_files:
        print(f"No files found matching prefix: {args.prefix}")
        return

    print(f"Found {len(found_files)} trace files for prefix '{args.prefix}'")

    for upload_path in found_files:
        # derive raw path from upload path
        # Replace _UPLOAD.json with _RAW_SUBMISSIONS.jsonl
        filename = upload_path.name
        raw_filename = filename.replace("_UPLOAD.json", "_RAW_SUBMISSIONS.jsonl")
        raw_path = raw_dir / raw_filename
        
        if raw_path.exists():
            reconstruct_trace(upload_path, raw_path)
        else:
            print(f"  ⚠️  Missing raw submissions file for {filename}: {raw_path}")

if __name__ == "__main__":
    main()
