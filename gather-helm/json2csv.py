import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- NEW: Metric Priority List ---
# The script will look for these score columns in this order.
# This makes it robust to different benchmarks using different primary metrics.
METRIC_PRIORITY = [
    "exact_match",
    "quasi_exact_match",
    "f1_score",
    "rougeL",
    "bleu_4",
    "accuracy"
]

def find_best_score_column(columns):
    """
    Finds the best available score column from a list of columns based on priority.
    """
    for metric in METRIC_PRIORITY:
        if f"stats.{metric}" in columns:
            return f"stats.{metric}"
    # Fallback: if no priority metrics are found, take the first 'stats.' column
    for col in columns:
        if str(col).startswith("stats."):
            return col
    return None

def process_run(path, required_files):
    """
    Safely loads, processes, and combines data for a single benchmark run.
    """
    try:
        json_data = {}
        for file_name in required_files:
            file_path = path / file_name
            with open(file_path, "rb") as f:
                json_data[file_name] = json.load(f)

        d_requests = pd.json_normalize(json_data["display_requests.json"])
        d_predictions = pd.json_normalize(json_data["display_predictions.json"])
        run_specs_list = json.load(open(path / "run_spec.json"))
        run_spec = run_specs_list[0] if isinstance(run_specs_list, list) else run_specs_list
        run_specs = pd.json_normalize(run_spec)
        instances = pd.json_normalize(json_data["instances.json"])

        benchmark = path.parts[-3]
        run_specs["benchmark"] = benchmark
        if not d_predictions.empty:
            run_specs = run_specs.loc[run_specs.index.repeat(d_predictions.shape[0])].reset_index(drop=True)

        result = pd.concat([d_requests, d_predictions, run_specs, instances], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        result["scenario"] = result['name'].str.split(r'[:,]', n=1, expand=True)[0]

        # --- Dynamic Score Discovery ---
        score_col = find_best_score_column(result.columns)
        if score_col:
            result["dicho_score"] = result[score_col]
        else:
            result["dicho_score"] = pd.NA
            
        return result

    except Exception:
        return None

if __name__ == "__main__":
    input_dir = Path("./helm_jsons")
    output_dir = Path("./data")
    BENCHMARKS = [
        "torr", "speech", "robo-reward-bench", "medhelm", "image2struct", "finance",
        "ewok", "capabilities", "call-center", "audio", "vhelm", "thaiexam",
        "safety", "image2structure", "instruct", "heim", "decodingtrust",
        "cleva", "air-bench", "mmlu", "classic", "lite"
    ]

    print("Finding all benchmark run paths...")
    all_paths = []
    for benchmark in BENCHMARKS:
        release_dir = input_dir / benchmark / "releases"
        
        if release_dir.exists():
            release_dirs = sorted([p.name for p in release_dir.iterdir() if p.is_dir()])
            if not release_dirs: continue
            latest_release = release_dirs[-1]
            folder_dict_path = release_dir / latest_release / "runs_to_run_suites.json"
            if folder_dict_path.exists():
                with open(folder_dict_path, "r") as f:
                    folder_dict = json.load(f)
                all_paths.extend([input_dir / benchmark / "runs" / s / r for r, s in folder_dict.items()])

    required_files = ["display_requests.json", "display_predictions.json", "run_spec.json", "instances.json"]
    valid_paths = [p for p in tqdm(all_paths, desc="Verifying paths") if all((p / f).exists() for f in required_files)]
    
    if not valid_paths:
        print("❌ Error: No complete benchmark runs found.")
        exit()
        
    print(f"\nFound {len(valid_paths)} complete runs to process.")

    # --- FINAL, SMARTER PRE-FLIGHT CHECK ---
    print("\n🕵️  Performing final pre-flight check...")
    try:
        sample_path = valid_paths[0]
        print(f"   - Using sample run: {sample_path}")

        # Load all necessary sample files
        sample_requests_cols = pd.json_normalize(json.load(open(sample_path / "display_requests.json"))).columns
        sample_instances_cols = pd.json_normalize(json.load(open(sample_path / "instances.json"))).columns
        sample_predictions_cols = pd.json_normalize(json.load(open(sample_path / "display_predictions.json"))).columns
        
        # Check essential columns
        if 'request.model' not in sample_requests_cols:
            raise KeyError("'request.model' not found in display_requests.json")
        if 'input.text' not in sample_instances_cols:
            raise KeyError("'input.text' not in instances.json")

        # Check for *any* valid score column
        best_score_col = find_best_score_column(sample_predictions_cols)
        if not best_score_col:
            stats_cols_found = [col for col in sample_predictions_cols if str(col).startswith('stats.')]
            raise ValueError(f"Could not find any of the priority metrics in display_predictions.json. Priority list: {METRIC_PRIORITY}. Stats columns found: {stats_cols_found}")

        print(f"✅ Pre-flight check passed. Found a valid score column: '{best_score_col}'")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ PRE-FLIGHT CHECK FAILED.")
        print(f"   - Error: {e}")
        print("="*80)
        exit()
    
    # --- MAIN PROCESSING ---
    print("\nStarting main processing of all runs...")
    results = [res for path in tqdm(valid_paths, desc="Processing runs") if (res := process_run(path, required_files)) is not None]

    if not results:
        print("\n❌ No data was successfully processed.")
    else:
        print("\nConcatenating all results...")
        results_df = pd.concat(results, axis=0, join='outer', ignore_index=True)
        
        essential_cols = ["request.model", "input.text", "scenario", "benchmark", "dicho_score"]
        final_cols = [col for col in essential_cols if col in results_df.columns]
        results_df = results_df[final_cols]

        print("Saving final DataFrame to pickle file...")
        output_dir.mkdir(exist_ok=True)
        results_df.to_pickle(output_dir / "long.pkl")
        print(f"\n✅ Done! Final data shape: {results_df.shape}. Saved to '{output_dir / 'long.pkl'}'")