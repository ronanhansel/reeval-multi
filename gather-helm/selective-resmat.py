import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Configuration for metric selection and thresholds
ACCURACY_METRICS = [
    "exact_match",
    "quasi_exact_match", 
    "f1_score",
    "compilation_success",
    "edit_similarity",
    "pixel_similarity",
    "fid_similarity",
    "ssim_similarity",
    "accuracy",
    # Finance-specific metrics
    "annotation_financebench_label_correct_answer",
    "execution_accuracy", 
    "program_accuracy"
]

# Thresholds for converting similarity metrics to dichotomy scores
SIMILARITY_THRESHOLDS = {
    "compilation_success": 0.5,  # Already binary-like
    "edit_similarity": 0.8,
    "pixel_similarity": 0.7,
    "fid_similarity": 0.7,
    "ssim_similarity": 0.7,
    "exact_match": 0.5,  # Already binary
    "quasi_exact_match": 0.5,  # Already binary
    "f1_score": 0.8,
    "accuracy": 0.5,  # Already binary
    # Finance-specific thresholds
    "annotation_financebench_label_correct_answer": 0.5,  # Binary-like annotation
    "execution_accuracy": 0.5,  # Already binary-like
    "program_accuracy": 0.5   # Already binary-like
}

def find_best_accuracy_metric(stats_data: List[Dict]) -> Optional[str]:
    """
    Find the best available accuracy metric from the stats data.
    Returns the metric name if found, None otherwise.
    Searches across all splits (valid, test) to handle different benchmark formats.
    """
    available_metrics = set()
    for stat in stats_data:
        if 'name' in stat and 'name' in stat['name']:
            metric_name = stat['name']['name']
            split = stat['name'].get('split', '')
            # Include metrics from both valid and test splits
            if split in ['valid', 'test']:
                available_metrics.add(metric_name)
    
    # Find the highest priority metric that's available
    for metric in ACCURACY_METRICS:
        if metric in available_metrics:
            return metric
    
    return None

def extract_metric_value(stats_data: List[Dict], metric_name: str, split: str = "valid") -> Optional[float]:
    """
    Extract the mean value for a specific metric from stats data.
    Tries both 'valid' and 'test' splits to handle different benchmark formats.
    """
    # Try the requested split first
    for stat in stats_data:
        if (stat.get('name', {}).get('name') == metric_name and 
            stat.get('name', {}).get('split') == split and 
            'perturbation' not in stat.get('name', {})):
            return stat.get('mean', None)
    
    # If not found and we tried 'valid', try 'test' split (for finance dataset)
    if split == "valid":
        for stat in stats_data:
            if (stat.get('name', {}).get('name') == metric_name and 
                stat.get('name', {}).get('split') == "test" and 
                'perturbation' not in stat.get('name', {})):
                return stat.get('mean', None)
    
    # If not found and we tried 'test', try 'valid' split (for image datasets)
    if split == "test":
        for stat in stats_data:
            if (stat.get('name', {}).get('name') == metric_name and 
                stat.get('name', {}).get('split') == "valid" and 
                'perturbation' not in stat.get('name', {})):
                return stat.get('mean', None)
    
    return None

def convert_to_dichotomy(value: float, metric_name: str) -> int:
    """
    Convert a continuous metric value to dichotomy score (0 or 1).
    """
    if value is None:
        return 0
    
    threshold = SIMILARITY_THRESHOLDS.get(metric_name, 0.5)
    return 1 if value >= threshold else 0

def process_single_run(run_path: Path) -> Optional[Dict[str, Any]]:
    """
    Process a single run directory and extract accuracy information.
    """
    try:
        # Check if required files exist
        stats_file = run_path / "stats.json"
        run_spec_file = run_path / "run_spec.json"
        
        if not stats_file.exists() or not run_spec_file.exists():
            return None
        
        # Load stats data
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        
        # Load run spec for metadata
        with open(run_spec_file, 'r') as f:
            run_spec_data = json.load(f)
            if isinstance(run_spec_data, list):
                run_spec_data = run_spec_data[0]
        
        # Extract run name and model info
        run_name = run_spec_data.get('name', run_path.name)
        model_name = run_spec_data.get('adapter_spec', {}).get('model', 'unknown')
        
        # Parse scenario info from run name
        scenario_parts = run_name.split(':')
        scenario = scenario_parts[0] if scenario_parts else 'unknown'
        
        # Extract difficulty and subset if available
        difficulty = 'unknown'
        subset = 'unknown'
        
        for part in scenario_parts[1:]:
            if part.startswith('difficulty='):
                difficulty = part.split('=')[1]
            elif part.startswith('subset='):
                subset = part.split('=')[1]
        
        # Find the best available accuracy metric
        best_metric = find_best_accuracy_metric(stats_data)
        
        if best_metric is None:
            return None
        
        # Extract the metric value
        metric_value = extract_metric_value(stats_data, best_metric)
        
        if metric_value is None:
            return None
        
        # Convert to dichotomy score
        dichotomy_score = convert_to_dichotomy(metric_value, best_metric)
        
        return {
            'run_name': run_name,
            'model': model_name,
            'scenario': scenario,
            'subset': subset,
            'difficulty': difficulty,
            'metric_used': best_metric,
            'metric_value': metric_value,
            'dichotomy_score': dichotomy_score,
            'run_path': str(run_path)
        }
        
    except Exception as e:
        print(f"Error processing {run_path}: {e}")
        return None

def collect_all_runs(base_path: str) -> pd.DataFrame:
    """
    Collect accuracy information from all HELM benchmark runs.
    """
    base_path = Path(base_path)
    
    # Find all run directories
    run_dirs = []
    benchmark_stats = {}
    
    # Look for run directories in the typical HELM structure
    if (base_path / "helm_jsons").exists():
        helm_base = base_path / "helm_jsons"
        for benchmark_dir in helm_base.iterdir():
            if not benchmark_dir.is_dir():
                continue
            
            benchmark_name = benchmark_dir.name
            benchmark_stats[benchmark_name] = 0
            
            runs_dir = benchmark_dir / "runs"
            if runs_dir.exists():
                for version_dir in runs_dir.iterdir():
                    if version_dir.is_dir():
                        # Look for individual run directories
                        for run_dir in version_dir.iterdir():
                            if run_dir.is_dir() and (run_dir / "stats.json").exists():
                                run_dirs.append(run_dir)
                                benchmark_stats[benchmark_name] += 1
    
    print(f"Found {len(run_dirs)} run directories across benchmarks:")
    for benchmark, count in benchmark_stats.items():
        if count > 0:
            print(f"  {benchmark}: {count} runs")
    
    # Process all runs
    results = []
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        result = process_single_run(run_dir)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nProcessed {len(results)} runs successfully!")
    print(f"Unique scenarios: {df['scenario'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Average dichotomy score: {df['dichotomy_score'].mean():.3f}")
    
    # Show breakdown by scenario
    print(f"\nBreakdown by scenario:")
    scenario_counts = df['scenario'].value_counts()
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count} runs")
    
    return df

def analyze_accuracy_by_groups(df: pd.DataFrame) -> None:
    """
    Analyze accuracy patterns by different groupings.
    """
    print("\n" + "="*60)
    print("ACCURACY ANALYSIS BY GROUPS")
    print("="*60)
    
    # By model
    print("\nAccuracy by Model:")
    model_stats = df.groupby('model')['dichotomy_score'].agg(['count', 'mean', 'std']).round(3)
    model_stats = model_stats.sort_values('mean', ascending=False)
    print(model_stats.head(10))
    
    # By scenario
    print("\nAccuracy by Scenario:")
    scenario_stats = df.groupby('scenario')['dichotomy_score'].agg(['count', 'mean', 'std']).round(3)
    scenario_stats = scenario_stats.sort_values('mean', ascending=False)
    print(scenario_stats.head(10))
    
    # By difficulty if available
    if 'difficulty' in df.columns and df['difficulty'].nunique() > 1:
        print("\nAccuracy by Difficulty:")
        difficulty_stats = df.groupby('difficulty')['dichotomy_score'].agg(['count', 'mean', 'std']).round(3)
        difficulty_stats = difficulty_stats.sort_values('mean', ascending=False)
        print(difficulty_stats)
    
    # By metric used
    print("\nAccuracy by Metric Type:")
    metric_stats = df.groupby('metric_used')['dichotomy_score'].agg(['count', 'mean', 'std']).round(3)
    metric_stats = metric_stats.sort_values('mean', ascending=False)
    print(metric_stats)

def create_multiindex_pivot(df: pd.DataFrame, output_dir: str = "./data") -> pd.DataFrame:
    """
    Create a MultiIndex DataFrame with scenarios as columns and models as index.
    Each cell contains the dichotomy score for that model-scenario combination.
    """
    # Clean up the difficulty column (remove ',model' suffix)
    df_clean = df.copy()
    df_clean['difficulty'] = df_clean['difficulty'].str.replace(',model', '')
    
    # Create pivot table with model as index and scenario combinations as columns
    pivot_df = df_clean.pivot_table(
        index='model',
        columns=['scenario', 'subset', 'difficulty'],
        values='dichotomy_score',
        aggfunc='first'  # Use first value if there are duplicates
    )
    
    # Sort columns and index for better organization
    pivot_df = pivot_df.sort_index(axis=1)
    pivot_df = pivot_df.sort_index(axis=0)
    
    # Save as pickle file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pkl_path = output_path / "helm_accuracy_multiindex.pkl"
    pivot_df.to_pickle(pkl_path)
    
    print(f"\nMultiIndex DataFrame saved to {pkl_path}")
    print(f"Shape: {pivot_df.shape} (models x scenario combinations)")
    print(f"Models: {len(pivot_df.index)}")
    print(f"Scenario combinations: {len(pivot_df.columns)}")
    
    # Also save as CSV for inspection 
    csv_path = output_path / "helm_accuracy_multiindex.csv"
    pivot_df.to_csv(csv_path)
    print(f"Also saved as CSV to {csv_path}")
    
    # Print sample of the MultiIndex structure
    print("\nSample of MultiIndex DataFrame:")
    print("Columns (scenario, subset, difficulty):", pivot_df.columns[:5].tolist())
    print("First 5 models and 3 scenario combinations:")
    print(pivot_df.iloc[:5, :3])
    
    # Print some statistics
    print(f"\nDataFrame Statistics:")
    print(f"- Non-null values: {pivot_df.count().sum()}")
    print(f"- Null values: {pivot_df.isnull().sum().sum()}")
    print(f"- Overall mean score: {pivot_df.mean().mean():.3f}")
    
    return pivot_df

def create_detailed_multiindex_pivot(df: pd.DataFrame, output_dir: str = "./data") -> pd.DataFrame:
    """
    Create a detailed MultiIndex DataFrame with both metric values and dichotomy scores.
    """
    df_clean = df.copy()
    df_clean['difficulty'] = df_clean['difficulty'].str.replace(',model', '')
    
    # Create separate pivots for metric values and dichotomy scores
    metric_pivot = df_clean.pivot_table(
        index='model',
        columns=['scenario', 'subset', 'difficulty'],
        values='metric_value',
        aggfunc='first'
    )
    
    dichotomy_pivot = df_clean.pivot_table(
        index='model',
        columns=['scenario', 'subset', 'difficulty'],
        values='dichotomy_score',
        aggfunc='first'
    )
    
    # Combine into a single DataFrame with four-level column index
    combined_data = {}
    
    for col in metric_pivot.columns:
        # Add metric values
        combined_data[('metric_value',) + col] = metric_pivot[col]
        # Add dichotomy scores  
        combined_data[('dichotomy_score',) + col] = dichotomy_pivot[col]
    
    # Create the combined DataFrame
    detailed_df = pd.DataFrame(combined_data)
    
    # Create proper MultiIndex for columns
    detailed_df.columns = pd.MultiIndex.from_tuples(
        detailed_df.columns,
        names=['score_type', 'scenario', 'subset', 'difficulty']
    )
    
    # Sort for better organization
    detailed_df = detailed_df.sort_index(axis=1)
    detailed_df = detailed_df.sort_index(axis=0)
    
    # Save detailed version
    output_path = Path(output_dir)
    detailed_pkl_path = output_path / "helm_accuracy_detailed_multiindex.pkl"
    detailed_df.to_pickle(detailed_pkl_path)
    
    print(f"Detailed MultiIndex DataFrame saved to {detailed_pkl_path}")
    print(f"Shape: {detailed_df.shape}")
    
    return detailed_df

def save_results(df: pd.DataFrame, output_dir: str = "./data") -> None:
    """
    Save the results to various formats including MultiIndex pivot tables.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save original data as CSV
    csv_path = output_path / "helm_accuracy_dichotomy.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nOriginal results saved to {csv_path}")
    
    # Save as pickle for faster loading
    pkl_path = output_path / "helm_accuracy_dichotomy.pkl"
    df.to_pickle(pkl_path)
    print(f"Original results saved to {pkl_path}")
    
    # Create and save MultiIndex pivot tables
    print("\nCreating MultiIndex pivot tables...")
    
    # Simple MultiIndex with dichotomy scores only
    pivot_df = create_multiindex_pivot(df, output_dir)
    
    # Detailed MultiIndex with both metric values and dichotomy scores
    detailed_df = create_detailed_multiindex_pivot(df, output_dir)
    
    # Save summary statistics
    summary_path = output_path / "helm_accuracy_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("HELM Benchmark Accuracy Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total runs processed: {len(df)}\n")
        f.write(f"Unique models: {df['model'].nunique()}\n")
        f.write(f"Unique scenarios: {df['scenario'].nunique()}\n")
        f.write(f"Average dichotomy score: {df['dichotomy_score'].mean():.3f}\n")
        f.write(f"Metrics used: {', '.join(df['metric_used'].unique())}\n")
        
        f.write("\nTop 10 Models by Accuracy:\n")
        model_stats = df.groupby('model')['dichotomy_score'].mean().sort_values(ascending=False)
        for model, score in model_stats.head(10).items():
            f.write(f"  {model}: {score:.3f}\n")
        
        f.write(f"\nMultiIndex DataFrame shape: {pivot_df.shape}\n")
        f.write(f"Detailed MultiIndex shape: {detailed_df.shape}\n")
    
    print(f"Summary saved to {summary_path}")

def main():
    """
    Main function to process HELM benchmark data and create MultiIndex DataFrames.
    """
    print("HELM Benchmark Accuracy to MultiIndex DataFrame Converter")
    print("="*60)
    
    # Set the base path - adjust this to your data location
    base_path = "/home/azureuser/cloudfiles/code/Users/manhductranvu/reeval-multi/gather-helm"
    
    # Collect all run data
    df = collect_all_runs(base_path)
    
    if df.empty:
        print("No data collected. Please check your data paths.")
        return
    
    # Analyze the results
    analyze_accuracy_by_groups(df)
    
    # Save results including MultiIndex pivot tables
    save_results(df)
    
    print(f"\n✅ Processing complete! Created MultiIndex DataFrames from {len(df)} runs.")

if __name__ == "__main__":
    main()