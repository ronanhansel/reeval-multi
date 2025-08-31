#!/usr/bin/env python3
"""
Script to create question-level MultiIndex DataFrame from HELM benchmark data.
This preserves individual questions rather than aggregating at scenario level.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple

class HELMQuestionLevelExtractor:
    """Extract individual question-level performance from HELM benchmarks."""
    
    def __init__(self):
        self.accuracy_metrics = [
            "exact_match", "quasi_exact_match", "f1_score", 
            "compilation_success", "edit_similarity", "pixel_similarity",
            "fid_similarity", "ssim_similarity", "accuracy",
            "annotation_financebench_label_correct_answer",
            "execution_accuracy", "program_accuracy"
        ]
        
        self.similarity_thresholds = {
            "compilation_success": 0.5,
            "edit_similarity": 0.8,
            "pixel_similarity": 0.7,
            "fid_similarity": 0.7,
            "ssim_similarity": 0.7,
            "exact_match": 0.5,
            "quasi_exact_match": 0.5,
            "f1_score": 0.8,
            "accuracy": 0.5,
            "annotation_financebench_label_correct_answer": 0.5,
            "execution_accuracy": 0.5,
            "program_accuracy": 0.5
        }

    def extract_question_level_data(self, run_path: Path) -> List[Dict[str, Any]]:
        """Extract individual question-level performance from a single run."""
        try:
            # Load required files
            per_instance_file = run_path / "per_instance_stats.json"
            instances_file = run_path / "instances.json"
            run_spec_file = run_path / "run_spec.json"
            stats_file = run_path / "stats.json"
            
            if not all(f.exists() for f in [per_instance_file, instances_file, run_spec_file, stats_file]):
                return []
            
            # Load data
            with open(per_instance_file, 'r') as f:
                per_instance_stats = json.load(f)
            
            with open(instances_file, 'r') as f:
                instances = json.load(f)
            
            with open(run_spec_file, 'r') as f:
                run_spec = json.load(f)
                if isinstance(run_spec, list):
                    run_spec = run_spec[0]
            
            with open(stats_file, 'r') as f:
                aggregated_stats = json.load(f)
            
            # Extract run metadata
            run_name = run_spec.get('name', run_path.name)
            model_name = run_spec.get('adapter_spec', {}).get('model', 'unknown')
            
            # Parse scenario info
            scenario_parts = run_name.split(':')
            scenario = scenario_parts[0] if scenario_parts else 'unknown'
            
            difficulty = 'unknown'
            subset = 'unknown'
            for part in scenario_parts[1:]:
                if part.startswith('difficulty='):
                    difficulty = part.split('=')[1].replace(',model', '')
                elif part.startswith('subset='):
                    subset = part.split('=')[1]
            
            # Find the best accuracy metric for this run
            best_metric = self.find_best_metric_in_instance_stats(per_instance_stats)
            
            # If no per-instance metric found, try aggregated stats
            aggregated_metric = None
            aggregated_value = None
            if not best_metric:
                aggregated_metric = self.find_best_metric_in_aggregated_stats(aggregated_stats)
                if aggregated_metric:
                    aggregated_value = self.extract_aggregated_metric_value(aggregated_stats, aggregated_metric)
            
            if not best_metric and not aggregated_metric:
                return []
            
            # Extract question-level results
            question_results = []
            
            for i, (instance_stat, instance_data) in enumerate(zip(per_instance_stats, instances)):
                # Get the question text/identifier
                question_id = f"{scenario}_{i:04d}"
                question_text = ""
                
                # Extract question from instance data
                if 'input' in instance_data:
                    if isinstance(instance_data['input'], dict):
                        question_text = instance_data['input'].get('text', '')[:200]
                    else:
                        question_text = str(instance_data['input'])[:200]
                
                if not question_text:
                    question_text = f"Question_{i+1}"
                
                # Extract metric value for this instance
                if best_metric:
                    # Use per-instance metric
                    metric_value = self.extract_instance_metric_value(instance_stat, best_metric)
                    metric_used = best_metric
                else:
                    # Use aggregated metric for all questions in this scenario
                    metric_value = aggregated_value
                    metric_used = aggregated_metric
                
                if metric_value is None:
                    continue
                
                # Convert to dichotomy
                threshold = self.similarity_thresholds.get(metric_used, 0.5)
                dichotomy_score = 1 if metric_value >= threshold else 0
                
                question_results.append({
                    'model': model_name,
                    'scenario': scenario,
                    'subset': subset,
                    'difficulty': difficulty,
                    'question_id': question_id,
                    'question_text': question_text,
                    'metric_used': metric_used,
                    'metric_value': metric_value,
                    'dichotomy_score': dichotomy_score,
                    'data_level': 'per_instance' if best_metric else 'aggregated',
                    'run_path': str(run_path)
                })
            
            return question_results
            
        except Exception as e:
            print(f"Error processing {run_path}: {e}")
            return []

    def find_best_metric_in_instance_stats(self, per_instance_stats: List[Dict]) -> Optional[str]:
        """Find the best available accuracy metric in per-instance stats."""
        if not per_instance_stats:
            return None
        
        # Check what metrics are available in the first instance
        available_metrics = set()
        first_instance = per_instance_stats[0]
        
        if 'stats' in first_instance:
            for stat in first_instance['stats']:
                if 'name' in stat and 'name' in stat['name']:
                    metric_name = stat['name']['name']
                    split = stat['name'].get('split', '')
                    if split in ['valid', 'test']:
                        available_metrics.add(metric_name)
        
        # Find the highest priority metric
        for metric in self.accuracy_metrics:
            if metric in available_metrics:
                return metric
        
        return None

    def find_best_metric_in_aggregated_stats(self, aggregated_stats: List[Dict]) -> Optional[str]:
        """Find the best available accuracy metric in aggregated stats."""
        available_metrics = set()
        for stat in aggregated_stats:
            if 'name' in stat and 'name' in stat['name']:
                metric_name = stat['name']['name']
                split = stat['name'].get('split', '')
                if split in ['valid', 'test']:
                    available_metrics.add(metric_name)
        
        # Find the highest priority metric
        for metric in self.accuracy_metrics:
            if metric in available_metrics:
                return metric
        
        return None

    def extract_aggregated_metric_value(self, aggregated_stats: List[Dict], metric_name: str) -> Optional[float]:
        """Extract metric value from aggregated stats."""
        for stat in aggregated_stats:
            if (stat.get('name', {}).get('name') == metric_name and 
                stat.get('name', {}).get('split') in ['valid', 'test'] and
                'perturbation' not in stat.get('name', {})):
                return stat.get('mean', None)
        return None

    def extract_instance_metric_value(self, instance_stat: Dict, metric_name: str) -> Optional[float]:
        """Extract metric value from a single instance's stats."""
        if 'stats' not in instance_stat:
            return None
        
        for stat in instance_stat['stats']:
            if (stat.get('name', {}).get('name') == metric_name and 
                stat.get('name', {}).get('split') in ['valid', 'test'] and
                'perturbation' not in stat.get('name', {})):
                return stat.get('mean', None)
        
        return None

    def collect_all_question_level_data(self, base_path: str) -> pd.DataFrame:
        """Collect question-level data from all HELM runs."""
        base_path = Path(base_path)
        
        # Find all run directories
        run_dirs = []
        if (base_path / "helm_jsons").exists():
            helm_base = base_path / "helm_jsons"
            for benchmark_dir in helm_base.iterdir():
                if not benchmark_dir.is_dir():
                    continue
                
                runs_dir = benchmark_dir / "runs"
                if runs_dir.exists():
                    for version_dir in runs_dir.iterdir():
                        if version_dir.is_dir():
                            for run_dir in version_dir.iterdir():
                                if (run_dir.is_dir() and 
                                    (run_dir / "per_instance_stats.json").exists() and
                                    (run_dir / "instances.json").exists()):
                                    run_dirs.append(run_dir)
        
        print(f"Found {len(run_dirs)} runs with per-instance data...")
        
        # Extract question-level data from all runs
        all_question_data = []
        
        for run_dir in tqdm(run_dirs, desc="Extracting question-level data"):
            question_data = self.extract_question_level_data(run_dir)
            all_question_data.extend(question_data)
        
        if not all_question_data:
            print("No question-level data found!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_question_data)
        
        print(f"\nExtracted {len(all_question_data)} question-level results!")
        print(f"Unique models: {df['model'].nunique()}")
        print(f"Unique scenarios: {df['scenario'].nunique()}")
        print(f"Total unique questions: {df['question_id'].nunique()}")
        
        return df

    def create_question_level_multiindex(self, df: pd.DataFrame, output_dir: str = "./data") -> pd.DataFrame:
        """Create MultiIndex DataFrame with individual questions as columns."""
        
        # Create a unique question identifier with scenario context
        df['full_question_id'] = (df['scenario'] + '_' + df['subset'] + '_' + 
                                 df['difficulty'] + '_' + df['question_id'])
        
        # Create pivot table with models as index and questions as columns
        pivot_df = df.pivot_table(
            index='model',
            columns='full_question_id',
            values='dichotomy_score',
            aggfunc='first'
        )
        
        # Create MultiIndex columns for better organization
        # Extract scenario info from the question IDs
        column_tuples = []
        for col in pivot_df.columns:
            parts = col.split('_')
            if len(parts) >= 4:
                scenario = parts[0]
                subset = parts[1] 
                difficulty = parts[2]
                question_num = '_'.join(parts[3:])
                column_tuples.append((scenario, subset, difficulty, question_num))
            else:
                column_tuples.append((col, '', '', ''))
        
        # Create MultiIndex
        multi_index = pd.MultiIndex.from_tuples(
            column_tuples,
            names=['scenario', 'subset', 'difficulty', 'question_id']
        )
        pivot_df.columns = multi_index
        
        # Sort for organization
        pivot_df = pivot_df.sort_index(axis=1)
        pivot_df = pivot_df.sort_index(axis=0)
        
        # Save as pickle
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        pkl_path = output_path / "helm_question_level_multiindex.pkl"
        pivot_df.to_pickle(pkl_path)
        
        print(f"\nQuestion-level MultiIndex DataFrame saved to {pkl_path}")
        print(f"Shape: {pivot_df.shape} (models x individual questions)")
        print(f"Total individual questions: {len(pivot_df.columns)}")
        
        return pivot_df

def main():
    """Main function to create question-level MultiIndex DataFrame."""
    print("HELM Question-Level Data Extractor")
    print("=" * 60)
    
    extractor = HELMQuestionLevelExtractor()
    base_path = "/home/azureuser/cloudfiles/code/Users/manhductranvu/reeval-multi/gather-helm"
    
    # Extract question-level data
    df = extractor.collect_all_question_level_data(base_path)
    
    if df.empty:
        print("No question-level data found!")
        return
    
    # Create question-level MultiIndex
    question_multiindex = extractor.create_question_level_multiindex(df)
    
    print(f"\n✅ Question-level extraction complete!")
    print(f"Extracted {len(df)} individual question responses")
    print(f"Created MultiIndex with {question_multiindex.shape[1]} unique questions")

if __name__ == "__main__":
    main()
