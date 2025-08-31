#!/usr/bin/env python3
"""
Demo script to show how to work with the MultiIndex DataFrame
that contains HELM benchmark accuracy data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_explore_multiindex():
    """Load and explore the MultiIndex DataFrame."""
    
    # Load the MultiIndex DataFrame
    data_path = Path("./data/helm_accuracy_multiindex.pkl")
    if not data_path.exists():
        print("MultiIndex file not found. Run selective-resmat.py first.")
        return None
    
    df = pd.read_pickle(data_path)
    
    print("HELM Benchmark MultiIndex DataFrame Structure")
    print("=" * 60)
    print(f"Shape: {df.shape} (models x scenario combinations)")
    print(f"Index (models): {len(df.index)}")
    print(f"Columns (scenarios): {len(df.columns)}")
    
    print("\nColumn MultiIndex levels:")
    for i, level_name in enumerate(df.columns.names):
        level_values = df.columns.get_level_values(i).unique()
        print(f"  Level {i} ({level_name}): {list(level_values)}")
    
    print("\nModel names (index):")
    print(df.index.tolist())
    
    return df

def analyze_multiindex_data(df):
    """Analyze the MultiIndex DataFrame in various ways."""
    
    print("\n" + "=" * 60)
    print("MULTIINDEX DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Overall model performance (average across all scenarios)
    print("\n1. Overall Model Performance (average across all scenarios):")
    overall_performance = df.mean(axis=1).sort_values(ascending=False)
    print(overall_performance.round(3))
    
    # 2. Performance by scenario (average across all models)
    print("\n2. Performance by Scenario:")
    scenario_performance = df.groupby(level='scenario', axis=1).mean().mean().sort_values(ascending=False)
    print(scenario_performance.round(3))
    
    # 3. Performance by subset
    print("\n3. Performance by Subset:")
    subset_performance = df.groupby(level='subset', axis=1).mean().mean().sort_values(ascending=False)
    print(subset_performance.round(3))
    
    # 4. Performance by difficulty
    print("\n4. Performance by Difficulty:")
    difficulty_performance = df.groupby(level='difficulty', axis=1).mean().mean().sort_values(ascending=False)
    print(difficulty_performance.round(3))
    
    # 5. Best models for each scenario
    print("\n5. Best Models for Each Scenario:")
    for scenario in df.columns.get_level_values('scenario').unique():
        scenario_cols = df.xs(scenario, level='scenario', axis=1)
        best_model = scenario_cols.mean(axis=1).idxmax()
        best_score = scenario_cols.mean(axis=1).max()
        print(f"  {scenario}: {best_model} ({best_score:.3f})")
    
    # 6. Hardest scenarios (lowest average performance)
    print("\n6. Hardest Scenarios (by subset and difficulty):")
    scenario_difficulty = df.mean(axis=0).sort_values()
    print("Hardest 5 scenario combinations:")
    for i, (scenario_combo, score) in enumerate(scenario_difficulty.head().items()):
        print(f"  {i+1}. {scenario_combo}: {score:.3f}")

def demonstrate_slicing(df):
    """Demonstrate various ways to slice the MultiIndex DataFrame."""
    
    print("\n" + "=" * 60)
    print("MULTIINDEX SLICING EXAMPLES")
    print("=" * 60)
    
    # 1. Select all image2latex scenarios
    print("\n1. All image2latex scenarios:")
    latex_scenarios = df.xs('image2latex', level='scenario', axis=1)
    print(f"Shape: {latex_scenarios.shape}")
    print("First 3 models:")
    print(latex_scenarios.head(3))
    
    # 2. Select specific subset and difficulty
    print("\n2. Algorithm subset, easy difficulty (all scenarios):")
    try:
        algorithm_easy = df.xs(('algorithm', 'easy'), level=['subset', 'difficulty'], axis=1)
        print(f"Shape: {algorithm_easy.shape}")
        print("Top 5 models:")
        model_scores = algorithm_easy.mean(axis=1).sort_values(ascending=False)
        print(model_scores.head().round(3))
    except KeyError:
        print("No exact match for algorithm + easy combination")
    
    # 3. Select specific model performance
    print("\n3. Performance of top model across all scenarios:")
    top_model = df.mean(axis=1).idxmax()
    top_model_performance = df.loc[top_model]
    print(f"Model: {top_model}")
    print("Performance by scenario:")
    for scenario in top_model_performance.groupby(level='scenario'):
        scenario_name, scenario_data = scenario
        print(f"  {scenario_name}: {scenario_data.mean():.3f}")

def create_summary_tables(df, output_dir="./data"):
    """Create and save summary tables."""
    
    print("\n" + "=" * 60)
    print("CREATING SUMMARY TABLES")
    print("=" * 60)
    
    output_path = Path(output_dir)
    
    # 1. Model x Scenario summary (average across subsets and difficulties)
    model_scenario_summary = df.groupby(level='scenario', axis=1).mean()
    summary_path = output_path / "model_scenario_summary.pkl"
    model_scenario_summary.to_pickle(summary_path)
    print(f"\nModel x Scenario summary saved to {summary_path}")
    print("Sample:")
    print(model_scenario_summary.head())
    
    # 2. Model x Subset summary (average across scenarios and difficulties)  
    model_subset_summary = df.groupby(level='subset', axis=1).mean()
    subset_path = output_path / "model_subset_summary.pkl"
    model_subset_summary.to_pickle(subset_path)
    print(f"\nModel x Subset summary saved to {subset_path}")
    
    # 3. Model x Difficulty summary (average across scenarios and subsets)
    model_difficulty_summary = df.groupby(level='difficulty', axis=1).mean()
    difficulty_path = output_path / "model_difficulty_summary.pkl"
    model_difficulty_summary.to_pickle(difficulty_path)
    print(f"\nModel x Difficulty summary saved to {difficulty_path}")
    
    # 4. Create a simple model ranking
    model_ranking = df.mean(axis=1).sort_values(ascending=False)
    ranking_path = output_path / "model_ranking.pkl"
    model_ranking.to_pickle(ranking_path)
    print(f"\nModel ranking saved to {ranking_path}")
    print("\nTop 10 Models:")
    print(model_ranking.head(10).round(3))

def main():
    """Main function to demonstrate MultiIndex DataFrame usage."""
    
    # Load and explore the MultiIndex DataFrame
    df = load_and_explore_multiindex()
    if df is None:
        return
    
    # Analyze the data
    analyze_multiindex_data(df)
    
    # Demonstrate slicing techniques
    demonstrate_slicing(df)
    
    # Create and save summary tables
    create_summary_tables(df)
    
    print(f"\n✅ MultiIndex analysis complete!")
    print("Files created:")
    print("  - helm_accuracy_multiindex.pkl (main MultiIndex DataFrame)")
    print("  - model_scenario_summary.pkl") 
    print("  - model_subset_summary.pkl")
    print("  - model_difficulty_summary.pkl")
    print("  - model_ranking.pkl")

if __name__ == "__main__":
    main()
