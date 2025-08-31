#!/usr/bin/env python3
"""
Simple example of how to load and work with the MultiIndex DataFrame.
"""

import pandas as pd
import numpy as np

# Load the MultiIndex DataFrame
df = pd.read_pickle("./data/helm_accuracy_multiindex.pkl")

print("MultiIndex DataFrame Structure:")
print(f"Shape: {df.shape}")
print(f"Index (Models): {list(df.index)}")
print(f"Columns: 3-level MultiIndex with {len(df.columns)} combinations")

print("\nColumn levels:")
print(f"- Scenarios: {df.columns.get_level_values('scenario').unique().tolist()}")
print(f"- Subsets: {df.columns.get_level_values('subset').unique().tolist()}")  
print(f"- Difficulties: {df.columns.get_level_values('difficulty').unique().tolist()}")

print("\nSample data (first 5 models, first 5 scenario combinations):")
print(df.iloc[:5, :5])

print("\n\nExample usages:")
print("\n1. Get all image2latex results:")
latex_results = df.xs('image2latex', level='scenario', axis=1)
print(f"   Shape: {latex_results.shape}")

print("\n2. Get best model overall:")
best_model = df.mean(axis=1).idxmax()
best_score = df.mean(axis=1).max()
print(f"   {best_model}: {best_score:.3f}")

print("\n3. Get performance on easy tasks:")
easy_tasks = df.xs('easy', level='difficulty', axis=1)
print(f"   Shape: {easy_tasks.shape}")
print(f"   Best on easy tasks: {easy_tasks.mean(axis=1).idxmax()}")

print("\n4. Compare two models:")
model1 = "openai/gpt-4o-2024-08-06"
model2 = "anthropic/claude-3-5-sonnet-20240620"

if model1 in df.index and model2 in df.index:
    comparison = pd.DataFrame({
        model1: df.loc[model1],
        model2: df.loc[model2]
    })
    print(f"\n   Comparison of {model1} vs {model2}:")
    print(f"   {model1} average: {df.loc[model1].mean():.3f}")
    print(f"   {model2} average: {df.loc[model2].mean():.3f}")
