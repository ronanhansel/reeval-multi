#!/usr/bin/env python3
"""
Run train_full.py with different lambda_tau values and aggregate results
"""

import subprocess
import json
import os
from pathlib import Path

# Lambda values to test
# lambda_values = [0.5, 1, 5, 10, 15, 25, 50, 75, 100, 125, 150, 200, 250]
lambda_values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

# Benchmarks to use
benchmarks = ['assistantbench', 'taubench_airline', 'corebench', 'scienceagentbench']
benchmark_str = '_'.join(benchmarks)

print("=" * 60)
print("Lambda_tau Hyperparameter Search")
print("=" * 60)
print(f"Testing {len(lambda_values)} lambda_tau values: {lambda_values}")
print(f"Benchmarks: {benchmarks}")
print("=" * 60)

results = []

for i, lambda_tau in enumerate(lambda_values, 1):
    print(f"\n[{i}/{len(lambda_values)}] Running with lambda_tau = {lambda_tau}")
    print("-" * 60)
    
    # Build command
    cmd = [
        'python', 'train_full.py',
        '--benchmark'] + benchmarks + [
        '--lambda_tau', str(lambda_tau)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Load the results file (convert lambda_tau to float format)
        result_file = f'lambda_tau_{float(lambda_tau)}_{benchmark_str}_results.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                run_result = json.load(f)
                results.append(run_result)
                print(f"✓ Test AUC: {run_result['test_auc']:.4f}")
        else:
            print(f"⚠ Warning: Result file {result_file} not found")
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running lambda_tau={lambda_tau}")
        print(f"Error output: {e.stderr}")
        continue

# Save aggregated results
summary_file = f'lambda_tau_summary_{benchmark_str}.json'
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Experiment Summary")
print("=" * 60)
print(f"\nCompleted {len(results)}/{len(lambda_values)} experiments")
print(f"Results saved to: {summary_file}")

if results:
    print("\nTest AUC Summary:")
    print("-" * 60)
    for r in results:
        print(f"lambda_tau={r['lambda_tau']:6.1f} | Test AUC: {r['test_auc']:.4f} | Active Dims: {r['active_dims']}")
    
    # Find best lambda_tau
    best_result = max(results, key=lambda x: x['test_auc'])
    print("-" * 60)
    print(f"Best lambda_tau: {best_result['lambda_tau']} (Test AUC: {best_result['test_auc']:.4f})")
    print("=" * 60)
