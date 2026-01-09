#!/usr/bin/env python3
"""
Comprehensive hyperparameter survey for train_full.py
Keeping lambda_tau=7.5 (best value) and varying other hyperparameters
"""

import subprocess
import json
import os
from pathlib import Path
import copy

# Base hyperparameters (with optimal lambda_tau)
base_config = {
    'lambda_tau': 7.5,
    'K_MODEL': 30,
    'dropout_p': 0.7,
    'reg_sparse_gates': 0.1,
    'reg_beta_gates': 0.1,
    'reg_theta': 0.5,
    'lr_tau': 0.01,
    'lr_proj': 0.005,
    'lr_latent': 0.01,
    'wd_proj': 1e-2,
    'wd_latent': 1e-4,
}

# Define hyperparameter search spaces
hyperparameter_grid = {
    'K_MODEL': [20, 30, 40, 50, 60, 80, 100],
    'dropout_p': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'reg_sparse_gates': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
    'reg_beta_gates': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
    'reg_theta': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
    'lr_proj': [0.001, 0.002, 0.005, 0.01, 0.02],
    'wd_proj': [0.0, 1e-3, 1e-2, 5e-2, 1e-1],
}

print("=" * 80)
print("COMPREHENSIVE HYPERPARAMETER SURVEY")
print("=" * 80)
print(f"\nBase configuration (lambda_tau={base_config['lambda_tau']}):")
for k, v in base_config.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 80)
print("Hyperparameters to Survey:")
print("=" * 80)
for param, values in hyperparameter_grid.items():
    print(f"\n{param}: {values}")
    print(f"  Total experiments: {len(values)}")

total_experiments = sum(len(values) for values in hyperparameter_grid.values())
print("\n" + "=" * 80)
print(f"TOTAL EXPERIMENTS: {total_experiments}")
print("=" * 80)

# Ask for confirmation
response = input("\nProceed with survey? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("Aborted.")
    exit(0)

all_results = []
experiment_num = 0

# Survey each hyperparameter
for param_name, param_values in hyperparameter_grid.items():
    print(f"\n{'=' * 80}")
    print(f"SURVEYING: {param_name}")
    print(f"{'=' * 80}")
    
    param_results = []
    
    for i, param_value in enumerate(param_values, 1):
        experiment_num += 1
        
        # Create config for this experiment
        config = copy.deepcopy(base_config)
        config[param_name] = param_value
        
        print(f"\n[{experiment_num}/{total_experiments}] {param_name}={param_value}")
        print("-" * 80)
        
        # Build command (lambda_tau is always passed)
        cmd = [
            'python', 'train_full_configurable.py',
            ] + [
            '--lambda_tau', str(config['lambda_tau']),
            '--K_MODEL', str(config['K_MODEL']),
            '--dropout_p', str(config['dropout_p']),
            '--reg_sparse_gates', str(config['reg_sparse_gates']),
            '--reg_beta_gates', str(config['reg_beta_gates']),
            '--reg_theta', str(config['reg_theta']),
            '--lr_tau', str(config['lr_tau']),
            '--lr_proj', str(config['lr_proj']),
            '--lr_latent', str(config['lr_latent']),
            '--wd_proj', str(config['wd_proj']),
            '--wd_latent', str(config['wd_latent']),
        ]
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            
            # Extract Test AUC from output
            output_lines = result.stdout.split('\n')
            test_auc = None
            for line in output_lines:
                if 'Test AUC:' in line:
                    test_auc = float(line.split('Test AUC:')[1].strip())
                    break
            
            # Load full results
            result_file = f'hyperparam_{param_name}_{param_value}_results.json'
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    run_result = json.load(f)
                    run_result['param_name'] = param_name
                    run_result['param_value'] = param_value
                    run_result['config'] = config
                    param_results.append(run_result)
                    all_results.append(run_result)
                    print(f"✓ Test AUC: {run_result['test_auc']:.4f} | Active Dims: {run_result['active_dims']}")
            else:
                print(f"⚠ Warning: Result file not found")
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout (>10 min)")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error: {e.stderr[:200]}")
        except Exception as e:
            print(f"✗ Unexpected error: {str(e)[:200]}")
    
    # Save results for this parameter
    param_summary_file = f'hyperparam_survey_{param_name}.json'
    with open(param_summary_file, 'w') as f:
        json.dump(param_results, f, indent=2)
    
    # Print summary for this parameter
    if param_results:
        print(f"\n{param_name} Summary:")
        print("-" * 80)
        sorted_results = sorted(param_results, key=lambda x: x['test_auc'], reverse=True)
        for r in sorted_results[:5]:  # Top 5
            marker = "★" if r == sorted_results[0] else " "
            print(f"{marker} {param_name}={r['param_value']:8} | Test AUC: {r['test_auc']:.4f} | Active: {r['active_dims']}")

# Save complete results
complete_summary_file = f'hyperparam_survey_complete.json'
with open(complete_summary_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 80)
print("SURVEY COMPLETE")
print("=" * 80)
print(f"\nTotal experiments completed: {len(all_results)}/{total_experiments}")
print(f"Results saved to: {complete_summary_file}")

# Print best configuration for each parameter
print("\n" + "=" * 80)
print("BEST VALUE FOR EACH HYPERPARAMETER")
print("=" * 80)
for param_name in hyperparameter_grid.keys():
    param_results = [r for r in all_results if r['param_name'] == param_name]
    if param_results:
        best = max(param_results, key=lambda x: x['test_auc'])
        baseline = [r for r in param_results if r['param_value'] == base_config[param_name]]
        baseline_auc = baseline[0]['test_auc'] if baseline else 0.0
        improvement = best['test_auc'] - baseline_auc
        sign = "+" if improvement > 0 else ""
        print(f"\n{param_name:20s}: {best['param_value']:8} (AUC: {best['test_auc']:.4f}, {sign}{improvement:.4f} vs baseline)")
