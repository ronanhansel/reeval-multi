#!/usr/bin/env python3
"""
Survey individual hyperparameters around lambda_tau=7.5
Run this for quick testing of specific parameters
"""

import subprocess
import json
import os
import sys

# Quick survey grids
survey_configs = {
    'lambda_tau': {
        'values': [60, 61, 62, 63, 64, 65, 66],
        'default': 66,
        'description': 'Temperature parameter for gate sharpening'
    },
    'K_MODEL': {
        'values': [20, 30, 40, 50, 60, 80, 100],
        'default': 50,
        'description': 'Number of latent dimensions'
    },
    'dropout_p': {
        'values': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'default': 0.5,
        'description': 'Input dropout probability'
    },
    'reg_sparse_gates': {
        'values': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        'default': 0.1,
        'description': 'L1 penalty coefficient for gate sparsity'
    },
    'reg_beta_gates': {
        'values': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        'default': 0.1,
        'description': 'Entropy penalty for gate binarization'
    },
    'reg_theta': {
        'values': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        'default': 0.5,
        'description': 'L2 regularization on latent abilities'
    },
}

def run_experiment(param_name, param_value):
    """Run one experiment with specified parameter value"""
    
    # Build command with all defaults except the tested parameter
    cmd = [
        'python', 'model.py',
        ]
    
    # Add parameter being tested
    cmd.extend([f'--{param_name}', str(param_value)])
    
    print(f"Running: {param_name}={param_value}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        
        # Parse output for key metrics
        test_auc = None
        active_dims = None
        for line in result.stdout.split('\n'):
            if 'Test AUC:' in line:
                test_auc = float(line.split('Test AUC:')[1].strip())
            if 'Active Dims:' in line and '|' in line:
                active_dims = int(line.split('Active Dims:')[1].strip())
        
        return {
            'param_name': param_name,
            'param_value': param_value,
            'test_auc': test_auc,
            'active_dims': active_dims,
            'success': True
        }
    except subprocess.TimeoutExpired:
        return {'param_name': param_name, 'param_value': param_value, 'success': False, 'error': 'timeout'}
    except subprocess.CalledProcessError as e:
        return {'param_name': param_name, 'param_value': param_value, 'success': False, 'error': str(e)[:100]}
    except Exception as e:
        return {'param_name': param_name, 'param_value': param_value, 'success': False, 'error': str(e)[:100]}

def survey_parameter(param_name):
    """Survey one hyperparameter"""
    
    if param_name not in survey_configs:
        print(f"Unknown parameter: {param_name}")
        print(f"Available: {list(survey_configs.keys())}")
        return
    
    config = survey_configs[param_name]
    values = config['values']
    
    print("=" * 80)
    print(f"SURVEYING: {param_name}")
    print(f"Description: {config['description']}")
    print(f"Default: {config['default']}")
    print(f"Testing {len(values)} values: {values}")
    print("=" * 80)
    
    results = []
    
    for i, value in enumerate(values, 1):
        print(f"\n[{i}/{len(values)}] Testing {param_name}={value}")
        print("-" * 80)
        
        result = run_experiment(param_name, value)
        results.append(result)
        
        if result['success']:
            print(f"✓ Test AUC: {result['test_auc']:.4f} | Active Dims: {result['active_dims']}")
        else:
            print(f"✗ Failed: {result.get('error', 'unknown')}")
    
    # Save results
    output_file = f'survey_{param_name}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {param_name}")
    print("=" * 80)
    
    successful = [r for r in results if r['success']]
    if successful:
        sorted_results = sorted(successful, key=lambda x: x['test_auc'], reverse=True)
        
        print(f"\n{'Value':<15} {'Test AUC':<12} {'Active Dims':<12} {'vs Default':<12}")
        print("-" * 80)
        
        # Find default performance
        default_result = [r for r in successful if r['param_value'] == config['default']]
        default_auc = default_result[0]['test_auc'] if default_result else 0.0
        
        for r in sorted_results:
            diff = r['test_auc'] - default_auc
            marker = "★" if r == sorted_results[0] else " "
            diff_str = f"{diff:+.4f}" if diff != 0 else "baseline"
            print(f"{marker} {str(r['param_value']):<14} {r['test_auc']:<12.4f} {r['active_dims']:<12} {diff_str}")
        
        best = sorted_results[0]
        improvement = best['test_auc'] - default_auc
        print("-" * 80)
        print(f"Best: {param_name}={best['param_value']} (Test AUC: {best['test_auc']:.4f}, {improvement:+.4f} vs default)")
    
    print(f"\nResults saved to: {output_file}")
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python survey_hyperparameter.py <parameter_name>")
        print("\nAvailable parameters:")
        for name, config in survey_configs.items():
            print(f"  {name:<20} - {config['description']}")
        sys.exit(1)
    
    param_name = sys.argv[1]
    survey_parameter(param_name)
