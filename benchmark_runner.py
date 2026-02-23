#!/usr/bin/env python3
import subprocess
import re
import pandas as pd
import itertools
import os

def run_experiment(emb_type, n_samples, model_type, lambda_tau, snapping_threshold):
    cmd = [
        "python", "model/amortized_irt.py",
        "--embedding-type", emb_type,
        "--n-samples", str(n_samples),
        "--model-type", model_type,
        "--lambda-tau", str(lambda_tau),
        "--epochs", "250",
        "--snapping-threshold", str(snapping_threshold),
        "--wd-theta", "5.0"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    # Example format:
    # -> RMSE | Mean: 0.2779 | Rasch: 0.2707 | Amortized: 0.2829
    # -> AUC  | Mean: 0.5000 | Rasch: 0.5398 | Amortized: 0.6799 | Active dims: 8
    
    metrics = {
        'emb_type': emb_type,
        'n_samples': n_samples,
        'model_type': model_type,
        'lambda_tau': lambda_tau,
        'snapping_threshold': snapping_threshold,
        'rmse_mean': None,
        'rmse_rasch': None,
        'rmse_amortized': None,
        'auc_mean': None,
        'auc_rasch': None,
        'auc_amortized': None,
        'active_dims': None
    }
    
    for line in result.stdout.split('\n'):
        if '-> RMSE | Mean:' in line:
            m = re.search(r'Mean: ([\d.]+) \| Rasch: ([\d.]+) \| Amortized: ([\d.]+)', line)
            if m:
                metrics['rmse_mean'] = float(m.group(1))
                metrics['rmse_rasch'] = float(m.group(2))
                metrics['rmse_amortized'] = float(m.group(3))
        elif '-> AUC  | Mean:' in line:
            m = re.search(r'Mean: ([\d.]+) \| Rasch: ([\d.]+) \| Amortized: ([\d.]+) \| Active dims: (\d+)', line)
            if m:
                metrics['auc_mean'] = float(m.group(1))
                metrics['auc_rasch'] = float(m.group(2))
                metrics['auc_amortized'] = float(m.group(3))
                metrics['active_dims'] = int(m.group(4))
                
    if metrics['active_dims'] is None:
        print("Failed to parse output! (Possible error in model)")
        print(result.stdout[-500:])
        print(result.stderr[-500:])
        
    return metrics

def main():
    results = []
    
# Grid definitions
    n54_lambdas_pca = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.38, 1.5, 1.8]
    n54_lambdas_sae = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.34, 1.5, 1.8]
    n54_lambdas_raw = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    configs = [
        ('pca', 54, 'beta', n54_lambdas_pca, [0.01]),
        ('sae', 54, 'beta', n54_lambdas_sae, [0.01]),
        ('raw', 54, 'beta', n54_lambdas_raw, [0.01]),
    ]
    
    print("Starting Grid Search on N=54 for MAX performance...")
    
    for emb_type, n, m_type, lambdas, snaps in configs:
        for l, s in itertools.product(lambdas, snaps):
            m = run_experiment(emb_type, n, m_type, l, s)
            results.append(m)
            df = pd.DataFrame(results)
            df.to_csv('model/tuning_results.csv', index=False)
            
            print(f"Result: Active Dims={m['active_dims']}, AUC={m['auc_amortized']}")

    print("\n\n=== CONSOLIDATED BEST N=54 RESULTS ===")
    df = pd.DataFrame(results)
    
    # Find best hyperparams for each category
    # Criteria: We want active_dims between 2 and 20, maximize auc_amortized
    best_rows = []
    
    for emb_type in ['pca', 'sae', 'raw']:
        sub_df = df[(df['emb_type'] == emb_type) & (df['n_samples'] == 54)]
        if sub_df.empty: continue
        
        # Filter by valid active dims constraint
        valid_df = sub_df[(sub_df['active_dims'] >= 2) & (sub_df['active_dims'] <= 35)]
        
        if not valid_df.empty:
            best_row = valid_df.sort_values('auc_amortized', ascending=False).iloc[0]
        else:
            # Fallback to whatever has highest AUC if no dimensions filter is met
            best_row = sub_df.sort_values('auc_amortized', ascending=False).iloc[0]
            
        best_rows.append(best_row)
            
    best_df = pd.DataFrame(best_rows)
    print(best_df[['emb_type', 'n_samples', 'model_type', 'lambda_tau', 'snapping_threshold', 'active_dims', 'rmse_rasch', 'rmse_amortized', 'auc_rasch', 'auc_amortized']].to_string())

if __name__ == '__main__':
    main()
