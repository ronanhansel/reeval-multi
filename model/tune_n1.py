import subprocess
import os
import pandas as pd
import numpy as np

python_path = "/home/v-qizhengli/miniconda3/envs/hal/bin/python"

def run_experiment(emb_type, lambda_tau):
    cmd = [
        python_path, "model/amortized_irt.py",
        "--embedding-type", emb_type,
        "--n-samples", "1",
        "--model-type", "bernoulli",
        "--lambda-tau", str(lambda_tau)
    ]
    
    print(f"Running {emb_type.upper()} N=1 (Bernoulli) with lambda={lambda_tau}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "model")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
    
    auc = None
    dims = None
    
    for line in result.stdout.split('\n'):
        if "Amortized:" in line and "Active dims:" in line:
            parts = line.split("|")
            for p in parts:
                if "Amortized:" in p:
                    try:
                        auc = float(p.split(":")[1].strip())
                    except: pass
                elif "Active dims:" in p:
                    try:
                        dims = int(p.split(":")[1].strip())
                    except: pass
                    
    if auc is None or dims is None:
        print(f"FAILED TO PARSE OUTPUT for {emb_type} {lambda_tau}")
        return 0, 0
        
    return auc, dims

if __name__ == "__main__":
    os.makedirs('model/result', exist_ok=True)
    results_file = 'model/result/tuning_n1_bernoulli_results.csv'

    # Final precision search for Bernoulli N=1
    configs = {
        'pca': [0.0151, 0.0153, 0.0155],
        'sae': [0.0159, 0.016, 0.0161, 0.0162],
        'raw': [0.0151, 0.0152, 0.0153]
    }
    
    if os.path.exists(results_file):
        df_old = pd.read_csv(results_file)
        results = df_old.to_dict('records')
        print(f"Resuming from {len(results)} existing results.")
    else:
        results = []
    
    for etype, lambdas in configs.items():
        for l in lambdas:
            if any(r['embedding_type'] == etype and r['lambda'] == l for r in results):
                continue
                
            auc, dims = run_experiment(etype, l)
            print(f"[{etype}] Lambda: {l} -> AUC: {auc:.4f}, Dims: {dims}")
            results.append({'embedding_type': etype, 'lambda': l, 'auc': auc, 'dims': dims})
            
            pd.DataFrame(results).to_csv(results_file, index=False)
            
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("BEST PARAMS (Post-Revision N=1 Bernoulli):")
    print("="*50)
    
    for etype in configs.keys():
        sub = df[df['embedding_type'] == etype]
        filtered = sub[(sub['dims'] >= 5) & (sub['dims'] <= 10)]
        if not filtered.empty:
            best = filtered.loc[filtered['auc'].idxmax()]
            print(f"{etype.upper()} (5-10 Dims): lambda={best['lambda']} (AUC={best['auc']:.4f}, Dims={best['dims']})")
        else:
            best_all = sub.loc[sub['auc'].idxmax()]
            print(f"{etype.upper()} (Best AUC, Any Dims): lambda={best_all['lambda']} (AUC={best_all['auc']:.4f}, Dims={best_all['dims']})")
