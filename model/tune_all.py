import subprocess
import os
import time
import pandas as pd

python_path = os.path.expanduser("~/miniconda3/envs/hal/bin/python")

def run_experiment(embedding, lambda_tau):
    cmd = [
        python_path, "model/amortized_irt.py",
        "--embedding-type", embedding,
        "--n-samples", "54",
        "--model-type", "beta",
        "--lambda-tau", str(lambda_tau)
    ]
    
    print(f"Running '{embedding}' with lambda={lambda_tau}...")
    
    # We set PYTHONPATH up so model/ works
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "model")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)) + "/..", env=env)
    
    auc = None
    dims = None
    
    for line in result.stdout.split('\n'):
        if "Amortized:" in line and "Active dims:" in line:
            parts = line.split("|")
            for p in parts:
                if "Amortized:" in p:
                    auc = float(p.split(":")[1].strip())
                elif "Active dims:" in p:
                    dims = int(p.split(":")[1].strip())
                    
    if auc is None or dims is None:
        print(f"FAILED TO PARSE OUTPUT for {embedding} {lambda_tau}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
        return 0, 0
        
    return auc, dims

if __name__ == "__main__":
    # Ensure result dir exists
    os.makedirs('model/result', exist_ok=True)

    embeddings = ['raw', 'pca', 'sae']
    
    # searching for 5-7 dim target
    lambdas_raw = [0.0285, 0.029, 0.0295]
    lambdas_pca = [0.0535, 0.054, 0.0545]
    lambdas_sae = [0.0532, 0.0535, 0.0538]
    
    results = []
    
    for emb in embeddings:
        if emb == 'sae': l_list = lambdas_sae
        elif emb == 'pca': l_list = lambdas_pca
        else: l_list = lambdas_raw
        
        for l in l_list:
            auc, dims = run_experiment(emb, l)
            print(f"[{emb}] Lambda: {l} -> AUC: {auc:.4f}, Dims: {dims}")
            results.append({'embedding': emb, 'lambda': l, 'auc': auc, 'dims': dims})
            
    df = pd.DataFrame(results)
    df.to_csv('model/result/tuning_results.csv', index=False)
    
    print("\n" + "="*50)
    print("BEST PARAMS (Dimensions <= 10):")
    print("="*50)
    
    for emb in embeddings:
        sub = df[(df['embedding'] == emb) & (df['dims'] <= 10) & (df['dims'] > 0)]
        if len(sub) > 0:
            best = sub.loc[sub['auc'].idxmax()]
            print(f"{emb}: lambda={best['lambda']} (AUC={best['auc']:.4f}, Dims={best['dims']})")
        else:
            print(f"{emb}: No parameters found with 0 < dims <= 10.")
            fallback = df[df['embedding'] == emb].loc[df[df['embedding'] == emb]['auc'].idxmax()]
            print(f"      Fallback (Highest AUC): lambda={fallback['lambda']} (AUC={fallback['auc']:.4f}, Dims={fallback['dims']})")
