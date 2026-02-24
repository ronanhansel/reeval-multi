import os, pandas as pd

def aggregate_remediation_resmats(benchmark_dir, benchmark_name):
    files = [f for f in os.listdir(benchmark_dir) if f.startswith('resmat')]
    
    base_file = None
    for f in files:
        if f.endswith('0.csv') or f.endswith('original.csv'):
            base_file = f
            break
            
    if base_file is None:
        print("No base file found in", benchmark_dir)
        return None
        
    df_base = pd.read_csv(os.path.join(benchmark_dir, base_file), index_col=0)
    df_base.columns = [f"{benchmark_name}.{c}" if not str(c).startswith(benchmark_name) and not str(c).startswith(benchmark_name.replace('_hard','')) else c for c in df_base.columns]
    
    remediation_dfs = []
    for f in files:
        if f != base_file:
            df = pd.read_csv(os.path.join(benchmark_dir, f), index_col=0)
            df.columns = [f"{benchmark_name}.{c}" if not str(c).startswith(benchmark_name) and not str(c).startswith(benchmark_name.replace('_hard','')) else c for c in df.columns]
            remediation_dfs.append(df)
            
    if not remediation_dfs:
        return df_base
        
    remediation_concat = pd.concat(remediation_dfs)
    remediation_avg = remediation_concat.groupby(remediation_concat.index).mean()
    
    print(f"{benchmark_name} - Base shape: {df_base.shape}, Rem avg shape: {remediation_avg.shape}")
    
    df_base.update(remediation_avg)
    return df_base

# test on corebench
df = aggregate_remediation_resmats('item-editor/eval_response_matrix/post-revision/corebench_hard/resmat', 'corebench_hard')
print(df.head())
