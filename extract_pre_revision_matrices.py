import os
import json
import glob
import pandas as pd

traces_dir = 'item-editor/eval_traces/traces'
output_base = 'item-editor/eval_response_matrix/pre-revision'

targets = {
    'corebench_hard': 'corebench_corebench_cloud*',
    'scicode': 'scicode_scicode_beach*',
    'scienceagentbench': 'scienceagentbench_scienceagentbench_sky*'
}

for benchmark, prefix in targets.items():
    print(f"Processing {benchmark}...")
    files = glob.glob(os.path.join(traces_dir, f"{prefix}_UPLOAD.json"))
    
    rows = []
    agent_names = []
    
    for f in files:
        basename = os.path.basename(f)
        # remove prefix and suffix
        # scienceagentbench_scienceagentbench_sky0_gpt-5-codex_sab_example_agent_...UPLOAD.json
        agent_name = basename.replace('_UPLOAD.json', '')
        # We can clean the agent name, but for matrix integrity let's keep it unique
        agent_names.append(agent_name)
        
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        results = data.get('results', {})
        res_row = {}
        
        if benchmark in ['corebench_hard', 'scicode']:
            successes = results.get('successful_tasks', [])
            if not isinstance(successes, list): successes = []
            successes = [str(t) for t in successes]
            
            failures = results.get('failed_tasks', [])
            if not isinstance(failures, list): failures = []
            failures = [str(t) for t in failures]
            
            for task in successes:
                res_row[task] = 1.0
            for task in failures:
                res_row[task] = 0.0
                
        elif benchmark == 'scienceagentbench':
            raw_eval = data.get('raw_eval_results', {})
            for task_id, subresults in raw_eval.items():
                if isinstance(subresults, dict):
                    sr = subresults.get('success_rate', 0.0)
                    res_row[str(task_id)] = 1.0 if sr >= 1.0 else 0.0
                    
        rows.append(res_row)
        
    if not rows:
        print(f"No traces found for {benchmark}")
        continue
        
    df = pd.DataFrame(rows, index=agent_names)
    
    # Fill remaining NaNs with NaN or 0? 
    # Usually in the matrix, if it wasn't evaluated, it's NaN.
    # The build_response_matrix uses empty strings "", which read as NaN in pandas.
    
    # Save to pre-revision folder
    out_dir = os.path.join(output_base, benchmark)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'raw_score.csv')
    
    # Add benchmark prefix to columns to match SOTA expectations
    df.columns = [f"{benchmark}.{c}" if not str(c).startswith(benchmark) else c for c in df.columns]
    
    df.to_csv(out_path)
    print(f"Saved {df.shape} to {out_path}")
