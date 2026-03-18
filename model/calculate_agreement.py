import pandas as pd
import numpy as np
import os
import glob
import argparse

def calculate_gwet_ac1(pair):
    """
    Calculates Gwet's AC1 for binary data.
    More robust than Kappa for unbalanced datasets.
    """
    n = len(pair)
    if n == 0: return np.nan
    
    p_observed = (pair.iloc[:,0] == pair.iloc[:,1]).mean()
    
    # Category prevalence (average across raters)
    p1 = (pair.iloc[:,0].mean() + pair.iloc[:,1].mean()) / 2
    p0 = 1 - p1
    
    # Chance agreement for AC1
    p_chance = 2 * p1 * p0
    
    if p_chance >= 1: return 1.0 
    
    ac1 = (p_observed - p_chance) / (1 - p_chance)
    return ac1

def calculate_pabak(pair):
    """
    Prevalence-Adjusted Bias-Adjusted Kappa.
    Assumes 50/50 prevalence to normalize scores.
    """
    n = len(pair)
    if n == 0: return np.nan
    p_observed = (pair.iloc[:,0] == pair.iloc[:,1]).mean()
    # PABAK formula: 2 * P(o) - 1
    return 2 * p_observed - 1

def load_csv_transpose(filepath, rater_id):
    try:
        df = pd.read_csv(filepath)
        if 'agent' in df.columns:
            df = df.set_index('agent')
        df_t = df.T
        df_t.columns = [rater_id]
        df_t.index = df_t.index.map(str)
        return df_t
    except Exception as e:
        return None

def main():
    base_dir = "/Users/ronan/Developer/agent-eval/item-editor/eval_response_matrix/post-revision"
    
    benchmarks = [
        ("SciCode", "scicode", ["beach"]),
        ("ScienceAgentBench", "scienceagentbench", ["sky"]),
        ("CORE", "corebench_hard", ["cloud"]),
        ("ColBench", "colbench_backend_programming", ["moon", "sun"])
    ]
    
    final_results = []

    for label, b_dir, prefixes in benchmarks:
        print(f"Processing {label}...")
        
        # Determine consensus file (index 0)
        consensus_prefix = prefixes[0]
        consensus_path = os.path.join(base_dir, b_dir, "verdicts", f"verdict_{consensus_prefix}0.csv")
        df_consensus = load_csv_transpose(consensus_path, "consensus")
        if df_consensus is not None:
            df_consensus["consensus"] = pd.to_numeric(df_consensus["consensus"], errors='coerce')

        files = []
        for prefix in prefixes:
            path_pattern = os.path.join(base_dir, b_dir, "verdicts", f"verdict_{prefix}*.csv")
            files.extend(glob.glob(path_pattern))
        
        files = sorted(list(set(files)))
        
        all_raters_data = []
        for f in files:
            rater_id = os.path.basename(f).replace("verdict_", "").replace(".csv", "")
            if rater_id == f"{consensus_prefix}0" or (len(prefixes)>1 and rater_id == f"{prefixes[1]}0"):
                continue
            
            df_rater = load_csv_transpose(f, rater_id)
            if df_rater is not None:
                all_raters_data.append(df_rater)
        
        if not all_raters_data:
            continue
            
        bench_df = pd.concat(all_raters_data, axis=1)
        if df_consensus is not None:
            bench_df = bench_df.join(df_consensus, how='left')
        
        for col in bench_df.columns:
            bench_df[col] = pd.to_numeric(bench_df[col], errors='coerce')
        
        prefix_match = b_dir if b_dir != "colbench_backend_programming" else "colbench"
        if b_dir == "scienceagentbench": prefix_match = "scienceagentbench"
        if b_dir == "corebench_hard": prefix_match = "corebench_hard"
        
        subset_df = bench_df[bench_df.index.str.contains(prefix_match, case=False)]
        
        if subset_df.empty:
            continue

        rater_cols = [c for c in subset_df.columns if c != "consensus"]
        
        ac1_vals = []
        pabak_vals = []
        percent_agreements = []
        
        for i in range(len(rater_cols)):
            for j in range(i + 1, len(rater_cols)):
                r1, r2 = rater_cols[i], rater_cols[j]
                pair = subset_df[[r1, r2]].dropna()
                if len(pair) > 0:
                    percent_agreements.append((pair[r1] == pair[r2]).mean())
                    ac1_vals.append(calculate_gwet_ac1(pair))
                    pabak_vals.append(calculate_pabak(pair))
        
        avg_ac1 = np.nanmean(ac1_vals) if ac1_vals else np.nan
        avg_pabak = np.nanmean(pabak_vals) if pabak_vals else np.nan
        avg_percent = np.nanmean(percent_agreements) if percent_agreements else np.nan
        
        # Consensus Agreement (using AC1 since it's more appropriate)
        cons_ac1_vals = []
        if "consensus" in subset_df.columns:
            for r in rater_cols:
                pair = subset_df[[r, "consensus"]].dropna()
                if len(pair) > 0:
                    cons_ac1_vals.append(calculate_gwet_ac1(pair))
        
        avg_cons_ac1 = np.nanmean(cons_ac1_vals) if cons_ac1_vals else np.nan
        
        final_results.append({
            "Benchmark": label,
            "Raters": len(rater_cols),
            "Tasks": len(subset_df),
            "% Agree": avg_percent,
            "Gwet's AC1": avg_ac1,
            "PABAK": avg_pabak,
            "Consensus Agreement (AC1)": avg_cons_ac1
        })

    results_df = pd.DataFrame(final_results)
    
    # Save to markdown
    with open("agreement_results.md", "w") as f:
        f.write("# Inter-Rater Agreement Analysis\n\n")
        f.write("This report summarizes the consistency between raters using metrics robust to unbalanced datasets.\n\n")
        f.write("### Why was Cohen's Kappa removed?\n")
        f.write("Cohen's Kappa is unreliable for these benchmarks due to the **Kappa Paradox**. Because our data is heavily skewed towards one category (Passing/0), Kappa penalizes agreement on that majority class, leading to near-zero scores even when raters agree on 95%+ of items. Gwet's AC1 and PABAK provide a more accurate measure of reliability for this data.\n\n")
        f.write(results_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### Metric Interpretations\n")
        f.write("- **% Agree**: The raw percentage of identical verdicts.\n")
        f.write("- **Gwet's AC1**: The primary reliability metric. It adjusts for chance while remaining stable even if one category is very rare (like our Faults).\n")
        f.write("- **PABAK**: Prevalence-Adjusted Bias-Adjusted Kappa. A normalized version of Kappa that assumes a neutral distribution to eliminate bias.\n")
        f.write("- **Consensus Agreement (AC1)**: Consistency between individual raters and their specific benchmark consensus.\n\n")

    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
