#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tueplots import bundles
import scipy.cluster.hierarchy as sch
from model.plotting import colors as pc
import re

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESMAT_DIR = os.path.join(REPO_ROOT, 'item-editor', 'eval_response_matrix')
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
RANDOM_SEED = 42

os.makedirs(FIGURE_DIR, exist_ok=True)

def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")

def normalize_agent_name(name):
    return str(name)

def get_match_key(name):
    if not isinstance(name, str): return str(name)
    k = name.lower()
    prefixes = ['scienceagentbench', 'corebench_hard', 'corebench.hard', 'assistantbench', 'scicode', 'colbench']
    for p in prefixes:
        k = k.replace(p.lower(), '')
    k = re.sub(r'[^a-z0-9]', '', k)
    return k

def collect_data(benchmarks):
    PRE_REV_DIR = os.path.join(RESMAT_DIR, 'pre-revision')
    POST_REV_DIR = os.path.join(RESMAT_DIR, 'post-revision')

    # 1. Collect PRE-REVISION (Aligned with amortized_irt.py logic)
    pre_res_list = []
    pre_ver_list = []
    pre_rub_list = []
    
    for bench in benchmarks:
        # Success Rate (Resmat)
        path = os.path.join(PRE_REV_DIR, bench, 'benchmark.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in df.columns]
            
            # [ALIGNMENT FIX]: Filter SciCode to match the 29 refined items used in post-revision
            if bench == 'scicode':
                SCICODE_POST_IDS = ['12', '14', '15', '16', '2', '23', '28', '32', '35', '41', '43', '46', '48', '52', '56', '58', '59', '61', '62', '63', '64', '66', '67', '71', '72', '77', '79', '80', '9']
                post_cols = [f"scicode.{it}" for it in SCICODE_POST_IDS]
                valid_df_cols = [c for c in df.columns if c in post_cols]
                df = df[valid_df_cols]
                
            df.index = [f"{bench}.{normalize_agent_name(a)}" for a in df.index]
            if df.index.duplicated().any(): df = df.groupby(level=0).mean()
            pre_res_list.append(df)
        
        # Verdict Strip
        v_path = os.path.join(PRE_REV_DIR, bench, 'verdicts', 'verdict_original.csv')
        if os.path.exists(v_path):
            vdf = pd.read_csv(v_path, index_col=0)
            judge = vdf[vdf.index == 'judge_verdict']
            if judge.empty and 'agent' in vdf.columns:
                judge = vdf[vdf['agent'] == 'judge_verdict'].drop(columns=['agent'])
            if not judge.empty:
                cols = [c for c in judge.columns if str(c).startswith(bench) or str(c).startswith(bench.replace('_hard',''))]
                if not cols:
                    judge.columns = [f"{bench}.{c}" for c in judge.columns]
                else:
                    judge = judge[cols]
                    judge.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in judge.columns]
                pre_ver_list.append(judge)

        # Rubrics (Red/White matrix)
        rub_path = os.path.join(PRE_REV_DIR, bench, 'rubrics', 'rubric_score_original.csv')
        if os.path.exists(rub_path):
            rub_df = pd.read_csv(rub_path, index_col=0)
            rub_df.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in rub_df.columns]
            rub_df.index = [f"{bench}.{normalize_agent_name(a)}" for a in rub_df.index]
            if rub_df.index.duplicated().any(): rub_df = rub_df.groupby(level=0).mean()
            pre_rub_list.append(rub_df)

    pre_res_merged = pd.concat(pre_res_list, axis=1, join='outer') if pre_res_list else pd.DataFrame()
    pre_ver_merged = pd.concat(pre_ver_list, axis=1, join='outer') if pre_ver_list else pd.DataFrame()
    pre_rub_merged = pd.concat(pre_rub_list, axis=1, join='outer') if pre_rub_list else pd.DataFrame()

    # [ALIGNMENT FIX]: Sample 8 agents per benchmark for pre-revision to equate with post-revision (32 total)
    sampled_pre_agents = []
    for df in pre_res_list:
        b_agents = df.dropna(how='all').index.tolist()
        np.random.seed(RANDOM_SEED)
        if len(b_agents) > 8:
            sampled = np.random.choice(b_agents, size=8, replace=False)
        else:
            sampled = b_agents
        sampled_pre_agents.extend(sampled)
    
    pre_revision_agents = sorted(list(set(sampled_pre_agents)))
    pre_revision_cols = sorted(list(pre_res_merged.columns))
    
    pre_revision_res_val = pre_res_merged.reindex(index=pre_revision_agents, columns=pre_revision_cols).apply(pd.to_numeric, errors='coerce').values
    pre_revision_ver_val = pre_ver_merged.reindex(columns=pre_revision_cols).apply(pd.to_numeric, errors='coerce').values
    
    # Fuzzy match Rubrics for pre-revision
    pre_revision_rub_val = np.full((len(pre_revision_agents), len(pre_revision_cols)), np.nan)
    if not pre_rub_merged.empty:
        rub_map = {get_match_key(a): a for a in pre_rub_merged.index}
        aligned_rows = []
        for a in pre_revision_agents:
            key = get_match_key(a)
            if key in rub_map:
                aligned_rows.append(pre_rub_merged.loc[rub_map[key]])
            else:
                found = False
                for r_key in sorted(rub_map.keys(), key=len, reverse=True):
                    if r_key and (r_key in key or key in r_key):
                        aligned_rows.append(pre_rub_merged.loc[rub_map[r_key]])
                        found = True
                        break
                if not found:
                    aligned_rows.append(pd.Series(index=pre_rub_merged.columns, dtype=float))
        
        aligned_rub_df = pd.DataFrame(aligned_rows, index=pre_revision_agents)
        pre_revision_rub_val = aligned_rub_df.reindex(columns=pre_revision_cols).apply(pd.to_numeric, errors='coerce').values

    # 2. Collect POST-REVISION (Averaged Oracle Matrix)
    bench_iters = {}
    for bench in benchmarks:
        b_dir = os.path.join(POST_REV_DIR, bench, 'resmat')
        if not os.path.exists(b_dir): continue
        
        files = [f for f in os.listdir(b_dir) if f.startswith('resmat')]
        base_f = next((f for f in files if (f.endswith('0.csv') and not f[-6].isdigit()) or 'original' in f), None)
        if not base_f: continue
        
        df_base = pd.read_csv(os.path.join(b_dir, base_f), index_col=0)
        valid_cols = [c for c in df_base.columns if str(c).startswith(bench)]
        if not valid_cols: continue
        df_base = df_base[valid_cols]
        df_base.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in df_base.columns]
        df_base.index = [normalize_agent_name(a) for a in df_base.index]
        
        v_dir = os.path.join(POST_REV_DIR, bench, 'verdicts')
        rub_dir = os.path.join(POST_REV_DIR, bench, 'rubrics')
        
        iters = []
        iters.append({'res': df_base.copy()}) # Base iteration
        
        for f in sorted(files):
            if f == base_f: continue
            df_rem = pd.read_csv(os.path.join(b_dir, f), index_col=0)
            v_rem_cols = [c for c in df_rem.columns if str(c).startswith(bench)]
            if not v_rem_cols: continue
            df_rem = df_rem[v_rem_cols]
            df_rem.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in df_rem.columns]
            df_rem.index = [normalize_agent_name(a) for a in df_rem.index]
            
            df_iter = df_base.copy()
            df_iter.update(df_rem)
            
            v_f = f.replace('resmat_', 'verdict_')
            rub_f = f.replace('resmat_', 'rubric_score_')
            
            v_data = None
            if os.path.exists(os.path.join(v_dir, v_f)):
                vdf = pd.read_csv(os.path.join(v_dir, v_f), index_col=0)
                judge = vdf[vdf.index == 'judge_verdict']
                if judge.empty and 'agent' in vdf.columns:
                    judge = vdf[vdf['agent'] == 'judge_verdict'].drop(columns=['agent'])
                if not judge.empty:
                    judge.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in judge.columns]
                    v_data = judge
            
            rub_data = None
            if os.path.exists(os.path.join(rub_dir, rub_f)):
                rub_df = pd.read_csv(os.path.join(rub_dir, rub_f), index_col=0)
                rub_df.columns = [f"{bench}.{c}" if not str(c).startswith(bench) and not str(c).startswith(bench.replace('_hard','')) else c for c in rub_df.columns]
                rub_df.index = [normalize_agent_name(a) for a in rub_df.index]
                rub_data = rub_df

            iters.append({'res': df_iter, 'ver': v_data, 'rub': rub_data})
        bench_iters[bench] = iters

    colbench_iters = bench_iters.get('colbench_backend_programming', [])
    all_combined_res = []
    all_combined_ver = []
    all_combined_rub = []
    
    np.random.seed(RANDOM_SEED)
    for c_iter in colbench_iters:
        current_res = [c_iter['res']]
        current_ver = [c_iter.get('ver')]
        current_rub = [c_iter.get('rub')]
        
        for bench, iters in bench_iters.items():
            if bench == 'colbench_backend_programming': continue
            idx = np.random.randint(0, len(iters))
            current_res.append(iters[idx]['res'])
            current_ver.append(iters[idx].get('ver'))
            current_rub.append(iters[idx].get('rub'))
        
        all_combined_res.append(pd.concat(current_res, axis=1, join='outer'))
        all_combined_ver.append(pd.concat([v for v in current_ver if v is not None], axis=1, join='outer'))
        all_combined_rub.append(pd.concat([r for r in current_rub if r is not None], axis=1, join='outer'))

    post_revision_cols = sorted(list(set().union(*[df.columns for df in all_combined_res])))
    post_revision_agents = sorted(list(set().union(*[df.index for df in all_combined_res])))
    
    stacked_res = np.array([df.reindex(index=post_revision_agents, columns=post_revision_cols).values for df in all_combined_res], dtype=float)
    post_revision_res_val = np.nanmean(stacked_res, axis=0) if all_combined_res else np.full((len(post_revision_agents), len(post_revision_cols)), np.nan)
    
    stacked_ver = np.array([df.reindex(columns=post_revision_cols).values for df in all_combined_ver if not df.empty], dtype=float)
    post_revision_ver_val = np.nanmean(stacked_ver, axis=0) if len(stacked_ver) > 0 else np.full((1, len(post_revision_cols)), np.nan)

    stacked_rub = np.array([df.reindex(index=post_revision_agents, columns=post_revision_cols).values for df in all_combined_rub if not df.empty], dtype=float)
    post_revision_rub_val = np.nanmean(stacked_rub, axis=0) if len(stacked_rub) > 0 else np.full((len(post_revision_agents), len(post_revision_cols)), np.nan)

    return {
        'pre_revision_res': pre_revision_res_val,
        'pre_revision_ver': pre_revision_ver_val,
        'pre_revision_rub': pre_revision_rub_val,
        'pre_revision_agents': pre_revision_agents,
        'pre_revision_cols': pre_revision_cols,
        'post_revision_res': post_revision_res_val,
        'post_revision_ver': post_revision_ver_val,
        'post_revision_rub': post_revision_rub_val,
        'post_revision_agents': post_revision_agents,
        'post_revision_cols': post_revision_cols
    }

def reorder_linkage(Z, metrics):
    n = len(metrics)
    node_means = list(metrics)
    node_counts = [1] * n
    for i in range(len(Z)):
        l_idx, r_idx = int(Z[i, 0]), int(Z[i, 1])
        m_l, c_l = node_means[l_idx], node_counts[l_idx]
        m_r, c_r = node_means[r_idx], node_counts[r_idx]
        node_means.append((m_l * c_l + m_r * c_r) / (c_l + c_r))
        node_counts.append(c_l + c_r)
        if node_means[r_idx] < node_means[l_idx]:
            Z[i, 0], Z[i, 1] = r_idx, l_idx
    return Z

def get_saturated_order(sub_mat):
    if sub_mat.shape[0] <= 1: return np.arange(sub_mat.shape[0])
    mask = (~np.isnan(sub_mat)).astype(float)
    clean_mat = np.nan_to_num(sub_mat, nan=0.0)
    clustering_mat = np.concatenate([mask * 10.0, clean_mat], axis=1)
    with np.errstate(all='ignore'):
        metrics = np.nanmean(sub_mat, axis=1)
    metrics = np.nan_to_num(metrics, nan=0.0)
    Z = sch.linkage(clustering_mat, method='ward')
    Z_reordered = reorder_linkage(Z, metrics)
    return sch.leaves_list(Z_reordered)

def plot_matrices(data):
    plt.rcParams.update(get_bundle())
    
    # Yellow/White judge colorbar
    verdict_yellow_cmap = LinearSegmentedColormap.from_list("binary_yellow_white", ["#FFFFFF", pc.YELLOW])
    # Red is 0.0, Blue is 1.0
    verdict_red_cmap = LinearSegmentedColormap.from_list("binary_red_blue", [pc.RED, pc.BLUE])
    
    bench_names = ['colbench', 'corebench_hard', 'scicode', 'scienceagentbench']
    bench_display = {'colbench': 'ColBench', 'corebench_hard': 'CORE', 'scicode': 'SciCode', 'scienceagentbench': 'ScienceAgentBench'}

    plot_configs = [
        ('pre_revision', data['pre_revision_res'], data['pre_revision_ver'], data['pre_revision_rub'], data['pre_revision_agents'], data['pre_revision_cols']), 
        ('post_revision', data['post_revision_res'], data['post_revision_ver'], data['post_revision_rub'], data['post_revision_agents'], data['post_revision_cols'])
    ]
    
    for label, res_raw, ver_raw, rub_raw, agents_list, cols_list in plot_configs:
        if res_raw is None or np.all(np.isnan(res_raw)): continue

        # 1. Filter out columns that are entirely NaN
        has_data = ~np.all(np.isnan(res_raw), axis=0) | (~np.all(np.isnan(rub_raw), axis=0) if rub_raw is not None else False)
        active_idx = np.where(has_data)[0]
        if len(active_idx) == 0: continue
        
        res_active = res_raw[:, active_idx]
        ver_active = ver_raw[:, active_idx]
        rub_active = rub_raw[:, active_idx] if rub_raw is not None else np.full_like(res_active, np.nan)
        cols_active = [cols_list[i] for i in active_idx]

        # 2. Reorder columns by Benchmark + Clustering
        final_cols_order = []
        for b in bench_names:
            b_indices = [i for i, c in enumerate(cols_active) if c.startswith(b) or c.startswith(b.replace('_hard',''))]
            if not b_indices: continue
            clust_idx = get_saturated_order(res_active[:, b_indices].T)
            final_cols_order.extend([b_indices[i] for i in clust_idx])
            
        res_aligned = res_active[:, final_cols_order]
        ver_aligned = ver_active[:, final_cols_order]
        rub_aligned = rub_active[:, final_cols_order]
        cols_f = [cols_active[i] for i in final_cols_order]

        # 3. Row Clustering (Agents)
        valid_rows = ~(np.all(np.isnan(res_aligned), axis=1) & np.all(np.isnan(rub_aligned), axis=1))
        res_v, rub_v = res_aligned[valid_rows], rub_aligned[valid_rows]
        
        if res_v.shape[0] > 0:
            row_idx = get_saturated_order(res_v)
            res_f, rub_f = res_v[row_idx][::-1], rub_v[row_idx][::-1]
        else:
            res_f, rub_f = res_v, rub_v

        # 4. Save Heatmaps
        for plot_type in ['resmat', 'verdict']:
            # [ALIGNMENT FIX]: Both pre and post revision now have 32 agents, so we equate the heights
            h = 4
            hr = 15
            fig, (ax_main, ax_verdict) = plt.subplots(2, 1, figsize=(14, h), gridspec_kw={'height_ratios': [hr, 1], 'hspace': 0.05})
            
            data_to_plot = res_f if plot_type == 'resmat' else rub_f
            
            cbar_asp = 40
            sns.heatmap(data_to_plot, ax=ax_main, cmap=verdict_red_cmap, cbar_kws={'aspect': cbar_asp, 'pad': 0.01}, xticklabels=False, yticklabels=False, vmin=0, vmax=1)
            
            # Yellow Judge Strip
            sns.heatmap(ver_aligned, ax=ax_verdict, cmap=verdict_yellow_cmap, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1)

            for b in bench_names:
                b_idx = [i for i, c in enumerate(cols_f) if c.startswith(b) or c.startswith(b.replace('_hard',''))]
                if b_idx:
                    end = max(b_idx) + 1
                    ax_main.axvline(x=end, color='black', linewidth=0.3, linestyle='--')
                    ax_verdict.axvline(x=end, color='black', linewidth=0.3, linestyle='--')
                    # Label on top of ax_main
                    ax_main.text((min(b_idx) + end)/2, -0.02, bench_display.get(b), ha='center', va='bottom', transform=ax_main.get_xaxis_transform(), fontsize=7)

            plt.savefig(os.path.join(FIGURE_DIR, f"{label}_{plot_type}.pdf"), bbox_inches='tight')
            plt.close()

def main():
    benchmarks = ['colbench_backend_programming', 'corebench_hard', 'scienceagentbench', 'scicode']
    data = collect_data(benchmarks)
    plot_matrices(data)

if __name__ == "__main__":
    main()
