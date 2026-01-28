import matplotlib
matplotlib.use("Agg")

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('..')
colors = sns.color_palette("muted")
import style_icml

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data-reeval-multi', 'hal', 'judge_output')

# ── Discover CSV files and group by dataset prefix ───────────────────────────
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
dataset_iters = {}  # prefix -> {suffix_int: filepath}

for fpath in csv_files:
    fname = os.path.basename(fpath)
    m = re.match(r'^(.+)_(\d+)\.csv$', fname)
    if not m:
        continue
    prefix, suffix = m.group(1), int(m.group(2))
    dataset_iters.setdefault(prefix, {})[suffix] = fpath

# Keep only datasets with more than 1 iteration
multi_iter = {k: v for k, v in dataset_iters.items() if len(v) > 1}
print(f"[INFO] Datasets with multiple iterations: {list(multi_iter.keys())}")

# ── Compute rubric satisfaction rate per iteration ───────────────────────────
display_names = {
    'colbench': 'ColBench',
    'corebench': 'CoreBench',
    'sab': 'SAB',
    'scicode': 'SciCode',
}

records = []
for prefix in sorted(multi_iter):
    iters = multi_iter[prefix]
    for suffix in sorted(iters):
        df = pd.read_csv(iters[suffix])
        rate = df['satisfies_rubric'].mean()
        count = df['satisfies_rubric'].sum()
        total = len(df)
        label = display_names.get(prefix, prefix)
        records.append({
            'Dataset': label,
            'Iteration': suffix,
            'Rate': rate,
            'Count': int(count),
            'Total': total,
        })
        print(f"  {label} iter {suffix}: {count}/{total} = {rate:.2%}")

df_plot = pd.DataFrame(records)

# ── Plot: Grouped bar chart ─────────────────────────────────────────────────
max_iter = df_plot['Iteration'].max()
iter_colors = [colors[i] for i in range(max_iter)]

datasets = df_plot['Dataset'].unique()
n_datasets = len(datasets)

fig, ax = plt.subplots(figsize=(8, 4))

bar_width = 0.22
offsets = np.arange(n_datasets, dtype=float)

for i, it in enumerate(sorted(df_plot['Iteration'].unique())):
    subset = df_plot[df_plot['Iteration'] == it]
    # Align to dataset order
    rates = []
    positions = []
    for j, ds in enumerate(datasets):
        row = subset[subset['Dataset'] == ds]
        if not row.empty:
            rates.append(row['Rate'].values[0])
            positions.append(offsets[j] + i * bar_width)

    ax.bar(positions, rates, width=bar_width, label=f'Iter {it}',
           color=iter_colors[i], edgecolor='white', linewidth=0.5)

    # Value labels on top of bars
    for pos, rate in zip(positions, rates):
        ax.text(pos, rate + 0.01, f'{rate:.0%}', ha='center', va='bottom',
                fontsize=11)

# Center tick labels under each group
center_offset = bar_width * (max_iter - 1) / 2
ax.set_xticks(offsets + center_offset)
ax.set_xticklabels(datasets)

ax.set_ylabel('Rubric Satisfaction Rate')
ax.set_ylim(0, min(1.0, df_plot['Rate'].max() + 0.15))
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(title='Iteration', loc='upper left', framealpha=0.9)

plt.tight_layout()
out = os.path.join(RESULT_DIR, 'judge_iteration_comparison.pdf')
plt.savefig(out, bbox_inches='tight')
print(f"\n[OUTPUT] Saved plot: {out}")
plt.close()
