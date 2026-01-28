import matplotlib
matplotlib.use("Agg")

import os
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

muted_blue   = colors[0]
muted_orange = colors[1]
muted_green  = colors[2]
muted_red    = colors[3]

# Hard-coded HELM Benchmark Data
data = {
    'Model': [
        'Average', 'Rasch-IRT', 'Amortised Difficulty',
        'Sub-Amortised IRT', 'Amortised IRT'
    ],
    'AUC': [0.6579, 0.6539, 0.7577, 0.7823, 0.8122],
}

df_helm = pd.DataFrame(data)

model_order = ['Average', 'Rasch-IRT', 'Amortised Difficulty', 'Sub-Amortised IRT', 'Amortised IRT']

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    data=df_helm,
    x='Model',
    y='AUC',
    order=model_order,
    ax=ax,
    color=muted_blue
)
ax.set_xlabel('')
ax.set_ylabel('AUC')
ax.set_ylim(0.5, 1)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.tick_params(axis='x', rotation=15)

for i, v in enumerate(df_helm.set_index('Model').loc[model_order]['AUC']):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
out = os.path.join(RESULT_DIR, 'auc_comparison_helm.pdf')
plt.savefig(out, bbox_inches='tight')
print(f"[OUTPUT] Saved plot: {out}")
plt.close()
