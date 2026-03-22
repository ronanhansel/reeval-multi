#!/usr/bin/env python3
"""
Observed-pair efficiency plots for beta-only separate study.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result", "pair_efficiency_study")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
PAIR_CSV = os.path.join(RESULT_DIR, "pair_efficiency_beta_grid.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

ARAF_VARIANTS = ['sae', 'pca', 'raw']
CHECKPOINTS = [
    ('4', 0.1, 'Low'),
    ('8', 0.3, 'Low-Mid'),
    ('16', 0.5, 'Mid'),
    ('32', 0.7, 'High-Mid'),
    ('max', 1.0, 'Saturation'),
]


def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_data():
    if not os.path.exists(PAIR_CSV):
        return None
    df = pd.read_csv(PAIR_CSV, on_bad_lines='skip')
    if df.empty:
        return None
    for col in ['lambda_tau', 'n_samples', 'j_percentage', 'observed_train_pairs',
                'auc_knn', 'rmse_knn', 'auc_araf', 'rmse_araf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['embedding_type', 'model_type', 'pre_revision', 'baseline_embedding_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df = df[df['model_type'] == 'beta']
    if df.empty:
        return None
    return df


def select_best_points(df):
    rows = []
    for order_idx, (pre_revision, j_pct, stage_label) in enumerate(CHECKPOINTS):
        sub = df[
            (df['pre_revision'] == str(pre_revision).lower()) &
            (np.isclose(df['j_percentage'], float(j_pct), atol=1e-9))
        ]
        if sub.empty:
            continue
        best_variant = None
        best_tau = None
        best_auc = None
        best_rows = None

        for variant in ARAF_VARIANTS:
            cand = sub[sub['embedding_type'] == variant]
            if cand.empty:
                continue
            tau_scores = cand.groupby('lambda_tau')['auc_araf'].mean().dropna()
            if tau_scores.empty:
                continue
            tau = float(tau_scores.idxmax())
            tau_rows = cand[np.isclose(cand['lambda_tau'], tau, atol=1e-8)]
            mean_auc = float(tau_rows['auc_araf'].mean())
            if best_auc is None or mean_auc > best_auc:
                best_auc = mean_auc
                best_variant = variant
                best_tau = tau
                best_rows = tau_rows

        if best_rows is None:
            continue

        rows.append({
            'stage_order': order_idx,
            'stage_label': stage_label,
            'pre_revision': pre_revision,
            'j_percentage': float(j_pct),
            'best_variant': best_variant,
            'best_tau': best_tau,
            'observed_train_pairs': float(best_rows['observed_train_pairs'].mean()),
            'auc_knn': float(best_rows['auc_knn'].mean()),
            'rmse_knn': float(best_rows['rmse_knn'].mean()),
            'auc_araf': float(best_rows['auc_araf'].mean()),
            'rmse_araf': float(best_rows['rmse_araf'].mean()),
        })

    if not rows:
        return None
    out = pd.DataFrame(rows)
    return out.sort_values('stage_order')


def plot(best_df):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)

    x = np.arange(len(best_df))
    labels = [
        f"{stage}\n{int(pairs):,}"
        for stage, pairs in zip(best_df['stage_label'], best_df['observed_train_pairs'])
    ]

    axes[0].plot(x, best_df['auc_knn'], color='orange', marker='o', linewidth=1.3, label='kNN')
    axes[0].plot(x, best_df['auc_araf'], color='steelblue', marker='o', linewidth=1.3, label='ARAF')
    axes[0].set_title('AUC', fontsize=10)
    axes[0].set_ylabel('AUC', fontsize=10)

    axes[1].plot(x, best_df['rmse_knn'], color='orange', marker='o', linewidth=1.3, label='kNN')
    axes[1].plot(x, best_df['rmse_araf'], color='steelblue', marker='o', linewidth=1.3, label='ARAF')
    axes[1].set_title('RMSE', fontsize=10)
    axes[1].set_ylabel('RMSE', fontsize=10)

    for ax in axes:
        ax.grid(linestyle=':', alpha=0.8)
        ax.tick_params(labelsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel('Observed-Pair Stage (mean train pairs)', fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, 'pair_efficiency_beta_grid.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Pair-efficiency figure saved to {out_path}")


def main():
    df = load_data()
    if df is None or df.empty:
        print("Pair-efficiency data missing or incomplete; skipping pair-efficiency plot.")
        return
    best_df = select_best_points(df)
    if best_df is None or best_df.empty:
        print("Pair-efficiency selection produced no rows; skipping pair-efficiency plot.")
        return
    plot(best_df)


if __name__ == '__main__':
    main()
