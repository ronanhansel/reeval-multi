#!/usr/bin/env python3
"""
Neighbor-support study plots for beta checkpoint configs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result", "neighbor_support_study")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
SUPPORT_CSV = os.path.join(RESULT_DIR, "neighbor_support_beta_grid.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

CHECKPOINTS = [
    ('4', 0.1, 'Low'),
    ('8', 0.3, 'Low-Mid'),
    ('16', 0.5, 'Mid'),
    ('32', 0.7, 'High-Mid'),
    ('max', 1.0, 'Saturation'),
]
SUPPORT_BIN_ORDER = ['zero', 'very_low', 'low_mid', 'mid', 'dense']
SUPPORT_BIN_LABELS = ['0%', '(0,20%]', '(20,40%]', '(40,70%]', '(70,100%]']


def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_data():
    if not os.path.exists(SUPPORT_CSV):
        return None
    df = pd.read_csv(SUPPORT_CSV, low_memory=False)
    if df.empty:
        return None
    for col in [
        'lambda_tau', 'n_samples', 'j_percentage', 'num_pairs', 'pair_fraction',
        'mean_coverage_count', 'mean_coverage_rate', 'fallback_rate',
        'auc_knn', 'rmse_knn', 'auc_araf', 'rmse_araf',
        'overall_auc_knn', 'overall_rmse_knn', 'overall_auc_araf', 'overall_rmse_araf'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['embedding_type', 'model_type', 'pre_revision', 'support_bin', 'support_bin_label']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df = df[df['model_type'] == 'beta']
    if df.empty:
        return None
    return df


def _pick_config(sub, metric):
    grouped = sub.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
        overall_auc_araf=('overall_auc_araf', 'mean'),
        overall_rmse_araf=('overall_rmse_araf', 'mean')
    )
    if grouped.empty:
        return None
    if metric == 'auc':
        pick = grouped.sort_values(['overall_auc_araf', 'overall_rmse_araf'], ascending=[False, True]).iloc[0]
    else:
        pick = grouped.sort_values(['overall_rmse_araf', 'overall_auc_araf'], ascending=[True, False]).iloc[0]
    return str(pick['embedding_type']), float(pick['lambda_tau'])


def build_gap_matrices(df):
    auc_rows = []
    rmse_rows = []
    stage_labels = []

    for pre_revision, j_pct, stage_label in CHECKPOINTS:
        sub = df[
            (df['pre_revision'] == str(pre_revision).lower()) &
            (np.isclose(df['j_percentage'], float(j_pct), atol=1e-9))
        ]
        if sub.empty:
            continue

        auc_config = _pick_config(sub, 'auc')
        rmse_config = _pick_config(sub, 'rmse')
        if auc_config is None or rmse_config is None:
            continue

        stage_labels.append(stage_label)

        auc_sub = sub[
            (sub['embedding_type'] == auc_config[0]) &
            (np.isclose(sub['lambda_tau'], auc_config[1], atol=1e-8))
        ]
        rmse_sub = sub[
            (sub['embedding_type'] == rmse_config[0]) &
            (np.isclose(sub['lambda_tau'], rmse_config[1], atol=1e-8))
        ]

        auc_gaps = []
        rmse_gains = []
        for bin_name in SUPPORT_BIN_ORDER:
            auc_bin = auc_sub[auc_sub['support_bin'] == bin_name]
            rmse_bin = rmse_sub[rmse_sub['support_bin'] == bin_name]
            auc_gaps.append(float((auc_bin['auc_araf'] - auc_bin['auc_knn']).mean()) if not auc_bin.empty else np.nan)
            rmse_gains.append(float((rmse_bin['rmse_knn'] - rmse_bin['rmse_araf']).mean()) if not rmse_bin.empty else np.nan)

        auc_rows.append(auc_gaps)
        rmse_rows.append(rmse_gains)

    if not auc_rows or not rmse_rows:
        return None, None, None
    return np.array(auc_rows, dtype=float), np.array(rmse_rows, dtype=float), stage_labels


def _draw_heatmap(ax, matrix, title, stage_labels, cmap):
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(SUPPORT_BIN_LABELS)))
    ax.set_xticklabels(SUPPORT_BIN_LABELS, fontsize=8)
    ax.set_yticks(np.arange(len(stage_labels)))
    ax.set_yticklabels(stage_labels, fontsize=8)
    ax.set_xlabel('Usable Neighbor Support', fontsize=9)
    ax.set_ylabel('Observed-Pair Stage', fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            label = 'NA' if np.isnan(val) else f"{val:+.02f}"
            ax.text(j, i, label, ha='center', va='center', fontsize=7, color='black')
    return im


def plot(auc_gap, rmse_gain, stage_labels):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), constrained_layout=True)

    im0 = _draw_heatmap(axes[0], auc_gap, 'Delta AUC (ARAF - kNN)', stage_labels, cmap='RdBu')
    im1 = _draw_heatmap(axes[1], rmse_gain, 'Delta RMSE (kNN - ARAF)', stage_labels, cmap='RdBu')

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    out_path = os.path.join(FIGURE_DIR, 'neighbor_support_beta_heatmap.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Neighbor-support figure saved to {out_path}")


def main():
    df = load_data()
    if df is None or df.empty:
        print("Neighbor-support data missing or incomplete; skipping support-study plot.")
        return
    auc_gap, rmse_gain, stage_labels = build_gap_matrices(df)
    if auc_gap is None:
        print("Neighbor-support selection produced no rows; skipping support-study plot.")
        return
    plot(auc_gap, rmse_gain, stage_labels)


if __name__ == '__main__':
    main()
