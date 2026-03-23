#!/usr/bin/env python3
"""
Train-observation thinning study plots for beta checkpoint configs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result", "support_thinning_study")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
THIN_CSV = os.path.join(RESULT_DIR, "support_thinning_beta_grid.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

CHECKPOINTS = [
    ('4', 0.1, 'Low'),
    ('8', 0.3, 'Low-Mid'),
    ('16', 0.5, 'Mid'),
    ('32', 0.7, 'High-Mid'),
    ('max', 1.0, 'Saturation'),
]
RETENTIONS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]


def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_data():
    if not os.path.exists(THIN_CSV):
        return None
    df = pd.read_csv(THIN_CSV, low_memory=False)
    if df.empty:
        return None
    for col in ['lambda_tau', 'n_samples', 'j_percentage', 'train_retention',
                'observed_train_pairs', 'auc_knn', 'rmse_knn', 'auc_araf', 'rmse_araf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['embedding_type', 'model_type', 'pre_revision']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df = df[df['model_type'] == 'beta']
    if df.empty:
        return None
    return df


def build_gap_matrices(df):
    auc_rows = []
    rmse_rows = []
    observed_rows = []
    stage_labels = []

    for pre_revision, j_pct, stage_label in CHECKPOINTS:
        stage = df[
            (df['pre_revision'] == str(pre_revision).lower()) &
            (np.isclose(df['j_percentage'], float(j_pct), atol=1e-9))
        ]
        if stage.empty:
            continue

        auc_gap_row = []
        rmse_gap_row = []
        obs_row = []
        for retention in RETENTIONS:
            sub = stage[np.isclose(stage['train_retention'], float(retention), atol=1e-9)]
            if sub.empty:
                auc_gap_row.append(np.nan)
                rmse_gap_row.append(np.nan)
                obs_row.append(np.nan)
                continue

            grouped = sub.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
                auc_araf=('auc_araf', 'mean'),
                rmse_araf=('rmse_araf', 'mean'),
                auc_knn=('auc_knn', 'mean'),
                rmse_knn=('rmse_knn', 'mean'),
                observed_train_pairs=('observed_train_pairs', 'mean'),
            )

            auc_pick = grouped.sort_values(['auc_araf', 'rmse_araf'], ascending=[False, True]).iloc[0]
            rmse_pick = grouped.sort_values(['rmse_araf', 'auc_araf'], ascending=[True, False]).iloc[0]

            auc_gap_row.append(float(auc_pick['auc_araf'] - auc_pick['auc_knn']))
            rmse_gap_row.append(float(rmse_pick['rmse_knn'] - rmse_pick['rmse_araf']))
            obs_row.append(float(auc_pick['observed_train_pairs']))

        auc_rows.append(auc_gap_row)
        rmse_rows.append(rmse_gap_row)
        observed_rows.append(obs_row)
        stage_labels.append(stage_label)

    if not auc_rows:
        return None, None, None, None
    return (
        np.array(auc_rows, dtype=float),
        np.array(rmse_rows, dtype=float),
        np.array(observed_rows, dtype=float),
        stage_labels,
    )


def _draw_heatmap(ax, matrix, title, stage_labels):
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(RETENTIONS)))
    ax.set_xticklabels([f"{int(r * 100)}%" for r in RETENTIONS], fontsize=8)
    ax.set_yticks(np.arange(len(stage_labels)))
    ax.set_yticklabels(stage_labels, fontsize=8)
    ax.set_xlabel('Train Retention', fontsize=9)
    ax.set_ylabel('Observed-Pair Stage', fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            label = 'NA' if np.isnan(val) else f"{val:+.02f}"
            ax.text(j, i, label, ha='center', va='center', fontsize=7, color='black')
    return im


def plot(auc_gap, rmse_gap, observed_pairs, stage_labels):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)

    im0 = _draw_heatmap(axes[0], auc_gap, 'Delta AUC (ARAF - kNN)', stage_labels)
    im1 = _draw_heatmap(axes[1], rmse_gap, 'Delta RMSE (kNN - ARAF)', stage_labels)

    obs_im = axes[2].imshow(observed_pairs, aspect='auto', cmap='Blues')
    axes[2].set_title('Observed Train Pairs', fontsize=10)
    axes[2].set_xticks(np.arange(len(RETENTIONS)))
    axes[2].set_xticklabels([f"{int(r * 100)}%" for r in RETENTIONS], fontsize=8)
    axes[2].set_yticks(np.arange(len(stage_labels)))
    axes[2].set_yticklabels(stage_labels, fontsize=8)
    axes[2].set_xlabel('Train Retention', fontsize=9)
    axes[2].set_ylabel('Observed-Pair Stage', fontsize=9)
    for i in range(observed_pairs.shape[0]):
        for j in range(observed_pairs.shape[1]):
            val = observed_pairs[i, j]
            label = 'NA' if np.isnan(val) else f"{int(round(val))}"
            axes[2].text(j, i, label, ha='center', va='center', fontsize=7, color='black')

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(obs_im, ax=axes[2], fraction=0.046, pad=0.04)

    out_path = os.path.join(FIGURE_DIR, 'support_thinning_beta_heatmap.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Support-thinning figure saved to {out_path}")


def main():
    df = load_data()
    if df is None or df.empty:
        print("Support-thinning data missing or incomplete; skipping thinning-study plot.")
        return
    auc_gap, rmse_gap, observed_pairs, stage_labels = build_gap_matrices(df)
    if auc_gap is None:
        print("Support-thinning selection produced no rows; skipping thinning-study plot.")
        return
    plot(auc_gap, rmse_gap, observed_pairs, stage_labels)


if __name__ == '__main__':
    main()
