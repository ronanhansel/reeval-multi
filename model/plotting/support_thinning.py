#!/usr/bin/env python3
"""
Train-observation thinning ladder plots for beta.
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

THIN_PRE_REVISION = "max"
THIN_J_PERCENTAGE = 1.0
RETENTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]


def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_data():
    if not os.path.exists(THIN_CSV):
        return None
    df = pd.read_csv(THIN_CSV, low_memory=False)
    if df.empty:
        return None
    if 'baseline_embedding_type' not in df.columns:
        df['baseline_embedding_type'] = 'raw'
    if 'knn_k' not in df.columns:
        df['knn_k'] = 10
    for col in [
        'lambda_tau', 'n_samples', 'j_percentage', 'train_retention', 'knn_k',
        'observed_train_pairs', 'auc_knn', 'rmse_knn', 'auc_araf', 'rmse_araf'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['embedding_type', 'baseline_embedding_type', 'model_type', 'pre_revision']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df = df[
        (df['model_type'] == 'beta') &
        (df['pre_revision'] == THIN_PRE_REVISION) &
        (np.isclose(df['j_percentage'], THIN_J_PERCENTAGE, atol=1e-9))
    ].copy()
    if df.empty:
        return None
    return df


def _select_best_rows(sub, metric):
    araf_grouped = sub.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
        auc_araf=('auc_araf', 'mean'),
        rmse_araf=('rmse_araf', 'mean'),
    )
    knn_grouped = sub.groupby(['baseline_embedding_type', 'knn_k'], as_index=False).agg(
        auc_knn=('auc_knn', 'mean'),
        rmse_knn=('rmse_knn', 'mean'),
    )

    if metric == 'auc':
        araf_pick = araf_grouped.sort_values(['auc_araf', 'rmse_araf'], ascending=[False, True]).iloc[0]
        knn_pick = knn_grouped.sort_values(['auc_knn', 'rmse_knn'], ascending=[False, True]).iloc[0]
    else:
        araf_pick = araf_grouped.sort_values(['rmse_araf', 'auc_araf'], ascending=[True, False]).iloc[0]
        knn_pick = knn_grouped.sort_values(['rmse_knn', 'auc_knn'], ascending=[True, False]).iloc[0]

    araf_rows = sub[
        (sub['embedding_type'] == araf_pick['embedding_type']) &
        (np.isclose(sub['lambda_tau'], float(araf_pick['lambda_tau']), atol=1e-8))
    ]
    knn_rows = sub[
        (sub['baseline_embedding_type'] == knn_pick['baseline_embedding_type']) &
        (np.isclose(sub['knn_k'], float(knn_pick['knn_k']), atol=1e-8))
    ]
    return araf_rows, knn_rows, araf_pick, knn_pick


def build_curves(df):
    auc_rows = []
    rmse_rows = []

    for retention in RETENTIONS:
        sub = df[np.isclose(df['train_retention'], float(retention), atol=1e-9)]
        if sub.empty:
            continue

        auc_araf_rows, auc_knn_rows, auc_araf_pick, auc_knn_pick = _select_best_rows(sub, 'auc')
        rmse_araf_rows, rmse_knn_rows, rmse_araf_pick, rmse_knn_pick = _select_best_rows(sub, 'rmse')

        seed_auc_araf = auc_araf_rows.groupby('seed', as_index=False)['auc_araf'].mean()['auc_araf']
        seed_auc_knn = auc_knn_rows.groupby('seed', as_index=False)['auc_knn'].mean()['auc_knn']
        seed_rmse_araf = rmse_araf_rows.groupby('seed', as_index=False)['rmse_araf'].mean()['rmse_araf']
        seed_rmse_knn = rmse_knn_rows.groupby('seed', as_index=False)['rmse_knn'].mean()['rmse_knn']

        observed_train_pairs = float(sub['observed_train_pairs'].mean())

        auc_rows.append({
            'retention': float(retention),
            'observed_train_pairs': observed_train_pairs,
            'araf_mean': float(seed_auc_araf.mean()),
            'araf_sem': float(seed_auc_araf.sem()) if len(seed_auc_araf) > 1 else 0.0,
            'knn_mean': float(seed_auc_knn.mean()),
            'knn_sem': float(seed_auc_knn.sem()) if len(seed_auc_knn) > 1 else 0.0,
            'araf_embedding_type': str(auc_araf_pick['embedding_type']),
            'araf_lambda_tau': float(auc_araf_pick['lambda_tau']),
            'knn_embedding_type': str(auc_knn_pick['baseline_embedding_type']),
            'knn_k': int(auc_knn_pick['knn_k']),
        })
        rmse_rows.append({
            'retention': float(retention),
            'observed_train_pairs': observed_train_pairs,
            'araf_mean': float(seed_rmse_araf.mean()),
            'araf_sem': float(seed_rmse_araf.sem()) if len(seed_rmse_araf) > 1 else 0.0,
            'knn_mean': float(seed_rmse_knn.mean()),
            'knn_sem': float(seed_rmse_knn.sem()) if len(seed_rmse_knn) > 1 else 0.0,
            'araf_embedding_type': str(rmse_araf_pick['embedding_type']),
            'araf_lambda_tau': float(rmse_araf_pick['lambda_tau']),
            'knn_embedding_type': str(rmse_knn_pick['baseline_embedding_type']),
            'knn_k': int(rmse_knn_pick['knn_k']),
        })

    if not auc_rows:
        return None, None

    auc_df = pd.DataFrame(auc_rows).sort_values('observed_train_pairs')
    rmse_df = pd.DataFrame(rmse_rows).sort_values('observed_train_pairs')
    return auc_df, rmse_df


def _plot_metric(ax, df, y_label, title, lower_better=False):
    x = df['observed_train_pairs'].to_numpy(dtype=float)
    araf_mean = df['araf_mean'].to_numpy(dtype=float)
    araf_sem = df['araf_sem'].to_numpy(dtype=float)
    knn_mean = df['knn_mean'].to_numpy(dtype=float)
    knn_sem = df['knn_sem'].to_numpy(dtype=float)

    ax.plot(x, knn_mean, color='orange', linestyle='--', marker='o', linewidth=1.6, label='kNN')
    ax.fill_between(x, knn_mean - knn_sem, knn_mean + knn_sem, color='orange', alpha=0.18)
    ax.plot(x, araf_mean, color='steelblue', marker='o', linewidth=1.6, label='ARAF')
    ax.fill_between(x, araf_mean - araf_sem, araf_mean + araf_sem, color='steelblue', alpha=0.18)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Mean Observed Train Pairs', fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(v)):,}" for v in x], rotation=35, ha='right', fontsize=8)
    ax.grid(linestyle=':', alpha=0.8)
    ax.tick_params(labelsize=8)
    if lower_better:
        ymin = min(np.nanmin(araf_mean - araf_sem), np.nanmin(knn_mean - knn_sem))
        ymax = max(np.nanmax(araf_mean + araf_sem), np.nanmax(knn_mean + knn_sem))
        pad = max((ymax - ymin) * 0.08, 1e-3)
        ax.set_ylim(ymin - pad, ymax + pad)


def plot(auc_df, rmse_df):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), constrained_layout=True)

    _plot_metric(axes[0], auc_df, 'AUC', 'AUC (Beta)', lower_better=False)
    _plot_metric(axes[1], rmse_df, 'RMSE', 'RMSE (Beta)', lower_better=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, 'support_thinning_beta_auc.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Support-thinning ladder figure saved to {out_path}")


def main():
    df = load_data()
    if df is None or df.empty:
        print("Support-thinning data missing or incomplete; skipping thinning-study plot.")
        return
    auc_df, rmse_df = build_curves(df)
    if auc_df is None:
        print("Support-thinning selection produced no rows; skipping thinning-study plot.")
        return
    plot(auc_df, rmse_df)


if __name__ == '__main__':
    main()
