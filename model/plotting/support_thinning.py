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
THIN_CSV = os.path.join(RESULT_DIR, "support_thinning_bernoulli_grid.csv")
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
    if 'auc_araf_post' not in df.columns:
        df['auc_araf_post'] = np.nan
    if 'rmse_araf_post' not in df.columns:
        df['rmse_araf_post'] = np.nan
    for col in [
        'lambda_tau', 'n_samples', 'j_percentage', 'train_retention', 'knn_k',
        'observed_train_pairs', 'auc_knn', 'rmse_knn', 'auc_araf', 'rmse_araf',
        'auc_araf_post', 'rmse_araf_post',
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['embedding_type', 'baseline_embedding_type', 'model_type', 'pre_revision']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df = df[
        (df['model_type'] == 'bernoulli') &
        (df['pre_revision'] == THIN_PRE_REVISION) &
        (np.isclose(df['j_percentage'], THIN_J_PERCENTAGE, atol=1e-9))
    ].copy()
    if df.empty:
        return None
    return df


def _select_best_rows(sub, metric):
    transfer_grouped = sub.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
        auc_araf=('auc_araf', 'mean'),
        rmse_araf=('rmse_araf', 'mean'),
    )
    post_source = sub.dropna(subset=['auc_araf_post', 'rmse_araf_post'], how='all')
    post_grouped = post_source.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
        auc_araf_post=('auc_araf_post', 'mean'),
        rmse_araf_post=('rmse_araf_post', 'mean'),
    )
    knn_grouped = sub.groupby(['baseline_embedding_type', 'knn_k'], as_index=False).agg(
        auc_knn=('auc_knn', 'mean'),
        rmse_knn=('rmse_knn', 'mean'),
    )

    if metric == 'auc':
        transfer_pick = transfer_grouped.sort_values(['auc_araf', 'rmse_araf'], ascending=[False, True]).iloc[0]
        post_pick = None if post_grouped.empty else post_grouped.sort_values(
            ['auc_araf_post', 'rmse_araf_post'], ascending=[False, True]
        ).iloc[0]
        knn_pick = knn_grouped.sort_values(['auc_knn', 'rmse_knn'], ascending=[False, True]).iloc[0]
    else:
        transfer_pick = transfer_grouped.sort_values(['rmse_araf', 'auc_araf'], ascending=[True, False]).iloc[0]
        post_pick = None if post_grouped.empty else post_grouped.sort_values(
            ['rmse_araf_post', 'auc_araf_post'], ascending=[True, False]
        ).iloc[0]
        knn_pick = knn_grouped.sort_values(['rmse_knn', 'auc_knn'], ascending=[True, False]).iloc[0]

    transfer_rows = sub[
        (sub['embedding_type'] == transfer_pick['embedding_type']) &
        (np.isclose(sub['lambda_tau'], float(transfer_pick['lambda_tau']), atol=1e-8))
    ]
    post_rows = post_source[
        (post_source['embedding_type'] == post_pick['embedding_type']) &
        (np.isclose(post_source['lambda_tau'], float(post_pick['lambda_tau']), atol=1e-8))
    ] if post_pick is not None else post_source.iloc[0:0]
    knn_rows = sub[
        (sub['baseline_embedding_type'] == knn_pick['baseline_embedding_type']) &
        (np.isclose(sub['knn_k'], float(knn_pick['knn_k']), atol=1e-8))
    ]
    return transfer_rows, post_rows, knn_rows, transfer_pick, post_pick, knn_pick


def build_curves(df):
    auc_rows = []
    rmse_rows = []

    for retention in RETENTIONS:
        sub = df[np.isclose(df['train_retention'], float(retention), atol=1e-9)]
        if sub.empty:
            continue

        auc_transfer_rows, auc_post_rows, auc_knn_rows, auc_transfer_pick, auc_post_pick, auc_knn_pick = _select_best_rows(sub, 'auc')
        rmse_transfer_rows, rmse_post_rows, rmse_knn_rows, rmse_transfer_pick, rmse_post_pick, rmse_knn_pick = _select_best_rows(sub, 'rmse')

        seed_auc_transfer = auc_transfer_rows.groupby('seed', as_index=False)['auc_araf'].mean()['auc_araf']
        seed_auc_post = auc_post_rows.groupby('seed', as_index=False)['auc_araf_post'].mean()['auc_araf_post']
        seed_auc_knn = auc_knn_rows.groupby('seed', as_index=False)['auc_knn'].mean()['auc_knn']
        seed_rmse_transfer = rmse_transfer_rows.groupby('seed', as_index=False)['rmse_araf'].mean()['rmse_araf']
        seed_rmse_post = rmse_post_rows.groupby('seed', as_index=False)['rmse_araf_post'].mean()['rmse_araf_post']
        seed_rmse_knn = rmse_knn_rows.groupby('seed', as_index=False)['rmse_knn'].mean()['rmse_knn']

        observed_train_pairs = float(sub['observed_train_pairs'].mean())

        auc_rows.append({
            'retention': float(retention),
            'observed_train_pairs': observed_train_pairs,
            'transfer_mean': float(seed_auc_transfer.mean()),
            'transfer_sem': float(seed_auc_transfer.sem()) if len(seed_auc_transfer) > 1 else 0.0,
            'post_mean': float(seed_auc_post.mean()) if len(seed_auc_post) else np.nan,
            'post_sem': float(seed_auc_post.sem()) if len(seed_auc_post) > 1 else 0.0,
            'knn_mean': float(seed_auc_knn.mean()),
            'knn_sem': float(seed_auc_knn.sem()) if len(seed_auc_knn) > 1 else 0.0,
            'transfer_embedding_type': str(auc_transfer_pick['embedding_type']),
            'transfer_lambda_tau': float(auc_transfer_pick['lambda_tau']),
            'post_embedding_type': str(auc_post_pick['embedding_type']) if auc_post_pick is not None else '',
            'post_lambda_tau': float(auc_post_pick['lambda_tau']) if auc_post_pick is not None else np.nan,
            'knn_embedding_type': str(auc_knn_pick['baseline_embedding_type']),
            'knn_k': int(auc_knn_pick['knn_k']),
        })
        rmse_rows.append({
            'retention': float(retention),
            'observed_train_pairs': observed_train_pairs,
            'transfer_mean': float(seed_rmse_transfer.mean()),
            'transfer_sem': float(seed_rmse_transfer.sem()) if len(seed_rmse_transfer) > 1 else 0.0,
            'post_mean': float(seed_rmse_post.mean()) if len(seed_rmse_post) else np.nan,
            'post_sem': float(seed_rmse_post.sem()) if len(seed_rmse_post) > 1 else 0.0,
            'knn_mean': float(seed_rmse_knn.mean()),
            'knn_sem': float(seed_rmse_knn.sem()) if len(seed_rmse_knn) > 1 else 0.0,
            'transfer_embedding_type': str(rmse_transfer_pick['embedding_type']),
            'transfer_lambda_tau': float(rmse_transfer_pick['lambda_tau']),
            'post_embedding_type': str(rmse_post_pick['embedding_type']) if rmse_post_pick is not None else '',
            'post_lambda_tau': float(rmse_post_pick['lambda_tau']) if rmse_post_pick is not None else np.nan,
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
    transfer_mean = df['transfer_mean'].to_numpy(dtype=float)
    transfer_sem = df['transfer_sem'].to_numpy(dtype=float)
    post_mean = df['post_mean'].to_numpy(dtype=float)
    post_sem = df['post_sem'].to_numpy(dtype=float)
    knn_mean = df['knn_mean'].to_numpy(dtype=float)
    knn_sem = df['knn_sem'].to_numpy(dtype=float)

    def _plot_series(y, sem, color, label, linestyle='-'):
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        xv = x[valid]
        yv = y[valid]
        sv = sem[valid]
        ax.plot(xv, yv, color=color, linestyle=linestyle, marker='o', linewidth=1.6, label=label)
        ax.fill_between(xv, yv - sv, yv + sv, color=color, alpha=0.18)

    _plot_series(knn_mean, knn_sem, 'orange', 'kNN', '--')
    _plot_series(post_mean, post_sem, 'forestgreen', 'ARAF')
    _plot_series(transfer_mean, transfer_sem, 'steelblue', 'ARAF-transfer')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Mean Observed Post Support Pairs', fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(v)):,}" for v in x], rotation=35, ha='right', fontsize=8)
    ax.grid(linestyle=':', alpha=0.8)
    ax.tick_params(labelsize=8)
    if lower_better:
        lower_candidates = []
        upper_candidates = []
        for mean_vals, sem_vals in [(transfer_mean, transfer_sem), (post_mean, post_sem), (knn_mean, knn_sem)]:
            valid = np.isfinite(mean_vals) & np.isfinite(sem_vals)
            if np.any(valid):
                lower_candidates.append(np.min(mean_vals[valid] - sem_vals[valid]))
                upper_candidates.append(np.max(mean_vals[valid] + sem_vals[valid]))
        ymin = min(lower_candidates)
        ymax = max(upper_candidates)
        pad = max((ymax - ymin) * 0.08, 1e-3)
        ax.set_ylim(ymin - pad, ymax + pad)


def plot(auc_df, rmse_df):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), constrained_layout=True)

    _plot_metric(axes[0], auc_df, 'AUC', 'AUC (Bernoulli)', lower_better=False)
    _plot_metric(axes[1], rmse_df, 'RMSE', 'RMSE (Bernoulli)', lower_better=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, 'support_thinning_bernoulli_auc.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Support-thinning ladder figure saved to {out_path}")


def plot_auc_degradation(auc_df):
    """Plot AUC drop relative to the full-retention (1.0) checkpoint."""
    full_mask = np.isclose(auc_df['retention'].to_numpy(dtype=float), 1.0, atol=1e-9)
    if not np.any(full_mask):
        print("Support-thinning degradation plot skipped: no retention=1.0 row found.")
        return

    full_row = auc_df[full_mask].iloc[0]
    transfer_ref = float(full_row['transfer_mean'])
    post_ref = float(full_row['post_mean']) if np.isfinite(float(full_row['post_mean'])) else np.nan
    knn_ref = float(full_row['knn_mean'])

    x = auc_df['retention'].to_numpy(dtype=float)
    transfer_drop = transfer_ref - auc_df['transfer_mean'].to_numpy(dtype=float)
    post_drop = post_ref - auc_df['post_mean'].to_numpy(dtype=float)
    knn_drop = knn_ref - auc_df['knn_mean'].to_numpy(dtype=float)
    transfer_sem = auc_df['transfer_sem'].to_numpy(dtype=float)
    post_sem = auc_df['post_sem'].to_numpy(dtype=float)
    knn_sem = auc_df['knn_sem'].to_numpy(dtype=float)

    plt.rcParams.update(get_bundle())
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.9), constrained_layout=True)

    ax.plot(x, knn_drop, color='orange', linestyle='--', marker='o', linewidth=1.6, label='kNN')
    ax.fill_between(x, knn_drop - knn_sem, knn_drop + knn_sem, color='orange', alpha=0.18)
    if np.isfinite(post_drop).any():
        ax.plot(x, post_drop, color='forestgreen', marker='o', linewidth=1.6, label='ARAF')
        ax.fill_between(x, post_drop - post_sem, post_drop + post_sem, color='forestgreen', alpha=0.18)
    ax.plot(x, transfer_drop, color='steelblue', marker='o', linewidth=1.6, label='ARAF-transfer')
    ax.fill_between(x, transfer_drop - transfer_sem, transfer_drop + transfer_sem, color='steelblue', alpha=0.18)

    ax.set_title('AUC Degradation vs Retention', fontsize=10)
    ax.set_xlabel('Train Retention Ratio', fontsize=9)
    ax.set_ylabel('AUC Drop From Retention 1.0', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x], fontsize=8)
    ax.grid(linestyle=':', alpha=0.8)
    ax.tick_params(labelsize=8)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, 'support_thinning_bernoulli_auc_degradation.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Support-thinning AUC degradation figure saved to {out_path}")


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
    plot_auc_degradation(auc_df)


if __name__ == '__main__':
    main()
