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
    if 'baseline_embedding_type' not in df.columns:
        df['baseline_embedding_type'] = 'raw'
    if 'knn_k' not in df.columns:
        df['knn_k'] = 10
    for col in ['lambda_tau', 'n_samples', 'j_percentage', 'train_retention', 'knn_k',
                'observed_train_pairs', 'auc_knn', 'rmse_knn', 'auc_araf', 'rmse_araf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['embedding_type', 'baseline_embedding_type', 'model_type', 'pre_revision']:
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

            araf_grouped = sub.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
                auc_araf=('auc_araf', 'mean'),
                rmse_araf=('rmse_araf', 'mean'),
                observed_train_pairs=('observed_train_pairs', 'mean'),
            )
            knn_grouped = sub.groupby(['baseline_embedding_type', 'knn_k'], as_index=False).agg(
                auc_knn=('auc_knn', 'mean'),
                rmse_knn=('rmse_knn', 'mean'),
                observed_train_pairs=('observed_train_pairs', 'mean'),
            )

            auc_araf_pick = araf_grouped.sort_values(['auc_araf', 'rmse_araf'], ascending=[False, True]).iloc[0]
            auc_knn_pick = knn_grouped.sort_values(['auc_knn', 'rmse_knn'], ascending=[False, True]).iloc[0]
            rmse_araf_pick = araf_grouped.sort_values(['rmse_araf', 'auc_araf'], ascending=[True, False]).iloc[0]
            rmse_knn_pick = knn_grouped.sort_values(['rmse_knn', 'auc_knn'], ascending=[True, False]).iloc[0]

            auc_gap_row.append(float(auc_araf_pick['auc_araf'] - auc_knn_pick['auc_knn']))
            rmse_gap_row.append(float(rmse_knn_pick['rmse_knn'] - rmse_araf_pick['rmse_araf']))
            obs_row.append(float(sub['observed_train_pairs'].mean()))

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


def build_auc_curves(df):
    curves = []
    for pre_revision, j_pct, stage_label in CHECKPOINTS:
        stage = df[
            (df['pre_revision'] == str(pre_revision).lower()) &
            (np.isclose(df['j_percentage'], float(j_pct), atol=1e-9))
        ]
        if stage.empty:
            continue

        rows = []
        for retention in RETENTIONS:
            sub = stage[np.isclose(stage['train_retention'], float(retention), atol=1e-9)]
            if sub.empty:
                continue

            araf_grouped = sub.groupby(['embedding_type', 'lambda_tau'], as_index=False).agg(
                auc_araf=('auc_araf', 'mean'),
                rmse_araf=('rmse_araf', 'mean'),
            )
            knn_grouped = sub.groupby(['baseline_embedding_type', 'knn_k'], as_index=False).agg(
                auc_knn=('auc_knn', 'mean'),
                rmse_knn=('rmse_knn', 'mean'),
            )
            araf_pick = araf_grouped.sort_values(['auc_araf', 'rmse_araf'], ascending=[False, True]).iloc[0]
            knn_pick = knn_grouped.sort_values(['auc_knn', 'rmse_knn'], ascending=[False, True]).iloc[0]
            araf_chosen = sub[
                (sub['embedding_type'] == araf_pick['embedding_type']) &
                (np.isclose(sub['lambda_tau'], float(araf_pick['lambda_tau']), atol=1e-8))
            ]
            knn_chosen = sub[
                (sub['baseline_embedding_type'] == knn_pick['baseline_embedding_type']) &
                (np.isclose(sub['knn_k'], float(knn_pick['knn_k']), atol=1e-8))
            ]

            seed_auc_araf = araf_chosen.groupby('seed', as_index=False)['auc_araf'].mean()['auc_araf']
            seed_auc_knn = knn_chosen.groupby('seed', as_index=False)['auc_knn'].mean()['auc_knn']
            rows.append({
                'retention': float(retention),
                'observed_train_pairs': float(sub['observed_train_pairs'].mean()),
                'auc_araf': float(seed_auc_araf.mean()),
                'auc_araf_sem': float(seed_auc_araf.sem()) if len(seed_auc_araf) > 1 else 0.0,
                'auc_knn': float(seed_auc_knn.mean()),
                'auc_knn_sem': float(seed_auc_knn.sem()) if len(seed_auc_knn) > 1 else 0.0,
                'embedding_type': str(araf_pick['embedding_type']),
                'lambda_tau': float(araf_pick['lambda_tau']),
                'baseline_embedding_type': str(knn_pick['baseline_embedding_type']),
                'knn_k': int(knn_pick['knn_k']),
            })

        if rows:
            curve_df = pd.DataFrame(rows).sort_values('observed_train_pairs', ascending=True)
            curve_df['stage_label'] = stage_label
            curves.append(curve_df)
    return curves


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


def plot_auc_curves(curves):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, len(curves), figsize=(12.0, 2.6), constrained_layout=True, sharey=True)
    if len(curves) == 1:
        axes = [axes]

    for ax, curve in zip(axes, curves):
        x = curve['observed_train_pairs'].to_numpy(dtype=float)
        knn_mean = curve['auc_knn'].to_numpy(dtype=float)
        knn_sem = curve['auc_knn_sem'].to_numpy(dtype=float)
        araf_mean = curve['auc_araf'].to_numpy(dtype=float)
        araf_sem = curve['auc_araf_sem'].to_numpy(dtype=float)

        ax.plot(x, knn_mean, color='orange', linestyle='--', marker='o', linewidth=1.4, label='kNN')
        ax.fill_between(x, knn_mean - knn_sem, knn_mean + knn_sem, color='orange', alpha=0.18)
        ax.plot(x, araf_mean, color='steelblue', marker='o', linewidth=1.4, label='ARAF')
        ax.fill_between(x, araf_mean - araf_sem, araf_mean + araf_sem, color='steelblue', alpha=0.18)
        ax.set_title(curve['stage_label'].iloc[0], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(round(v)):,}" for v in x], rotation=45, ha='right', fontsize=8)
        ax.grid(linestyle=':', alpha=0.8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('Mean Observed Train Pairs', fontsize=9)

    axes[0].set_ylabel('AUC', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, 'support_thinning_beta_auc.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Support-thinning AUC figure saved to {out_path}")


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
    curves = build_auc_curves(df)
    if curves:
        plot_auc_curves(curves)


if __name__ == '__main__':
    main()
