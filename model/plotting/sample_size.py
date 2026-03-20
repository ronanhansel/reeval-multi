#!/usr/bin/env python3
"""
Beta-only sample-size scaling plot.

Generates a 4-panel figure with:
- AUC vs Number of Agents
- AUC vs Percentage of Items
- RMSE vs Number of Agents
- RMSE vs Percentage of Items
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles

# ══════════════════════════════════════════════════════════════════════════════
# Config & Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)
RESULT_DIR = os.path.join(MODEL_DIR, 'result')
BASELINE_PATH = os.path.join(RESULT_DIR, 'baselines', 'baseline_metrics.csv')
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Beta setup only
SIZES_BETA = ['4', '8', '16', '32', '64', 'max']
X_VALS_BETA = [4, 8, 16, 32, 64, 143]
J_PCTS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

MODELS = ['sae', 'pca', 'raw']
MODEL_LABELS = {'sae': 'ARAF (SAE)', 'pca': 'ARAF (PCA)', 'raw': 'ARAF (RAW)'}
MODEL_COLORS = {'sae': "lightblue", 'pca': "deepskyblue", 'raw': "steelblue"}
BASELINE_GRAY = 'slategray'
BASELINE_KEYS = ['rasch', 'mirt', 'knn']
BASELINE_LABELS = {
    'rasch': 'Rasch',
    'mirt': 'MIRT',
    'knn': 'kNN',
}
BASELINE_COLORS = {
    'rasch': '#B9C4CC',
    'mirt': '#D3D8DE',
    'knn': '#5F6F7A',
}
BASELINE_MARKERS = {
    'rasch': 'o',
    'mirt': 'o',
    'knn': 'o',
}
BASELINE_LINE_ALPHA = {
    'rasch': 0.65,
    'mirt': 0.6,
    'knn': 0.9,
}
BASELINE_FILL_ALPHA = {
    'rasch': 0.08,
    'mirt': 0.06,
    'knn': 0.12,
}

# Typography
FONT_SIZE_TITLE = 11
FONT_SIZE_TICK = 10
FONT_SIZE_ANNOTATION = 10
FONT_SIZE_GROUP_LABEL = 11
FONT_SIZE_LEGEND = 10

# Marker sizing
BASELINE_MARKER_SIZE = 2.75
MODEL_MARKER_SIZE = 3.0


def _load_metrics_from_file(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        if df.empty:
            return None
        if 'lambda_tau' in df.columns and 'auc_amortized' in df.columns:
            tau_stats = df.groupby('lambda_tau')['auc_amortized'].mean()
            best_tau = tau_stats.idxmax()
            df = df[df['lambda_tau'] == best_tau]

        metrics = {}
        for metric in ['auc', 'rmse']:
            col = f'{metric}_amortized'
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if not vals.empty:
                    metrics[metric] = (float(vals.mean()), float(vals.sem() if len(vals) > 1 else 0.0))
                else:
                    metrics[metric] = (None, None)
            else:
                metrics[metric] = (None, None)
        return metrics
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def load_baseline_cache():
    if not os.path.exists(BASELINE_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(BASELINE_PATH, on_bad_lines='skip')
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    for col in ['model_type', 'pre_revision']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    for col in ['n_samples', 'j_percentage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def baseline_stats(df, metric_col, model_type='beta', n_samples=1, pre_revision='none', j_percentage=None):
    if df.empty or metric_col not in df.columns:
        return None, None

    sub = df[(df['model_type'] == str(model_type).lower())]
    if 'n_samples' in sub.columns and n_samples is not None:
        sub = sub[sub['n_samples'] == float(n_samples)]
    if 'pre_revision' in sub.columns and pre_revision is not None:
        sub = sub[sub['pre_revision'] == str(pre_revision).strip().lower()]
    if 'j_percentage' in sub.columns and j_percentage is not None:
        sub = sub[np.isclose(sub['j_percentage'].astype(float), float(j_percentage), atol=1e-9)]

    vals = pd.to_numeric(sub[metric_col], errors='coerce').dropna()
    if vals.empty:
        return None, None
    return float(vals.mean()), float(vals.sem() if len(vals) > 1 else 0.0)


def load_beta_agent_metrics(model, size):
    n_suffix = 'n_max'
    options = [
        os.path.join(RESULT_DIR, f'amortized_irt_{model}_beta_pre_{size}_{n_suffix}.csv'),
        os.path.join(RESULT_DIR, f'amortized_irt_{model}_beta_n_max.csv'),
    ]
    for path in options:
        metrics = _load_metrics_from_file(path)
        if metrics is not None:
            return metrics
    return None


def load_beta_item_metrics(model, j_pct):
    # 100% uses the original non-subsampled file (no _j suffix).
    if float(j_pct) >= 1.0:
        filename = f'amortized_irt_{model}_beta_pre_32_n_max.csv'
    else:
        filename = f'amortized_irt_{model}_beta_pre_32_n_max_j{j_pct}.csv'
    path = os.path.join(RESULT_DIR, filename)
    return _load_metrics_from_file(path)


def gather_beta_agent_data():
    data = {
        'auc': {m: [] for m in MODELS},
        'rmse': {m: [] for m in MODELS},
        'baselines': {
            'auc': {b: [] for b in BASELINE_KEYS},
            'rmse': {b: [] for b in BASELINE_KEYS},
        },
    }
    baseline_df = load_baseline_cache()
    for i, size in enumerate(SIZES_BETA):
        x = X_VALS_BETA[i]
        for model in MODELS:
            metrics = load_beta_agent_metrics(model, size)
            if metrics is None:
                continue
            for metric in ['auc', 'rmse']:
                mean_val, sem_val = metrics[metric]
                if mean_val is not None:
                    data[metric][model].append((x, mean_val, sem_val))

        for metric in ['auc', 'rmse']:
            for b in BASELINE_KEYS:
                mean_val, sem_val = baseline_stats(
                    baseline_df,
                    metric_col=f'{metric}_{b}',
                    model_type='beta',
                    n_samples=1,
                    pre_revision=size,
                    j_percentage=1.0,
                )
                if mean_val is not None:
                    data['baselines'][metric][b].append((x, mean_val, sem_val))
    return data


def gather_beta_item_data():
    data = {
        'auc': {m: [] for m in MODELS},
        'rmse': {m: [] for m in MODELS},
        'baselines': {
            'auc': {b: [] for b in BASELINE_KEYS},
            'rmse': {b: [] for b in BASELINE_KEYS},
        },
    }
    baseline_df = load_baseline_cache()
    for j_pct in J_PCTS:
        x = int(round(j_pct * 100))
        for model in MODELS:
            metrics = load_beta_item_metrics(model, j_pct)
            if metrics is None:
                continue
            for metric in ['auc', 'rmse']:
                mean_val, sem_val = metrics[metric]
                if mean_val is not None:
                    data[metric][model].append((x, mean_val, sem_val))

        for metric in ['auc', 'rmse']:
            for b in BASELINE_KEYS:
                mean_val, sem_val = baseline_stats(
                    baseline_df,
                    metric_col=f'{metric}_{b}',
                    model_type='beta',
                    n_samples=1,
                    pre_revision='32',
                    j_percentage=j_pct,
                )
                if mean_val is not None:
                    data['baselines'][metric][b].append((x, mean_val, sem_val))
    return data


def plot_model_curves(ax, series_by_model, baseline_series, metric, title, x_ticks, x_ticklabels, log2_x=False):
    for b in BASELINE_KEYS:
        series = baseline_series.get(b, [])
        if not series:
            continue
        x, y, e = zip(*series)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        e = np.asarray(e, dtype=float)
        ax.fill_between(
            x,
            y - e,
            y + e,
            color=BASELINE_COLORS[b],
            alpha=BASELINE_FILL_ALPHA[b],
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=BASELINE_COLORS[b],
            label=BASELINE_LABELS[b],
            marker=BASELINE_MARKERS[b],
            linestyle='--',
            markersize=BASELINE_MARKER_SIZE,
            alpha=BASELINE_LINE_ALPHA[b],
            zorder=2,
        )

    for model in MODELS:
        series = series_by_model.get(model, [])
        if not series:
            continue
        x, y, e = zip(*series)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        e = np.asarray(e, dtype=float)
        ax.fill_between(
            x,
            y - e,
            y + e,
            color=MODEL_COLORS[model],
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            marker='o',
            linestyle='-',
            markersize=MODEL_MARKER_SIZE,
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
        )

    if metric == 'auc':
        ax.axhline(0.5, color=BASELINE_GRAY, linestyle='--', linewidth=1, alpha=0.8)
        ax.text(
            0.5,
            0.505,
            'Naive Baseline',
            transform=ax.get_yaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=FONT_SIZE_ANNOTATION,
            color=BASELINE_GRAY,
        )

    if log2_x:
        ax.set_xscale('log', base=2)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)

    if metric == 'auc':
        ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_ylim(0.48, 0.78)
    else:
        ax.set_yticks([0.4, 0.5, 0.6])
        ax.set_ylim(0.38, 0.65)

    ax.grid(True, axis='y', linestyle=':', alpha=0.6)


def plot_combined_beta_quad():
    print('Generating Beta-only 4-panel plot (AUC/RMSE by Agents and Item Percentage)...')

    beta_agents = gather_beta_agent_data()
    beta_items = gather_beta_item_data()

    plt.rcParams.update(bundles.icml2024(usetex=False, family='serif'))
    # Keep the original formatting style: one row with four panels.
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), constrained_layout=False)
    ax1, ax2, ax3, ax4 = axes

    plot_model_curves(
        ax1,
        beta_agents['auc'],
        beta_agents['baselines']['auc'],
        metric='auc',
        title='AUC',
        x_ticks=X_VALS_BETA,
        x_ticklabels=[str(v) for v in X_VALS_BETA[:-1]] + ['143'],
        log2_x=True,
    )

    plot_model_curves(
        ax2,
        beta_agents['rmse'],
        beta_agents['baselines']['rmse'],
        metric='rmse',
        title='RMSE',
        x_ticks=X_VALS_BETA,
        x_ticklabels=[str(v) for v in X_VALS_BETA[:-1]] + ['143'],
        log2_x=True,
    )

    plot_model_curves(
        ax3,
        beta_items['auc'],
        beta_items['baselines']['auc'],
        metric='auc',
        title='AUC',
        x_ticks=[10, 30, 50, 70, 90],
        x_ticklabels=['10', '30', '50', '70', '90'],
        log2_x=False,
    )

    plot_model_curves(
        ax4,
        beta_items['rmse'],
        beta_items['baselines']['rmse'],
        metric='rmse',
        title='RMSE',
        x_ticks=[10, 30, 50, 70, 90],
        x_ticklabels=['10', '30', '50', '70', '90'],
        log2_x=False,
    )

    # Sync Y-scales by metric pair to match original behavior.
    ax3.set_ylim(ax1.get_ylim())
    ax4.set_ylim(ax2.get_ylim())

    # Grouped shared x-axis labels centered from the actual subplot geometry.
    left_pair_center = 0.5 * (ax1.get_position().x0 + ax2.get_position().x1)
    right_pair_center = 0.5 * (ax3.get_position().x0 + ax4.get_position().x1)
    fig.text(left_pair_center, 0.18, 'Number of Agents ($N$)', ha='center', va='center', fontsize=FONT_SIZE_GROUP_LABEL)
    fig.text(right_pair_center, 0.18, 'Percentage of Items ($J\\%$)', ha='center', va='center', fontsize=FONT_SIZE_GROUP_LABEL)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
    )

    plt.subplots_adjust(bottom=0.32, top=0.88, wspace=0.35)

    out_pdf = os.path.join(FIGURE_DIR, 'sample_size_quad.pdf')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Success! 4-panel beta plot generated: {out_pdf}')

def main():
    plot_combined_beta_quad()

if __name__ == "__main__":
    main()
