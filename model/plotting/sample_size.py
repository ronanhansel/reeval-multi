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
import io
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
from contextlib import redirect_stdout
from functools import lru_cache

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

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

import amortized_irt as observed_pair_exp

# Beta setup only
SIZES_BETA = ['4', '8', '16', '32', '64', 'max']
X_VALS_BETA = [4, 8, 16, 32, 64, 143]
J_PCTS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

ARAF_VARIANTS = ['sae', 'pca', 'raw']
ARAF_KEY = 'araf'
ARAF_LABEL = 'ARAF'
ARAF_COLOR = 'steelblue'
OBSERVED_PAIR_REGEX = re.compile(r'^amortized_irt_(sae|pca|raw)_beta_n_(max|\d+)(?:_j([0-9.]+))?\.csv$')
BASELINE_GRAY = 'slategray'
BASELINE_KEYS = ['rasch', 'mirt', 'knn']
BASELINE_LABELS = {
    'rasch': 'Rasch',
    'mirt': 'MIRT',
    'knn': 'kNN',
}
BASELINE_COLORS = {
    'rasch': 'wheat',
    'mirt': 'tan',
    'knn': 'orange',
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
FONT_SIZE_GROUP_LABEL = 13
FONT_SIZE_LEGEND = 10

# Marker sizing
BASELINE_MARKER_SIZE = 2.75
MODEL_MARKER_SIZE = 3.0


def _load_metrics_from_file(path):
    if not os.path.exists(path):
        return None
    try:
        df = _load_best_tau_subset(path)
        if df is None:
            return None

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


def _load_best_tau_subset(path):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, on_bad_lines='skip')
    if df.empty:
        return None
    if 'lambda_tau' in df.columns and 'auc_amortized' in df.columns:
        tau_stats = df.groupby('lambda_tau')['auc_amortized'].mean()
        best_tau = tau_stats.idxmax()
        df = df[df['lambda_tau'] == best_tau]
    return df


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


def select_best_araf_point(metric, candidates):
    """Choose the best available ARAF variant for one x-value and metric."""
    best_variant = None
    best_point = None

    for variant, metrics in candidates.items():
        if metrics is None:
            continue
        mean_val, sem_val = metrics.get(metric, (None, None))
        if mean_val is None:
            continue

        if best_point is None:
            best_variant = variant
            best_point = (mean_val, sem_val)
            continue

        best_mean, _ = best_point
        is_better = mean_val > best_mean if metric == 'auc' else mean_val < best_mean
        if is_better:
            best_variant = variant
            best_point = (mean_val, sem_val)

    return best_variant, best_point


def normalize_n_token(n_token):
    return 'max' if str(n_token) == 'max' else str(int(n_token))


def n_token_to_int(n_token):
    n_token = normalize_n_token(n_token)
    if n_token == 'max':
        return len(_get_post_revision_sources()[0])
    return int(n_token)


def discover_observed_pair_sweep_configs():
    configs = {}
    for filename in os.listdir(RESULT_DIR):
        match = OBSERVED_PAIR_REGEX.match(filename)
        if match is None:
            continue

        variant, n_token, j_token = match.groups()
        j_pct = float(j_token) if j_token is not None else 1.0
        key = (normalize_n_token(n_token), float(j_pct))
        configs.setdefault(key, {})[variant] = os.path.join(RESULT_DIR, filename)
    return configs


@lru_cache(maxsize=1)
def _get_post_revision_sources():
    with redirect_stdout(io.StringIO()):
        all_dfs, global_shared_indices, raw_embs_map, _ = observed_pair_exp.load_data(
            embedding_type='pca',
            embedding_dim=48,
            pre_revision='none',
        )
    return all_dfs, global_shared_indices, raw_embs_map


@lru_cache(maxsize=None)
def _recompute_observed_pairs(seed, n_token, j_pct):
    seed = int(seed)
    n_token = normalize_n_token(n_token)
    j_pct = float(j_pct)
    all_dfs, global_shared_indices, raw_embs_map = _get_post_revision_sources()

    observed_pair_exp.RANDOM_SEED = seed
    with redirect_stdout(io.StringIO()):
        data = observed_pair_exp.prepare_experiment_data(
            all_dfs,
            global_shared_indices,
            raw_embs_map,
            embedding_type='pca',
            j_percentage=j_pct,
        )
    np.random.seed(seed)
    _, _, _, train_mask_current_t = observed_pair_exp.build_training_targets(
        n_token_to_int(n_token),
        all_dfs,
        global_shared_indices,
        data,
        model_type='beta',
        quiet=True,
    )
    return int(train_mask_current_t.sum().item())


def load_observed_pair_stats(paths_by_variant, n_token, j_pct):
    seeds = None

    for path in paths_by_variant.values():
        df = _load_best_tau_subset(path)
        if df is None or df.empty:
            continue

        if 'observed_train_pairs' in df.columns:
            vals = pd.to_numeric(df['observed_train_pairs'], errors='coerce').dropna()
            if not vals.empty:
                return float(vals.mean()), float(vals.sem() if len(vals) > 1 else 0.0)

        if 'seed' in df.columns:
            seed_vals = pd.to_numeric(df['seed'], errors='coerce').dropna().astype(int).tolist()
            if seed_vals:
                seeds = sorted(set(seed_vals))

    if not seeds:
        return None, None

    vals = [_recompute_observed_pairs(seed, n_token, j_pct) for seed in seeds]
    if not vals:
        return None, None
    vals = np.asarray(vals, dtype=float)
    sem = float(pd.Series(vals).sem()) if len(vals) > 1 else 0.0
    return float(vals.mean()), sem


def gather_observed_pair_efficiency_data():
    data = {
        'auc': {ARAF_KEY: [], 'knn': []},
        'rmse': {ARAF_KEY: [], 'knn': []},
    }
    baseline_df = load_baseline_cache()
    configs = discover_observed_pair_sweep_configs()

    for (n_token, j_pct), paths_by_variant in configs.items():
        candidates = {
            variant: _load_metrics_from_file(path)
            for variant, path in paths_by_variant.items()
        }
        pair_mean, _ = load_observed_pair_stats(paths_by_variant, n_token, j_pct)
        if pair_mean is None:
            continue

        n_samples = n_token_to_int(n_token)
        for metric in ['auc', 'rmse']:
            _, best_point = select_best_araf_point(metric, candidates)
            if best_point is None:
                continue

            knn_mean, knn_sem = baseline_stats(
                baseline_df,
                metric_col=f'{metric}_knn',
                model_type='beta',
                n_samples=n_samples,
                pre_revision='none',
                j_percentage=j_pct,
            )
            if knn_mean is None:
                continue

            mean_val, sem_val = best_point
            data[metric][ARAF_KEY].append((pair_mean, mean_val, sem_val))
            data[metric]['knn'].append((pair_mean, knn_mean, knn_sem))

    for metric in ['auc', 'rmse']:
        for key in [ARAF_KEY, 'knn']:
            data[metric][key].sort(key=lambda row: row[0])

    return data


def gather_beta_agent_data():
    data = {
        'auc': {ARAF_KEY: []},
        'rmse': {ARAF_KEY: []},
        'baselines': {
            'auc': {b: [] for b in BASELINE_KEYS},
            'rmse': {b: [] for b in BASELINE_KEYS},
        },
    }
    baseline_df = load_baseline_cache()
    for i, size in enumerate(SIZES_BETA):
        x = X_VALS_BETA[i]
        candidate_metrics = {
            variant: load_beta_agent_metrics(variant, size)
            for variant in ARAF_VARIANTS
        }

        for metric in ['auc', 'rmse']:
            _, best_point = select_best_araf_point(metric, candidate_metrics)
            if best_point is not None:
                mean_val, sem_val = best_point
                data[metric][ARAF_KEY].append((x, mean_val, sem_val))

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
        'auc': {ARAF_KEY: []},
        'rmse': {ARAF_KEY: []},
        'baselines': {
            'auc': {b: [] for b in BASELINE_KEYS},
            'rmse': {b: [] for b in BASELINE_KEYS},
        },
    }
    baseline_df = load_baseline_cache()
    for j_pct in J_PCTS:
        x = int(round(j_pct * 100))
        candidate_metrics = {
            variant: load_beta_item_metrics(variant, j_pct)
            for variant in ARAF_VARIANTS
        }

        for metric in ['auc', 'rmse']:
            _, best_point = select_best_araf_point(metric, candidate_metrics)
            if best_point is not None:
                mean_val, sem_val = best_point
                data[metric][ARAF_KEY].append((x, mean_val, sem_val))

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

    for model_key, series in series_by_model.items():
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
            color=ARAF_COLOR,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=ARAF_COLOR,
            label=ARAF_LABEL,
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
    fig.text(left_pair_center, 0.18, '(a) Number of Agents ($N$)', ha='center', va='center', fontsize=FONT_SIZE_GROUP_LABEL)
    fig.text(right_pair_center, 0.18, '(b) Percentage of Items ($J\\%$)', ha='center', va='center', fontsize=FONT_SIZE_GROUP_LABEL)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.075),
        ncol=3,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
    )

    plt.subplots_adjust(bottom=0.32, top=0.88, wspace=0.35)

    out_pdf = os.path.join(FIGURE_DIR, 'sample_size_quad.pdf')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Success! 4-panel beta plot generated: {out_pdf}')


def plot_observed_pair_efficiency():
    print('Generating observed-pair efficiency plot (post-revision beta)...')

    efficiency = gather_observed_pair_efficiency_data()
    if not efficiency['auc'][ARAF_KEY] or not efficiency['auc']['knn']:
        print('Skipping observed-pair efficiency plot: no sweep files found.')
        return

    plt.rcParams.update(bundles.icml2024(usetex=False, family='serif'))
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), constrained_layout=False)

    series_specs = [
        ('auc', 'AUC', (0.48, 0.78)),
        ('rmse', 'RMSE', (0.20, 0.46)),
    ]
    colors = {
        ARAF_KEY: ARAF_COLOR,
        'knn': BASELINE_COLORS['knn'],
    }
    labels = {
        ARAF_KEY: ARAF_LABEL,
        'knn': 'kNN',
    }
    linestyles = {
        ARAF_KEY: '-',
        'knn': '--',
    }

    for ax, (metric_key, title, ylim) in zip(axes, series_specs):
        for series_key in [ARAF_KEY, 'knn']:
            series = efficiency[metric_key][series_key]
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
                color=colors[series_key],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x,
                y,
                color=colors[series_key],
                label=labels[series_key],
                linestyle=linestyles[series_key],
                marker='o',
                markersize=3.0,
                linewidth=1.5,
                alpha=0.9,
                zorder=2,
            )

        if metric_key == 'auc':
            ax.axhline(0.5, color=BASELINE_GRAY, linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_xlabel('Observed Train Pairs', fontsize=FONT_SIZE_TICK)
        ax.set_ylim(*ylim)
        ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
    )

    plt.subplots_adjust(bottom=0.30, top=0.88, wspace=0.30)

    out_pdf = os.path.join(FIGURE_DIR, 'observed_pair_efficiency_beta.pdf')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Success! Observed-pair efficiency plot generated: {out_pdf}')

def main():
    plot_combined_beta_quad()
    plot_observed_pair_efficiency()

if __name__ == "__main__":
    main()
