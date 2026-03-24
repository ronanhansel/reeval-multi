#!/usr/bin/env python3
"""
Plots for beta outlier-item and probability-robustness study.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result", "outlier_robustness_study")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
CSV_PATH = os.path.join(RESULT_DIR, "outlier_robustness_beta_grid.csv")
CONFIG_PATH = os.path.join(RESULT_DIR, "outlier_robustness_beta_configs.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

NOVELTY_ORDER = ["inlier", "moderate", "outlier"]
NOVELTY_LABELS = ["Inlier", "Moderate", "Outlier"]
METRIC_SPECS = [
    ("auc", "AUC", "higher"),
    ("rmse", "RMSE", "lower"),
    ("brier", "Brier", "lower"),
    ("logloss", "Log Loss", "lower"),
]


def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def load_data():
    if not os.path.exists(CSV_PATH) or not os.path.exists(CONFIG_PATH):
        return None, None

    df = pd.read_csv(CSV_PATH, low_memory=False)
    cfg = pd.read_csv(CONFIG_PATH)
    if df.empty or cfg.empty:
        return None, None

    for col in ["pre_revision", "embedding_type", "model_type", "novelty_bin"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    for col in ["pre_revision", "embedding_type", "selection_metric"]:
        if col in cfg.columns:
            cfg[col] = cfg[col].astype(str).str.strip().str.lower()

    numeric_cols = [
        "j_percentage", "lambda_tau", "auc_knn", "auc_araf", "rmse_knn", "rmse_araf",
        "brier_knn", "brier_araf", "logloss_knn", "logloss_araf", "ece_knn", "ece_araf",
        "p90ae_knn", "p90ae_araf", "p95ae_knn", "p95ae_araf", "mean_dmin", "num_pairs",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["j_percentage", "lambda_tau"]:
        if col in cfg.columns:
            cfg[col] = pd.to_numeric(cfg[col], errors="coerce")

    df = df[df["model_type"] == "beta"]
    return df, cfg


def select_metric_frame(df, cfg, selection_metric):
    chosen = cfg[cfg["selection_metric"] == selection_metric].copy()
    if chosen.empty:
        return pd.DataFrame()
    merged = df.merge(
        chosen,
        on=["pre_revision", "j_percentage", "embedding_type", "lambda_tau"],
        how="inner",
    )
    return merged


def aggregate_by_novelty(df):
    metric_cols = [
        "auc_knn", "auc_araf", "rmse_knn", "rmse_araf",
        "brier_knn", "brier_araf", "logloss_knn", "logloss_araf",
        "ece_knn", "ece_araf", "p90ae_knn", "p90ae_araf",
        "p95ae_knn", "p95ae_araf", "mean_dmin", "num_pairs",
    ]
    seed_level = (
        df.groupby(["seed", "novelty_bin"], as_index=False)[metric_cols]
        .mean()
    )

    rows = []
    for novelty in NOVELTY_ORDER:
        sub = seed_level[seed_level["novelty_bin"] == novelty]
        if sub.empty:
            continue
        row = {"novelty_bin": novelty, "n_seeds": int(sub["seed"].nunique())}
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{col}_sem"] = float(vals.sem()) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_novelty_metrics(auc_df, rmse_df):
    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), constrained_layout=True)
    x = np.arange(len(NOVELTY_ORDER))

    def _plot_panel(ax, metric_key, title, source_df):
        sdf = source_df.set_index("novelty_bin").reindex(NOVELTY_ORDER)
        knn_mean = sdf[f"{metric_key}_knn"].to_numpy(dtype=float)
        knn_sem = sdf[f"{metric_key}_knn_sem"].to_numpy(dtype=float)
        araf_mean = sdf[f"{metric_key}_araf"].to_numpy(dtype=float)
        araf_sem = sdf[f"{metric_key}_araf_sem"].to_numpy(dtype=float)

        ax.plot(x, knn_mean, marker="o", color="orange", linestyle="--", linewidth=1.5, label="kNN")
        ax.fill_between(x, knn_mean - knn_sem, knn_mean + knn_sem, color="orange", alpha=0.18)
        ax.plot(x, araf_mean, marker="o", color="steelblue", linewidth=1.5, label="ARAF")
        ax.fill_between(x, araf_mean - araf_sem, araf_mean + araf_sem, color="steelblue", alpha=0.18)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(NOVELTY_LABELS, fontsize=8)
        ax.grid(linestyle=":", alpha=0.8)
        ax.tick_params(labelsize=8)

    _plot_panel(axes[0, 0], "auc", "AUC by Novelty Bin", auc_df)
    _plot_panel(axes[0, 1], "rmse", "RMSE by Novelty Bin", rmse_df)
    _plot_panel(axes[1, 0], "brier", "Brier by Novelty Bin", rmse_df)
    _plot_panel(axes[1, 1], "logloss", "Log Loss by Novelty Bin", rmse_df)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, "outlier_robustness_beta_metrics.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Outlier robustness metrics figure saved to {out_path}")


def plot_calibration_summary(df):
    summary = aggregate_by_novelty(df).set_index("novelty_bin").reindex(NOVELTY_ORDER)

    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), constrained_layout=True)
    x = np.arange(len(NOVELTY_ORDER))

    metric_triplets = [
        ("ece", "ECE"),
        ("p90ae", "P90AE"),
        ("p95ae", "P95AE"),
    ]
    for ax, (metric, title) in zip(axes, metric_triplets):
        knn_mean = summary[f"{metric}_knn"].to_numpy(dtype=float)
        knn_sem = summary[f"{metric}_knn_sem"].to_numpy(dtype=float)
        araf_mean = summary[f"{metric}_araf"].to_numpy(dtype=float)
        araf_sem = summary[f"{metric}_araf_sem"].to_numpy(dtype=float)
        ax.plot(x, knn_mean, marker="o", color="orange", linestyle="--", linewidth=1.5, label="kNN")
        ax.fill_between(x, knn_mean - knn_sem, knn_mean + knn_sem, color="orange", alpha=0.18)
        ax.plot(x, araf_mean, marker="o", color="steelblue", linewidth=1.5, label="ARAF")
        ax.fill_between(x, araf_mean - araf_sem, araf_mean + araf_sem, color="steelblue", alpha=0.18)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(NOVELTY_LABELS, fontsize=8)
        ax.grid(linestyle=":", alpha=0.8)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=8)

    out_path = os.path.join(FIGURE_DIR, "outlier_robustness_beta_calibration.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Outlier robustness calibration figure saved to {out_path}")


def main():
    df, cfg = load_data()
    if df is None or cfg is None:
        print("Outlier-robustness data missing or incomplete; skipping outlier-robustness plots.")
        return

    auc_frame = select_metric_frame(df, cfg, "auc")
    rmse_frame = select_metric_frame(df, cfg, "rmse")
    if auc_frame.empty or rmse_frame.empty:
        print("Outlier-robustness selection produced no rows; skipping plots.")
        return

    plot_novelty_metrics(aggregate_by_novelty(auc_frame), aggregate_by_novelty(rmse_frame))
    plot_calibration_summary(rmse_frame)


if __name__ == "__main__":
    main()
