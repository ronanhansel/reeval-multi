#!/usr/bin/env python3
"""
Support-thinning plots.

Generates a 3-panel figure with:
- Binary pre AUC
- Binary post AUC
- Beta post RMSE
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tueplots import bundles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result", "support_thinning_study")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
THIN_CSV = os.path.join(RESULT_DIR, "support_thinning_grid.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

RETENTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]
PREFERRED_BASELINE_EMBEDDING = "raw"
MODEL_ORDER = ["rasch", "mirt", "knn", "araf"]

MODEL_STYLES = {
    "rasch": {"label": "Rasch", "color": "navajowhite", "linestyle": "--", "alpha": 0.65},
    "mirt": {"label": "MIRT", "color": "tan", "linestyle": "--", "alpha": 0.6},
    "knn": {"label": "kNN", "color": "orange", "linestyle": "--", "alpha": 0.9},
    "araf": {"label": "ARAF", "color": "steelblue", "linestyle": "-", "alpha": 0.85},
}

PANEL_SPECS = [
    ("pre_binary", "auc", "(a) AUC (Pre Revision)", False),
    ("post_binary", "auc", "(b) AUC (Post Revision)", False),
    ("post_beta", "rmse", "(c) RMSE (Post Revision)", True),
]


def get_bundle():
    return bundles.icml2024(usetex=False, family="serif")


def _infer_slice_from_row(row):
    pre_revision = str(row.get("pre_revision", "")).strip().lower()
    model_type = str(row.get("model_type", "")).strip().lower()
    try:
        user_count = int(float(row.get("user_count", 0) or 0))
    except Exception:
        user_count = 0
    if pre_revision == "none" and model_type == "bernoulli" and user_count == 32:
        return "post_binary"
    if pre_revision == "max" and model_type == "beta":
        return "pre_binary"
    if pre_revision == "none" and model_type == "beta" and user_count == 32:
        return "post_beta"
    return ""


def load_data():
    if not os.path.exists(THIN_CSV):
        return None

    df = pd.read_csv(THIN_CSV, low_memory=False)
    if df.empty:
        return None

    numeric_cols = [
        "seed",
        "lambda_tau",
        "n_samples",
        "user_count",
        "j_percentage",
        "train_retention",
        "knn_k",
        "observed_train_pairs",
        "auc_knn",
        "rmse_knn",
        "auc_rasch",
        "rmse_rasch",
        "auc_mirt",
        "rmse_mirt",
        "auc_araf",
        "rmse_araf",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["model_type", "pre_revision", "embedding_type", "baseline_embedding_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "comparison_slice" in df.columns:
        df["comparison_slice"] = df["comparison_slice"].astype(str).str.strip().str.lower()
    else:
        df["comparison_slice"] = df.apply(_infer_slice_from_row, axis=1)

    df = df[
        np.isclose(df["j_percentage"], 1.0, atol=1e-9)
        & df["comparison_slice"].isin({"post_binary", "pre_binary", "post_beta"})
    ].copy()
    if df.empty:
        return None

    return df


def _select_best_araf(sub, metric):
    grouped = sub.groupby(["embedding_type", "lambda_tau"], as_index=False).agg(
        auc_araf=("auc_araf", "mean"),
        rmse_araf=("rmse_araf", "mean"),
    )
    if grouped.empty:
        return sub.iloc[0:0], None

    if metric == "auc":
        pick = grouped.sort_values(["auc_araf", "rmse_araf"], ascending=[False, True]).iloc[0]
    else:
        pick = grouped.sort_values(["rmse_araf", "auc_araf"], ascending=[True, False]).iloc[0]

    rows = sub[
        (sub["embedding_type"] == pick["embedding_type"])
        & (np.isclose(sub["lambda_tau"], float(pick["lambda_tau"]), atol=1e-8))
    ]
    return rows, pick


def _select_best_knn(sub, metric):
    grouped = sub.groupby(["baseline_embedding_type", "knn_k"], as_index=False).agg(
        auc_knn=("auc_knn", "mean"),
        rmse_knn=("rmse_knn", "mean"),
    )
    grouped = grouped.dropna(subset=["auc_knn", "rmse_knn"], how="all")
    if grouped.empty:
        return sub.iloc[0:0], None

    if metric == "auc":
        pick = grouped.sort_values(["auc_knn", "rmse_knn"], ascending=[False, True]).iloc[0]
    else:
        pick = grouped.sort_values(["rmse_knn", "auc_knn"], ascending=[True, False]).iloc[0]

    rows = sub[
        (sub["baseline_embedding_type"] == pick["baseline_embedding_type"])
        & (np.isclose(sub["knn_k"], float(pick["knn_k"]), atol=1e-8))
    ]
    return rows, pick


def _select_preferred_baseline_rows(sub):
    if "baseline_embedding_type" not in sub.columns:
        return sub

    available = set(sub["baseline_embedding_type"].dropna().astype(str))
    if PREFERRED_BASELINE_EMBEDDING in available:
        return sub[sub["baseline_embedding_type"] == PREFERRED_BASELINE_EMBEDDING]
    if "raw" in available:
        return sub[sub["baseline_embedding_type"] == "raw"]
    if "pca" in available:
        return sub[sub["baseline_embedding_type"] == "pca"]
    return sub


def _seed_summary(rows, metric_col):
    if rows.empty or metric_col not in rows.columns:
        return None
    vals = rows.groupby("seed", as_index=False)[metric_col].mean()[metric_col]
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean()), float(vals.sem()) if len(vals) > 1 else 0.0


def build_panel_df(df, comparison_slice, metric):
    rows = []
    for retention in RETENTIONS:
        sub = df[
            (df["comparison_slice"] == comparison_slice)
            & (np.isclose(df["train_retention"], float(retention), atol=1e-9))
        ].copy()
        if sub.empty:
            continue

        araf_rows, araf_pick = _select_best_araf(sub, metric)
        knn_rows, knn_pick = _select_best_knn(sub, metric)
        if araf_rows.empty:
            continue

        baseline_rows = _select_preferred_baseline_rows(sub)
        araf_stats = _seed_summary(araf_rows, f"{metric}_araf")
        knn_stats = _seed_summary(knn_rows, f"{metric}_knn")
        rasch_stats = _seed_summary(baseline_rows, f"{metric}_rasch")
        mirt_stats = _seed_summary(baseline_rows, f"{metric}_mirt")

        if araf_stats is None:
            continue

        row = {
            "retention": float(retention),
            "retention_pct": float(retention) * 100.0,
            "observed_train_pairs": float(araf_rows["observed_train_pairs"].mean()),
            "araf_mean": araf_stats[0],
            "araf_sem": araf_stats[1],
            "araf_embedding_type": str(araf_pick["embedding_type"]),
            "araf_lambda_tau": float(araf_pick["lambda_tau"]),
        }
        if knn_stats is not None and knn_pick is not None:
            row["knn_mean"] = knn_stats[0]
            row["knn_sem"] = knn_stats[1]
            row["knn_embedding_type"] = str(knn_pick["baseline_embedding_type"])
            row["knn_k"] = int(knn_pick["knn_k"])
        else:
            row["knn_mean"] = np.nan
            row["knn_sem"] = np.nan
            row["knn_embedding_type"] = ""
            row["knn_k"] = np.nan

        for key, stats in [("rasch", rasch_stats), ("mirt", mirt_stats)]:
            if stats is not None:
                row[f"{key}_mean"] = stats[0]
                row[f"{key}_sem"] = stats[1]
            else:
                row[f"{key}_mean"] = np.nan
                row[f"{key}_sem"] = np.nan

        rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("retention_pct")


def plot_panel(ax, panel_df, title, y_label, lower_better=False):
    x = panel_df["retention_pct"].to_numpy(dtype=float)

    for key in MODEL_ORDER:
        style = MODEL_STYLES[key]
        mean = panel_df[f"{key}_mean"].to_numpy(dtype=float)
        sem = panel_df[f"{key}_sem"].to_numpy(dtype=float)
        valid = np.isfinite(mean) & np.isfinite(sem)
        if not np.any(valid):
            continue
        ax.plot(
            x[valid],
            mean[valid],
            color=style["color"],
            linestyle=style["linestyle"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=style["label"],
            alpha=style["alpha"],
        )
        ax.fill_between(
            x[valid],
            mean[valid] - sem[valid],
            mean[valid] + sem[valid],
            color=style["color"],
            alpha=0.18 if key == "araf" else 0.10,
        )

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x)
    x_labels = ["" if np.isclose(v, 5.0, atol=1e-9) else f"{int(round(v))}%" for v in x]
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(linestyle=":", alpha=0.8)
    ax.tick_params(labelsize=8)

    if not lower_better:
        ax.axhline(0.5, color="slategray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            0.75,
            0.505,
            "Naive Baseline",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="slategray",
        )

    lower_bounds = []
    upper_bounds = []
    for key in MODEL_ORDER:
        mean = panel_df[f"{key}_mean"].to_numpy(dtype=float)
        sem = panel_df[f"{key}_sem"].to_numpy(dtype=float)
        valid = np.isfinite(mean) & np.isfinite(sem)
        if np.any(valid):
            lower_bounds.append(np.min(mean[valid] - sem[valid]))
            upper_bounds.append(np.max(mean[valid] + sem[valid]))
    if lower_bounds and upper_bounds:
        ymin = min(lower_bounds)
        ymax = max(upper_bounds)
        pad = max((ymax - ymin) * 0.08, 1e-3)
        ax.set_ylim(ymin - pad, ymax + pad)


def plot(df):
    panel_dfs = {}
    for comparison_slice, metric, _, _ in PANEL_SPECS:
        panel_dfs[(comparison_slice, metric)] = build_panel_df(df, comparison_slice, metric)

    if any(panel_dfs[key] is None for key in panel_dfs):
        missing = [
            f"{comparison_slice}:{metric}"
            for (comparison_slice, metric), value in panel_dfs.items()
            if value is None
        ]
        raise RuntimeError(f"Missing support-thinning panels: {', '.join(missing)}")

    plt.rcParams.update(get_bundle())
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.2), constrained_layout=False)

    for ax, (comparison_slice, metric, title, lower_better) in zip(axes, PANEL_SPECS):
        plot_panel(
            ax,
            panel_dfs[(comparison_slice, metric)],
            title=title,
            # y_label="AUC" if metric == "auc" else "RMSE",
            y_label=None,
            lower_better=lower_better,
        )
        if metric == 'auc':
            ax.set_yticks([0.5, 0.6, 0.7, 0.8])
            ax.set_ylim(0.48, 0.78)
        else:
            ax.set_yticks([0.24, 0.26, 0.28])
            ax.set_ylim(0.235, 0.285)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Build legend handles explicitly so dashed styles remain visible in the legend.
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_STYLES[key]["color"],
            linestyle=MODEL_STYLES[key]["linestyle"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            alpha=MODEL_STYLES[key]["alpha"],
            label=MODEL_STYLES[key]["label"],
            dash_capstyle="butt",
        )
        for key in MODEL_ORDER
    ]
    labels = [MODEL_STYLES[key]["label"] for key in MODEL_ORDER]
    fig.supxlabel("Percentage of Observed Train Pairs", fontsize=10, y=0.08)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=True,
        fontsize=8,
        handlelength=2.6,
    )
    fig.subplots_adjust(bottom=0.26, wspace=0.28)

    out_path = os.path.join(FIGURE_DIR, "support_thinning.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Support-thinning figure saved to {out_path}")


def main():
    df = load_data()
    if df is None:
        raise FileNotFoundError(f"No support-thinning summary found at {THIN_CSV}")
    plot(df)


if __name__ == "__main__":
    main()
