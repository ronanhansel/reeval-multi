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
import numpy as np
import pandas as pd
from tueplots import bundles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MODEL_DIR)

RESULT_DIR = os.path.join(MODEL_DIR, "result", "support_thinning_study")
FIGURE_DIR = os.path.join(REPO_ROOT, "paper", "figures")
THIN_CSV = os.path.join(RESULT_DIR, "support_thinning_grid.csv")
os.makedirs(FIGURE_DIR, exist_ok=True)

RETENTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]

MODEL_STYLES = {
    "araf": {"label": "ARAF", "color": "steelblue", "linestyle": "-"},
    "knn": {"label": "kNN", "color": "orange", "linestyle": "--"},
}

PANEL_SPECS = [
    ("pre_binary", "auc", "AUC (Binary Pre)", False),
    ("post_binary", "auc", "AUC (Binary Post)", False),
    ("post_beta", "rmse", "RMSE (Beta Post)", True),
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
        if araf_rows.empty or knn_rows.empty:
            continue

        araf_col = f"{metric}_araf"
        knn_col = f"{metric}_knn"
        seed_araf = araf_rows.groupby("seed", as_index=False)[araf_col].mean()[araf_col]
        seed_knn = knn_rows.groupby("seed", as_index=False)[knn_col].mean()[knn_col]

        rows.append(
            {
                "retention": float(retention),
                "retention_pct": float(retention) * 100.0,
                "observed_train_pairs": float(araf_rows["observed_train_pairs"].mean()),
                "araf_mean": float(seed_araf.mean()),
                "araf_sem": float(seed_araf.sem()) if len(seed_araf) > 1 else 0.0,
                "knn_mean": float(seed_knn.mean()),
                "knn_sem": float(seed_knn.sem()) if len(seed_knn) > 1 else 0.0,
                "araf_embedding_type": str(araf_pick["embedding_type"]),
                "araf_lambda_tau": float(araf_pick["lambda_tau"]),
                "knn_embedding_type": str(knn_pick["baseline_embedding_type"]),
                "knn_k": int(knn_pick["knn_k"]),
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("retention_pct")


def plot_panel(ax, panel_df, title, y_label, lower_better=False):
    x = panel_df["retention_pct"].to_numpy(dtype=float)

    for key in ["araf", "knn"]:
        style = MODEL_STYLES[key]
        mean = panel_df[f"{key}_mean"].to_numpy(dtype=float)
        sem = panel_df[f"{key}_sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker="o",
            linewidth=1.6,
            label=style["label"],
        )
        ax.fill_between(x, mean - sem, mean + sem, color=style["color"], alpha=0.18)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(v))}%" for v in x], fontsize=8)
    ax.grid(linestyle=":", alpha=0.8)
    ax.tick_params(labelsize=8)

    if not lower_better:
        ax.axhline(0.5, color="slategray", linestyle="--", linewidth=1.0, alpha=0.7)

    lower_bounds = []
    upper_bounds = []
    for key in ["araf", "knn"]:
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
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), constrained_layout=False)

    for ax, (comparison_slice, metric, title, lower_better) in zip(axes, PANEL_SPECS):
        plot_panel(
            ax,
            panel_dfs[(comparison_slice, metric)],
            title=title,
            y_label="AUC" if metric == "auc" else "RMSE",
            lower_better=lower_better,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel("Percentage of Observed Train Pairs", fontsize=9, y=0.12)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=True, fontsize=8)
    fig.subplots_adjust(bottom=0.34, wspace=0.28)

    out_path = os.path.join(FIGURE_DIR, "support_thinning_triptych.pdf")
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
