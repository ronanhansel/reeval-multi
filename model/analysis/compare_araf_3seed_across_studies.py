from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


SEEDS = {0, 1, 2}
OUT = Path("model/result/default_araf_flat/summaries")


def safe_read(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def norm_pre(value: object) -> str:
    if pd.isna(value):
        return "none"
    text = str(value).strip()
    if text in {"", "false", "False", "none", "None", "nan"}:
        return "none"
    return text


def norm_num(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"max", "all"}:
        return text
    try:
        number = float(text)
    except Exception:
        return text
    return int(number) if number.is_integer() else number


def infer_study(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "default_araf_flat" in parts:
        return "default_flat", "default"
    if "sample_size_study" in parts:
        idx = parts.index("sample_size_study")
        return "sample_size", parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    if "support_thinning_study" in parts:
        idx = parts.index("support_thinning_study")
        return "support_thinning", parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    return "unknown", "unknown"


def add_join_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "pre_revision": "none",
        "j_percentage": 1.0,
        "train_retention": 1.0,
        "n_samples": np.nan,
        "model_type": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    df["pre_join"] = df["pre_revision"].map(norm_pre)
    df["n_join"] = df["n_samples"].map(norm_num)
    df["j_join"] = df["j_percentage"].astype(float).round(8)
    df["ret_join"] = df["train_retention"].astype(float).round(8)
    return df


def normalize_araf_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Legacy support-thinning pre-max ARAF files store pre_revision as "none",
    # but the intended same-seed baselines are the beta pre=max n=1 files.
    support_pre_max = (
        (df["study"] == "support_thinning")
        & (df["model_type"] == "beta")
        & (df["n_join"] == 1)
        & df["artifact_path"].str.contains("beta_pre_max_n_max", na=False)
    )
    df.loc[support_pre_max, "pre_join"] = "max"

    # Default flat beta n=max rows correspond to the concrete n=54 baseline.
    default_beta_post = (
        (df["study"] == "default_flat")
        & (df["model_type"] == "beta")
        & (df["pre_join"] == "none")
        & (df["n_join"] == "max")
    )
    df.loc[default_beta_post, "n_join"] = 54

    # Default flat pre=max rows are compared to the legacy n=1 baseline.
    default_pre_max = (df["study"] == "default_flat") & (df["pre_join"] == "max")
    df.loc[default_pre_max, "n_join"] = 1
    return df


def baseline_dir_for_araf(path: Path, embedding_type: object, pre_join: object) -> Path | None:
    parts = path.parts
    if "default_araf_flat" in parts:
        return Path("model/result/main/baselines")
    if "sample_size_study" in parts:
        idx = parts.index("sample_size_study")
        return Path(*parts[: idx + 2]) / "baselines"
    if "support_thinning_study" in parts:
        idx = parts.index("support_thinning_study")
        retain_dir = Path(*parts[: idx + 2])
        if str(pre_join) == "max":
            return retain_dir / f"knn_{embedding_type}_k10" / "baselines"
        return retain_dir / "shared_baselines"
    return None


def load_baseline_wide(base_dir: Path | None, prefer_raw: bool) -> pd.DataFrame:
    if base_dir is None or not base_dir.exists():
        return pd.DataFrame()

    frames = []
    for path in sorted(base_dir.glob("baseline_*.csv")):
        df = safe_read(path)
        if df is None or df.empty or "seed" not in df.columns:
            continue
        metric_cols = [c for c in df.columns if c.startswith("auc_") or c.startswith("rmse_")]
        if not metric_cols:
            continue
        df = add_join_cols(df)
        df = df[df["seed"].isin(SEEDS)].copy()
        if df.empty:
            continue
        if "baseline_embedding_type" not in df.columns:
            match = re.search(r"baseline_[^_]+_([^_]+)_", path.name)
            df["baseline_embedding_type"] = match.group(1) if match else "raw"

        keep = [
            "seed",
            "model_type",
            "pre_join",
            "n_join",
            "j_join",
            "ret_join",
            "baseline_embedding_type",
        ]
        aux = [
            c
            for c in ["selected_knn_k", "knn_selection_version", "val_auc_knn", "val_rmse_knn"]
            if c in df.columns
        ]
        frames.append(df[keep + aux + metric_cols])

    if not frames:
        return pd.DataFrame()

    all_baselines = pd.concat(frames, ignore_index=True, sort=False)
    selected = all_baselines
    if prefer_raw:
        raw = all_baselines[
            all_baselines["baseline_embedding_type"].isna()
            | (all_baselines["baseline_embedding_type"] == "raw")
        ]
        if not raw.empty:
            selected = raw

    key = ["seed", "model_type", "pre_join", "n_join", "j_join", "ret_join"]
    values = [
        c
        for c in selected.columns
        if c.startswith("auc_")
        or c.startswith("rmse_")
        or c in {"selected_knn_k", "knn_selection_version", "val_auc_knn", "val_rmse_knn"}
    ]
    return selected.groupby(key, dropna=False)[values].first().reset_index()


def load_araf_rows() -> pd.DataFrame:
    frames = []

    manifest = pd.read_csv(OUT / "result_manifest.csv")
    metrics = pd.read_csv(OUT / "metrics_long.csv")
    complete_paths = set(
        manifest[(manifest["status"] == "complete") & (manifest["expected_seeds"] == 3)][
            "artifact_path"
        ]
    )
    flat = metrics[metrics["artifact_path"].isin(complete_paths) & metrics["seed"].isin(SEEDS)].copy()
    for path, group in flat.groupby("artifact_path"):
        study, study_group = infer_study(Path(path))
        group = group.copy()
        group["study"] = study
        group["study_group"] = study_group
        group["artifact_path"] = path
        frames.append(group)

    study_paths = list(Path("model/result/sample_size_study").glob("**/amortized_irt_*.csv"))
    study_paths.extend(Path("model/result/support_thinning_study").glob("**/araf_sweeps/amortized_irt_*.csv"))
    for path in sorted(study_paths):
        df = safe_read(path)
        if df is None or df.empty or "seed" not in df.columns:
            continue
        df = df[df["seed"].isin(SEEDS)].copy()
        if df.empty:
            continue
        study, study_group = infer_study(path)
        df["study"] = study
        df["study_group"] = study_group
        df["artifact_path"] = str(path)
        if study == "support_thinning" and study_group.startswith("retain_"):
            df["train_retention"] = float(study_group.replace("retain_", ""))
        if "araf_latent_dim" not in df.columns:
            df["araf_latent_dim"] = 30
        if "araf_dropout" not in df.columns:
            df["araf_dropout"] = 0.5
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return normalize_araf_keys(add_join_cols(pd.concat(frames, ignore_index=True, sort=False)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    araf = load_araf_rows()
    if araf.empty:
        raise SystemExit("No ARAF rows found")

    merged_parts = []
    coverage_rows = []
    cache: dict[tuple[str, bool], pd.DataFrame] = {}
    for keys, group in araf.groupby(["artifact_path", "embedding_type", "pre_join"], dropna=False):
        artifact_path, embedding_type, pre_join = keys
        base_dir = baseline_dir_for_araf(Path(artifact_path), embedding_type, pre_join)
        prefer_raw = not (group["study"].iloc[0] == "support_thinning" and str(pre_join) == "max")
        cache_key = (str(base_dir), prefer_raw)
        if cache_key not in cache:
            cache[cache_key] = load_baseline_wide(base_dir, prefer_raw=prefer_raw)
        baseline = cache[cache_key]
        join_key = ["seed", "model_type", "pre_join", "n_join", "j_join", "ret_join"]
        merged = group.merge(baseline, on=join_key, how="left") if not baseline.empty else group.copy()
        merged["baseline_dir"] = str(base_dir) if base_dir else ""
        merged_parts.append(merged)
        coverage_rows.append(
            {
                "study": group["study"].iloc[0],
                "study_group": group["study_group"].iloc[0],
                "artifact_path": artifact_path,
                "araf_rows": len(group),
                "araf_seeds": group["seed"].nunique(),
                "baseline_missing": int(merged.get("auc_knn", pd.Series(np.nan, index=merged.index)).isna().sum()),
            }
        )

    rows = pd.concat(merged_parts, ignore_index=True, sort=False)
    rows["auc_diff_vs_knn"] = rows["auc_amortized"] - rows.get("auc_knn")
    rows["rmse_diff_vs_knn"] = rows["rmse_amortized"] - rows.get("rmse_knn")

    group_cols = [
        "study",
        "study_group",
        "embedding_type",
        "model_type",
        "pre_join",
        "n_join",
        "j_join",
        "ret_join",
        "araf_latent_dim",
        "araf_dropout",
        "lambda_tau",
        "artifact_path",
    ]
    summary = (
        rows.groupby(group_cols, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_auc=("auc_amortized", "mean"),
            mean_rmse=("rmse_amortized", "mean"),
            mean_knn_auc=("auc_knn", "mean"),
            mean_knn_rmse=("rmse_knn", "mean"),
            mean_auc_diff_vs_knn=("auc_diff_vs_knn", "mean"),
            mean_rmse_diff_vs_knn=("rmse_diff_vs_knn", "mean"),
            auc_wins_vs_knn=("auc_diff_vs_knn", lambda s: int((s > 0).sum())),
            rmse_wins_vs_knn=("rmse_diff_vs_knn", lambda s: int((s < 0).sum())),
            missing_baseline_rows=("auc_knn", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    best_rows = []
    best_group_cols = [
        "study",
        "study_group",
        "embedding_type",
        "model_type",
        "pre_join",
        "n_join",
        "j_join",
        "ret_join",
    ]
    for _, group in summary.groupby(best_group_cols, dropna=False):
        best_rows.append(
            group.sort_values(
                ["mean_auc", "mean_rmse", "araf_latent_dim"],
                ascending=[False, True, True],
            ).iloc[0]
        )
    best = pd.DataFrame(best_rows)
    coverage = pd.DataFrame(coverage_rows)

    rows.to_csv(OUT / "araf_vs_baselines_3seed_all_studies_rows.csv", index=False)
    summary.to_csv(OUT / "araf_vs_baselines_3seed_all_studies_summary.csv", index=False)
    best.to_csv(OUT / "araf_vs_baselines_3seed_all_studies_best.csv", index=False)
    coverage.to_csv(OUT / "araf_vs_baselines_3seed_all_studies_coverage.csv", index=False)

    print(f"Wrote {OUT / 'araf_vs_baselines_3seed_all_studies_rows.csv'} ({len(rows)} rows)")
    print(f"Wrote {OUT / 'araf_vs_baselines_3seed_all_studies_summary.csv'} ({len(summary)} rows)")
    print(f"Wrote {OUT / 'araf_vs_baselines_3seed_all_studies_best.csv'} ({len(best)} rows)")
    print(f"Wrote {OUT / 'araf_vs_baselines_3seed_all_studies_coverage.csv'} ({len(coverage)} rows)")
    print()
    print(
        coverage.groupby("study")
        .agg(
            artifacts=("artifact_path", "nunique"),
            araf_rows=("araf_rows", "sum"),
            missing_baseline_rows=("baseline_missing", "sum"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
