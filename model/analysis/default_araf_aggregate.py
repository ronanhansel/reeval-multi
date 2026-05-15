from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_config(path: Path) -> dict:
    config_paths = [
        Path(str(path) + ".config.json"),
        Path(str(path) + ".config.expected.json"),
    ]
    for config_path in config_paths:
        if not config_path.exists():
            continue
        try:
            return json.loads(config_path.read_text())
        except Exception:
            continue
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate flat default response-matrix ARAF artifacts.")
    parser.add_argument("--flat-root", default="model/result/default_araf_flat")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    flat_root = Path(args.flat_root)
    araf_dir = flat_root / "araf"
    output_dir = Path(args.output_dir) if args.output_dir else flat_root / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    metric_frames = []
    for csv_path in sorted(araf_dir.glob("*.csv")):
        cfg = read_config(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            manifest_rows.append({
                "artifact_path": str(csv_path),
                "config_path": str(csv_path) + ".config.json",
                "status": "read_failed",
                "error": str(exc),
                **cfg,
            })
            continue
        required = {"seed", "lambda_tau", "rmse_amortized", "auc_amortized", "araf_latent_dim", "araf_dropout"}
        seeds_cfg = [int(x) for x in str(cfg.get("seeds", "")).replace(",", " ").split() if x]
        taus_cfg = [float(x) for x in str(cfg.get("lambda_tau", "")).replace(",", " ").split() if x]
        
        status = "incomplete"
        if required.issubset(df.columns) and not df.empty:
            status = "complete"
            import numpy as np
            for s_val in seeds_cfg:
                for t_val in taus_cfg:
                    match = df[(df["seed"].astype(int) == s_val) & (np.isclose(df["lambda_tau"].astype(float), t_val, atol=1e-7))]
                    if match.empty:
                        status = "incomplete"
                        break
                if status == "incomplete":
                    break

        row = {
            "artifact_path": str(csv_path),
            "config_path": str(csv_path) + ".config.json",
            "status": status,
            "rows": int(len(df)),
            "seeds_completed": int(df["seed"].nunique()) if "seed" in df else 0,
            "taus_completed": int(df["lambda_tau"].nunique()) if "lambda_tau" in df else 0,
            "expected_seeds": len(seeds_cfg),
            "expected_taus": len(taus_cfg),
        }
        row.update(cfg)
        if required.issubset(df.columns) and not df.empty:
            row.update({
                "mean_auc": float(df["auc_amortized"].mean()),
                "mean_rmse": float(df["rmse_amortized"].mean()),
                "min_auc": float(df["auc_amortized"].min()),
                "max_auc": float(df["auc_amortized"].max()),
                "min_rmse": float(df["rmse_amortized"].min()),
                "max_rmse": float(df["rmse_amortized"].max()),
            })
            metrics = df.copy()
            for key, value in cfg.items():
                if key not in metrics.columns:
                    metrics[key] = value
            metrics["artifact_path"] = str(csv_path)
            metrics["config_path"] = str(csv_path) + ".config.json"
            metric_frames.append(metrics)
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "result_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    if metric_frames:
        metrics_long = pd.concat(metric_frames, ignore_index=True, sort=False)
        metrics_long_path = output_dir / "metrics_long.csv"
        metrics_long.to_csv(metrics_long_path, index=False)

        group_cols = [
            "embedding_type",
            "model_type",
            "pre_revision",
            "n_samples",
            "j_percentage",
            "train_retention",
            "user_count",
            "no_tau",
        ]
        available_group_cols = [c for c in group_cols if c in metrics_long.columns]
        best_rows = []
        if available_group_cols:
            summary = (
                metrics_long
                .groupby(available_group_cols + ["araf_latent_dim", "araf_dropout", "lambda_tau", "artifact_path"], dropna=False)
                .agg(mean_auc=("auc_amortized", "mean"), mean_rmse=("rmse_amortized", "mean"), rows=("seed", "count"), seeds=("seed", "nunique"))
                .reset_index()
            )
            for _, grp in summary.groupby(available_group_cols, dropna=False):
                ordered = grp.sort_values(["mean_auc", "mean_rmse", "araf_latent_dim"], ascending=[False, True, True])
                best_rows.append(ordered.iloc[0])
            best = pd.DataFrame(best_rows)
        else:
            best = pd.DataFrame()
        best_path = output_dir / "best_by_config.csv"
        best.to_csv(best_path, index=False)
    else:
        (output_dir / "metrics_long.csv").write_text("")
        (output_dir / "best_by_config.csv").write_text("")

    print(f"Wrote {manifest_path}")
    if metric_frames:
        print(f"Wrote {output_dir / 'metrics_long.csv'}")
        print(f"Wrote {output_dir / 'best_by_config.csv'}")


if __name__ == "__main__":
    main()
