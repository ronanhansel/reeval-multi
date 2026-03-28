#!/usr/bin/env python3
"""
Rebuild the support-thinning summary grid.

The rebuilt CSV includes:
- post-revision Bernoulli rows (used for Binary Post panels)
- post-revision Beta rows (kept for completeness)
- legacy pre-revision pre-max rows from the discarded thinning run
  (stored as model_type=beta in the historical CSVs, but treated as Binary Pre)
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from baseline_cache import load_baseline_store


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "result" / "support_thinning_study"
OUTPUT_CSV = RESULT_DIR / "support_thinning_grid.csv"

RETENTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]
RETENTION_SET = {round(x, 3) for x in RETENTIONS}
MODEL_TYPES = ["bernoulli", "beta"]
ARAF_EMBEDDINGS = ["raw", "pca"]
KNN_EMBEDDINGS = ["raw", "pca"]
K_VALUES = [5, 10, 20, 50]
POST_USER_COUNT = 32
J_PERCENTAGE = 1.0

RESULT_RE = re.compile(
    r"amortized_irt_(?P<embedding_type>raw|pca)_(?P<model_type>beta|bernoulli)"
    r"(?:_pre_(?P<pre_revision>[^_]+))?"
    r"(?:_u(?P<user_count>\d+))?"
    r"_n_(?P<n_token>max|\d+)"
    r"(?:_j(?P<j_percentage>[0-9.]+))?"
    r"(?:_b(?P<baseline_embedding_type>raw|pca)_k(?P<knn_k>\d+))?"
    r"\.csv$"
)


def infer_comparison_slice(model_type: str, pre_revision: str, user_count: int) -> str:
    model_type = str(model_type).strip().lower()
    pre_revision = str(pre_revision).strip().lower()
    if pre_revision == "none" and model_type == "bernoulli" and int(user_count) == POST_USER_COUNT:
        return "post_binary"
    if pre_revision == "none" and model_type == "beta" and int(user_count) == POST_USER_COUNT:
        return "post_beta"
    if pre_revision == "max" and model_type == "beta":
        return "pre_binary"
    return ""


def parse_result_path(path: Path):
    match = RESULT_RE.match(path.name)
    if match is None:
        return None

    meta = match.groupdict()
    retention = 1.0
    for part in path.parts:
        if part.startswith("retain_"):
            try:
                retention = float(part.split("_", 1)[1])
            except ValueError:
                retention = 1.0
            break

    return {
        "embedding_type": meta["embedding_type"],
        "model_type": meta["model_type"],
        "pre_revision": (meta["pre_revision"] or "none").lower(),
        "user_count": int(meta["user_count"] or "0"),
        "n_token": meta["n_token"],
        "j_percentage": float(meta["j_percentage"] or "1.0"),
        "baseline_embedding_type": meta["baseline_embedding_type"] or "raw",
        "knn_k": int(meta["knn_k"] or "10"),
        "train_retention": retention,
        "generic_araf": "araf_sweeps" in path.parts,
    }


def scan_result_file(path: Path, meta: dict):
    rows = []
    try:
        source_path = str(path.relative_to(SCRIPT_DIR.parent))
    except Exception:
        source_path = str(path)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        required = {"seed", "lambda_tau", "n_samples", "auc_amortized", "rmse_amortized"}
        if not required.issubset(set(reader.fieldnames)):
            return rows

        for record in reader:
            try:
                user_count = int(float(record.get("user_count", meta["user_count"] or 0) or 0))
                comparison_slice = infer_comparison_slice(
                    model_type=meta["model_type"],
                    pre_revision=meta["pre_revision"],
                    user_count=user_count,
                )
                if not comparison_slice:
                    continue

                base_row = {
                    "seed": int(float(record["seed"])),
                    "lambda_tau": float(record["lambda_tau"]),
                    "n_samples": int(float(record["n_samples"])),
                    "model_type": meta["model_type"],
                    "pre_revision": meta["pre_revision"],
                    "user_count": user_count,
                    "j_percentage": meta["j_percentage"],
                    "embedding_type": meta["embedding_type"],
                    "train_retention": meta["train_retention"],
                    "observed_train_pairs": (
                        float(record.get("observed_train_pairs", ""))
                        if record.get("observed_train_pairs") not in ("", None)
                        else ""
                    ),
                    "auc_araf": float(record["auc_amortized"]),
                    "rmse_araf": float(record["rmse_amortized"]),
                    "auc_knn": "",
                    "rmse_knn": "",
                    "comparison_slice": comparison_slice,
                    "source_path": source_path,
                }
                if meta["generic_araf"]:
                    for baseline_embedding_type in KNN_EMBEDDINGS:
                        for knn_k in K_VALUES:
                            rows.append(
                                {
                                    **base_row,
                                    "baseline_embedding_type": baseline_embedding_type,
                                    "knn_k": knn_k,
                                }
                            )
                else:
                    rows.append(
                        {
                            **base_row,
                            "baseline_embedding_type": meta["baseline_embedding_type"],
                            "knn_k": meta["knn_k"],
                        }
                    )
            except (TypeError, ValueError):
                continue
    return rows


def build_knn_lookup(result_dir: Path):
    lookup = {}
    combo_re = re.compile(r"knn_(?:(?P<model_type>beta|bernoulli)_)?(?P<emb>raw|pca)_k(?P<k>\d+)")

    for baselines_dir in result_dir.rglob("baselines"):
        combo_match = combo_re.search(str(baselines_dir.parent))
        if combo_match is None:
            continue
        baseline_store = load_baseline_store(str(baselines_dir / "baseline_metrics.csv"))
        if baseline_store.empty:
            continue
        retention = 1.0
        for part in baselines_dir.parts:
            if part.startswith("retain_"):
                retention = float(part.split("_", 1)[1])
                break
        for _, record in baseline_store.iterrows():
            try:
                seed = int(float(record["seed"]))
                model_type = combo_match.group("model_type") or str(record.get("model_type", "")).strip().lower()
                pre_revision = str(record.get("pre_revision", "none")).strip().lower()
                j_percentage = float(record.get("j_percentage", "1.0") or "1.0")
                auc_knn = record.get("auc_knn")
                rmse_knn = record.get("rmse_knn")
                if auc_knn == "" or rmse_knn == "":
                    continue
                if auc_knn is None or rmse_knn is None:
                    continue
                if model_type not in MODEL_TYPES:
                    continue
                lookup[
                    (
                        seed,
                        model_type,
                        pre_revision,
                        round(j_percentage, 3),
                        round(retention, 3),
                        combo_match.group("emb"),
                        int(combo_match.group("k")),
                    )
                ] = {
                    "auc_knn": float(auc_knn),
                    "rmse_knn": float(rmse_knn),
                }
            except (TypeError, ValueError, KeyError):
                continue
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Rebuild support-thinning summary CSV from per-config results.")
    parser.add_argument("--result-dir", type=str, default=str(RESULT_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_csv = Path(args.output)

    rows = []
    for path in sorted(result_dir.rglob("amortized_irt_*.csv")):
        if "araf_sweeps" not in path.parts:
            continue
        meta = parse_result_path(path)
        if meta is None:
            continue
        if meta["model_type"] not in MODEL_TYPES:
            continue
        if meta["pre_revision"] not in {"none", "max"}:
            continue
        if meta["n_token"] != "max":
            continue
        if round(meta["j_percentage"], 3) != J_PERCENTAGE:
            continue
        if round(meta["train_retention"], 3) not in RETENTION_SET:
            continue
        if meta["embedding_type"] not in ARAF_EMBEDDINGS:
            continue
        if meta["pre_revision"] == "none" and meta["user_count"] != POST_USER_COUNT:
            continue
        if meta["pre_revision"] == "max" and meta["model_type"] != "beta":
            continue
        rows.extend(scan_result_file(path, meta))

    knn_lookup = build_knn_lookup(result_dir)
    for row in rows:
        key = (
            row["seed"],
            row["model_type"],
            row["pre_revision"],
            round(float(row["j_percentage"]), 3),
            round(float(row["train_retention"]), 3),
            row["baseline_embedding_type"],
            int(row["knn_k"]),
        )
        found = knn_lookup.get(key)
        if found is not None:
            row["auc_knn"] = found["auc_knn"]
            row["rmse_knn"] = found["rmse_knn"]

    deduped = {}
    for row in rows:
        key = (
            row["seed"],
            round(float(row["lambda_tau"]), 10),
            row["n_samples"],
            row["model_type"],
            row["pre_revision"],
            row["user_count"],
            round(float(row["j_percentage"]), 3),
            row["embedding_type"],
            row["baseline_embedding_type"],
            int(row["knn_k"]),
            round(float(row["train_retention"]), 3),
        )
        deduped[key] = row

    final_rows = list(deduped.values())
    final_rows.sort(
        key=lambda r: (
            r["comparison_slice"],
            float(r["train_retention"]),
            r["embedding_type"],
            r["baseline_embedding_type"],
            int(r["knn_k"]),
            float(r["lambda_tau"]),
            int(r["seed"]),
        )
    )

    missing_knn = sum(1 for r in final_rows if r["auc_knn"] == "" or r["rmse_knn"] == "")
    slice_counts = Counter(r["comparison_slice"] for r in final_rows)

    print(f"Scanned rows: {len(rows)}")
    print(f"Deduped rows: {len(final_rows)}")
    print("Rows by slice:")
    for key in sorted(slice_counts):
        print(f"  {key}: {slice_counts[key]}")
    if missing_knn:
        print(f"Warning: {missing_knn} rows are still missing kNN metrics after rescanning baseline files.")

    if args.dry_run:
        print("Dry run only; not writing output CSV.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "lambda_tau",
        "n_samples",
        "model_type",
        "pre_revision",
        "user_count",
        "j_percentage",
        "embedding_type",
        "baseline_embedding_type",
        "knn_k",
        "train_retention",
        "observed_train_pairs",
        "auc_knn",
        "rmse_knn",
        "auc_araf",
        "rmse_araf",
        "comparison_slice",
        "source_path",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Wrote rebuilt support-thinning summary to {output_csv}")


if __name__ == "__main__":
    main()
