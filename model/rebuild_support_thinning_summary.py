#!/usr/bin/env python3
"""
Rescan support-thinning result folders and rebuild the aggregate beta summary CSV.

This script is intended for recovery after interrupted runs or lock contention:
- scans per-config amortized result CSVs under model/result/support_thinning_study/
- extracts metadata from filenames and retention subfolders
- rebuilds support_thinning_beta_grid.csv
- reports missing / partial combo files relative to the expected thinning ladder
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "result" / "support_thinning_study"
OUTPUT_CSV = RESULT_DIR / "support_thinning_bernoulli_grid.csv"

RETENTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]
ARAF_EMBEDDINGS = ["raw", "pca"]
KNN_EMBEDDINGS = ["raw", "pca"]
K_VALUES = [5, 10, 20, 50]
PRE_REVISION = "max"
J_PERCENTAGE = 1.0
MODEL_TYPE = "bernoulli"
EXPECTED_ROWS_PER_COMBO = 107 * 50

RESULT_RE = re.compile(
    r"amortized_irt_(?P<embedding_type>raw|pca)_(?P<model_type>beta|bernoulli)"
    r"(?:_pre_(?P<pre_revision>[^_]+))?"
    r"_n_(?P<n_token>max|\d+)"
    r"(?:_j(?P<j_percentage>[0-9.]+))?"
    r"(?:_b(?P<baseline_embedding_type>raw|pca)_k(?P<knn_k>\d+))?"
    r"\.csv$"
)


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
        "n_token": meta["n_token"],
        "j_percentage": float(meta["j_percentage"] or "1.0"),
        "baseline_embedding_type": meta["baseline_embedding_type"] or "raw",
        "knn_k": int(meta["knn_k"] or "10"),
        "train_retention": retention,
        "generic_araf": "araf_sweeps" in path.parts and meta["baseline_embedding_type"] is None,
        "araf_variant": "post" if "post_araf_sweeps" in path.parts else "transfer",
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
                base_row = {
                    "seed": int(float(record["seed"])),
                    "lambda_tau": float(record["lambda_tau"]),
                    "n_samples": int(float(record["n_samples"])),
                    "model_type": meta["model_type"],
                    "pre_revision": meta["pre_revision"],
                    "j_percentage": meta["j_percentage"],
                    "embedding_type": meta["embedding_type"],
                    "train_retention": meta["train_retention"],
                    "observed_train_pairs": float(record.get("observed_train_pairs", "")) if record.get("observed_train_pairs") not in ("", None) else "",
                    "auc_araf": float(record["auc_amortized"]),
                    "rmse_araf": float(record["rmse_amortized"]),
                    # Keep kNN metrics decoupled from ARAF files. They are backfilled
                    # from kNN baseline artifacts so expensive ARAF sweeps do not need
                    # reruns when only kNN baselines are refreshed.
                    "auc_knn": "",
                    "rmse_knn": "",
                    "auc_araf_post": "",
                    "rmse_araf_post": "",
                    "source_path": source_path,
                }
                if meta.get("generic_araf", False):
                    for baseline_embedding_type in KNN_EMBEDDINGS:
                        for knn_k in K_VALUES:
                            rows.append({
                                **base_row,
                                "baseline_embedding_type": baseline_embedding_type,
                                "knn_k": knn_k,
                            })
                else:
                    rows.append({
                        **base_row,
                        "baseline_embedding_type": meta["baseline_embedding_type"],
                        "knn_k": meta["knn_k"],
                    })
            except (TypeError, ValueError):
                continue
    return rows


def build_knn_lookup(result_dir: Path):
    """Recover kNN metrics from matching baseline cache files inside each thinning combo folder."""
    lookup = {}
    for baseline_path in result_dir.rglob("baseline_metrics.csv"):
        try:
            with baseline_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                combo_match = re.search(r"knn_(raw|pca)_k(\d+)", str(baseline_path))
                bemb = combo_match.group(1) if combo_match else "raw"
                knn_k = int(combo_match.group(2)) if combo_match else 10
                retention = 1.0
                for part in baseline_path.parts:
                    if part.startswith("retain_"):
                        retention = float(part.split("_", 1)[1])
                        break
                for record in reader:
                    try:
                        seed = int(float(record["seed"])) if record.get("seed") not in ("", None) else None
                        if seed is None:
                            continue
                        pre_revision = str(record.get("pre_revision", "none")).lower()
                        j_percentage = float(record.get("j_percentage", "1.0") or "1.0")

                        lookup[(seed, pre_revision, round(j_percentage, 3), round(retention, 3), bemb, knn_k)] = {
                            "auc_knn": float(record["auc_knn"]),
                            "rmse_knn": float(record["rmse_knn"]),
                        }
                    except (TypeError, ValueError, KeyError):
                        continue
        except Exception:
            continue
    for baseline_path in result_dir.rglob("baseline_knn_*.csv"):
        try:
            with baseline_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for record in reader:
                    try:
                        retention = 1.0
                        for part in baseline_path.parts:
                            if part.startswith("retain_"):
                                retention = float(part.split("_", 1)[1])
                                break
                        combo_match = re.search(r"knn_(raw|pca)_k(\d+)", str(baseline_path))
                        bemb = combo_match.group(1) if combo_match else "raw"
                        knn_k = int(combo_match.group(2)) if combo_match else 10
                        pre_match = re.search(r"_pre_([^_]+)_", baseline_path.name)
                        j_match = re.search(r"_j([0-9.]+)\.csv$", baseline_path.name)
                        pre_revision = pre_match.group(1).lower() if pre_match else "none"
                        j_percentage = float(j_match.group(1)) if j_match else 1.0
                        seed = int(float(record["seed"])) if "seed" in record and record["seed"] not in ("", None) else None
                        if seed is None:
                            continue
                        lookup[(seed, pre_revision, round(j_percentage, 3), round(retention, 3), bemb, knn_k)] = {
                            "auc_knn": float(record["auc_knn"]),
                            "rmse_knn": float(record["rmse_knn"]),
                        }
                    except (TypeError, ValueError, KeyError):
                        continue
        except Exception:
            continue
    return lookup


def build_post_araf_lookup(result_dir: Path):
    """Recover post-trained ARAF metrics from sidecar post_araf_sweeps outputs."""
    lookup = {}
    for path in sorted(result_dir.rglob("amortized_irt_*.csv")):
        if "post_araf_sweeps" not in path.parts:
            continue
        meta = parse_result_path(path)
        if meta is None:
            continue
        if meta["model_type"] != MODEL_TYPE:
            continue
        if meta["pre_revision"] != PRE_REVISION:
            continue
        if round(meta["j_percentage"], 3) != J_PERCENTAGE:
            continue
        if round(meta["train_retention"], 3) not in {round(x, 3) for x in RETENTIONS}:
            continue
        if meta["embedding_type"] not in ARAF_EMBEDDINGS:
            continue

        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                required = {"seed", "lambda_tau", "n_samples", "auc_amortized", "rmse_amortized"}
                if not required.issubset(set(reader.fieldnames)):
                    continue
                for record in reader:
                    try:
                        key = (
                            int(float(record["seed"])),
                            round(float(record["lambda_tau"]), 10),
                            int(float(record["n_samples"])),
                            meta["model_type"],
                            meta["pre_revision"],
                            round(meta["j_percentage"], 3),
                            meta["embedding_type"],
                            round(meta["train_retention"], 3),
                        )
                        lookup[key] = {
                            "auc_araf_post": float(record["auc_amortized"]),
                            "rmse_araf_post": float(record["rmse_amortized"]),
                        }
                    except (TypeError, ValueError):
                        continue
        except Exception:
            continue
    return lookup


def expected_combos():
    for retention in RETENTIONS:
        for emb in ARAF_EMBEDDINGS:
            for bemb in KNN_EMBEDDINGS:
                for knn_k in K_VALUES:
                    yield (round(retention, 3), emb, bemb, knn_k)


def main():
    parser = argparse.ArgumentParser(description="Rebuild support-thinning beta summary CSV from per-config results.")
    parser.add_argument("--result-dir", type=str, default=str(RESULT_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_csv = Path(args.output)

    rows = []
    combo_counts = defaultdict(int)
    file_counts = {}

    for path in sorted(result_dir.rglob("amortized_irt_*.csv")):
        meta = parse_result_path(path)
        if meta is None:
            continue
        if meta.get("araf_variant") == "post":
            continue
        if meta["model_type"] != MODEL_TYPE:
            continue
        if meta["pre_revision"] != PRE_REVISION:
            continue
        if round(meta["j_percentage"], 3) != J_PERCENTAGE:
            continue
        if round(meta["train_retention"], 3) not in {round(x, 3) for x in RETENTIONS}:
            continue
        if meta["embedding_type"] not in ARAF_EMBEDDINGS:
            continue
        if meta["baseline_embedding_type"] not in KNN_EMBEDDINGS:
            continue
        if meta["knn_k"] not in K_VALUES:
            continue

        scanned = scan_result_file(path, meta)
        rows.extend(scanned)
        combo_key = (
            round(meta["train_retention"], 3),
            meta["embedding_type"],
            meta["baseline_embedding_type"],
            meta["knn_k"],
        )
        combo_counts[combo_key] += len(scanned)
        file_counts[str(path.relative_to(result_dir))] = len(scanned)

    knn_lookup = build_knn_lookup(result_dir)
    post_araf_lookup = build_post_araf_lookup(result_dir)
    for row in rows:
        key = (
            row["seed"],
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
        post_key = (
            row["seed"],
            round(float(row["lambda_tau"]), 10),
            row["n_samples"],
            row["model_type"],
            row["pre_revision"],
            round(float(row["j_percentage"]), 3),
            row["embedding_type"],
            round(float(row["train_retention"]), 3),
        )
        post_found = post_araf_lookup.get(post_key)
        if post_found is not None:
            row["auc_araf_post"] = post_found["auc_araf_post"]
            row["rmse_araf_post"] = post_found["rmse_araf_post"]

    deduped = {}
    for row in rows:
        key = (
            row["seed"],
            row["lambda_tau"],
            row["n_samples"],
            row["model_type"],
            row["pre_revision"],
            round(float(row["j_percentage"]), 3),
            row["embedding_type"],
            row["baseline_embedding_type"],
            int(row["knn_k"]),
            round(float(row["train_retention"]), 3),
        )
        deduped[key] = row

    final_rows = list(deduped.values())
    final_rows.sort(key=lambda r: (
        float(r["train_retention"]),
        r["embedding_type"],
        r["baseline_embedding_type"],
        int(r["knn_k"]),
        float(r["lambda_tau"]),
        int(r["seed"]),
    ))

    final_combo_counts = defaultdict(int)
    for row in final_rows:
        combo_key = (
            round(float(row["train_retention"]), 3),
            row["embedding_type"],
            row["baseline_embedding_type"],
            int(row["knn_k"]),
        )
        final_combo_counts[combo_key] += 1

    print(f"Scanned rows: {len(rows)}")
    print(f"Deduped rows: {len(final_rows)}")
    print(f"Expected rows: {len(RETENTIONS) * len(ARAF_EMBEDDINGS) * len(KNN_EMBEDDINGS) * len(K_VALUES) * EXPECTED_ROWS_PER_COMBO}")

    print("\nPer-combo row counts:")
    for combo in expected_combos():
        count = final_combo_counts.get(combo, 0)
        status = "complete" if count >= EXPECTED_ROWS_PER_COMBO else "partial" if count > 0 else "missing"
        print(f"  retention={combo[0]:>4} emb={combo[1]:>3} baseline={combo[2]:>3} k={combo[3]:>2}: {count:>5} rows [{status}]")

    missing_knn = sum(1 for r in final_rows if r["auc_knn"] == "" or r["rmse_knn"] == "")
    if missing_knn:
        print(f"\nWarning: {missing_knn} rows are still missing kNN metrics after rescanning baseline files.")

    if args.dry_run:
        print("\nDry run only; not writing output CSV.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed", "lambda_tau", "n_samples", "model_type", "pre_revision",
        "j_percentage", "embedding_type", "baseline_embedding_type", "knn_k",
        "train_retention", "observed_train_pairs", "auc_knn", "rmse_knn",
        "auc_araf", "rmse_araf", "auc_araf_post", "rmse_araf_post", "source_path",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nWrote rebuilt support-thinning summary to {output_csv}")


if __name__ == "__main__":
    main()
