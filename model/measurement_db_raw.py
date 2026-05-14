#!/usr/bin/env python3
"""RAW-only measurement-db corpus, embedding, and summary helpers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RESULT_ROOT = Path("model/result/measurement_db_raw")
DEFAULT_HF_REPO = "aims-foundations/measurement-db"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_MAX_CHARS = 20000
DEFAULT_BATCH_SIZE = 8
MATRIX_VERSION = "mdb-raw-matrix-v1"
EMBEDDING_VERSION = "mdb-raw-embedding-v1"

REGISTRY_FILES = {"benchmarks.parquet", "subjects.parquet", "items.parquet"}
REQUIRED_RESULT_COLS = {
    "araf": ["rmse_amortized", "auc_amortized"],
    "knn": ["rmse_knn", "auc_knn"],
}


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("qwen/qwen3-embedding-8b", "qwen3-8b")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "na"


def ensure_result_dirs(result_root: Path) -> None:
    for subdir in ["data_cache", "embeddings", "baselines", "araf", "summaries", "logs"]:
        (result_root / subdir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def config_matches(path: Path, payload: dict[str, Any]) -> bool:
    return read_json(path) == payload


def parse_dataset_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    return sorted(set(names)) if names else None


def dataset_selector_slug(dataset_names: list[str] | None, all_ready: bool) -> str:
    if all_ready or not dataset_names:
        return "all-ready"
    if len(dataset_names) <= 6:
        return "ds-" + "-".join(slugify(name) for name in dataset_names)
    return f"ds-{len(dataset_names)}-explicit"


def corpus_config(args: argparse.Namespace) -> dict[str, Any]:
    dataset_names = parse_dataset_names(args.datasets)
    all_ready = bool(args.all_ready or dataset_names is None)
    return {
        "kind": "measurement_db_raw_corpus",
        "source": args.source,
        "hf_repo": args.hf_repo,
        "source_dir": str(Path(args.source_dir).resolve()) if args.source_dir else "",
        "dataset_names": dataset_names or [],
        "all_ready": all_ready,
        "min_subjects": int(args.min_subjects),
        "min_items": int(args.min_items),
        "response_policy": args.response_policy,
        "matrix_version": MATRIX_VERSION,
    }


def corpus_slug(config: dict[str, Any]) -> str:
    selector = "all-ready" if config["all_ready"] else dataset_selector_slug(config["dataset_names"], False)
    return "_".join(
        [
            f"src-{slugify(config['source'])}",
            selector,
            f"mins{config['min_subjects']}",
            f"mini{config['min_items']}",
            slugify(config["response_policy"]),
            slugify(config["matrix_version"]),
        ]
    )


def embedding_config(corpus_slug_value: str, corpus_path: Path, item_content_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kind": "measurement_db_raw_embeddings",
        "corpus_slug": corpus_slug_value,
        "corpus_path": str(corpus_path),
        "item_content_path": str(item_content_path),
        "embedding_model": args.embedding_model,
        "max_chars": int(args.max_chars),
        "batch_size": int(args.batch_size),
        "chunk_size": int(args.chunk_size),
        "embedding_version": EMBEDDING_VERSION,
    }


def embedding_slug(config: dict[str, Any]) -> str:
    return "_".join(
        [
            f"rawemb-{config['corpus_slug']}",
            f"model-{slugify(config['embedding_model'])}",
            f"maxchars{config['max_chars']}",
            slugify(config["embedding_version"]),
        ]
    )


def list_hf_dataset_names(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset")
    names = []
    for filename in files:
        if not filename.endswith(".parquet"):
            continue
        if filename in REGISTRY_FILES or filename.endswith("_traces.parquet"):
            continue
        names.append(filename.removesuffix(".parquet"))
    return sorted(set(names))


def load_hf_frames(repo_id: str, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    response_path = hf_hub_download(repo_id, f"{dataset_name}.parquet", repo_type="dataset")
    items_path = hf_hub_download(repo_id, "items.parquet", repo_type="dataset")
    subjects_path = hf_hub_download(repo_id, "subjects.parquet", repo_type="dataset")
    benchmarks_path = hf_hub_download(repo_id, "benchmarks.parquet", repo_type="dataset")

    responses = pd.read_parquet(response_path)
    items = pd.read_parquet(items_path)
    subjects = pd.read_parquet(subjects_path)
    benchmarks = pd.read_parquet(benchmarks_path)
    info_rows = benchmarks[benchmarks["benchmark_id"].astype(str) == str(dataset_name)]
    info = info_rows.iloc[0].to_dict() if not info_rows.empty else {}
    return responses, items, subjects, info


def list_source_dataset_names(source_dir: Path) -> list[str]:
    names = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_") or child.name.startswith("."):
            continue
        if (child / "responses.parquet").exists():
            names.append(child.name)
    return names


def load_source_frames(source_dir: Path, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    response_path = source_dir / dataset_name / "responses.parquet"
    responses = pd.read_parquet(response_path)
    items_path = source_dir / "_registry" / "items.parquet"
    subjects_path = source_dir / "_registry" / "subjects.parquet"
    benchmarks_path = source_dir / "_registry" / "benchmarks.parquet"
    if not items_path.exists():
        items_path = source_dir / "items.parquet"
    if not subjects_path.exists():
        subjects_path = source_dir / "subjects.parquet"
    if not benchmarks_path.exists():
        benchmarks_path = source_dir / "benchmarks.parquet"
    items = pd.read_parquet(items_path) if items_path.exists() else pd.DataFrame()
    subjects = pd.read_parquet(subjects_path) if subjects_path.exists() else pd.DataFrame()
    benchmarks = pd.read_parquet(benchmarks_path) if benchmarks_path.exists() else pd.DataFrame()
    info = {}
    if not benchmarks.empty and "benchmark_id" in benchmarks.columns:
        info_rows = benchmarks[benchmarks["benchmark_id"].astype(str) == str(dataset_name)]
        info = info_rows.iloc[0].to_dict() if not info_rows.empty else {}
    return responses, items, subjects, info


def response_policy_allows(values: pd.Series, response_policy: str) -> tuple[bool, str]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return False, "no_finite_responses"
    min_val = float(finite.min())
    max_val = float(finite.max())
    unique = set(float(v) for v in finite.drop_duplicates().head(32).tolist())
    if response_policy == "binary":
        ok = unique.issubset({0.0, 1.0}) and min_val >= 0.0 and max_val <= 1.0
        return ok, "" if ok else f"non_binary_range_{min_val:g}_{max_val:g}"
    if response_policy == "bounded01":
        ok = min_val >= 0.0 and max_val <= 1.0
        return ok, "" if ok else f"outside_0_1_range_{min_val:g}_{max_val:g}"
    return True, ""


def metadata_list_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, np.ndarray):
            return "|".join(map(str, value.tolist()))
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        return "|".join(map(str, value))
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def build_one_matrix(
    dataset_name: str,
    responses: pd.DataFrame,
    items: pd.DataFrame,
    info: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, Any]]:
    required = {"subject_id", "item_id", "response"}
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "included": False,
        "skip_reason": "",
        "subjects": 0,
        "items": 0,
        "responses": 0,
        "density": np.nan,
        "response_type": info.get("response_type", ""),
        "response_min": np.nan,
        "response_max": np.nan,
        "categorical": info.get("categorical", ""),
        "modality": metadata_list_text(info.get("modality")),
        "domain": metadata_list_text(info.get("domain")),
    }
    if not required.issubset(responses.columns):
        row["skip_reason"] = "missing_required_columns"
        return None, pd.DataFrame(), row

    responses = responses[["subject_id", "item_id", "response"]].copy()
    responses["subject_id"] = responses["subject_id"].astype(str)
    responses["item_id"] = responses["item_id"].astype(str)
    responses["response"] = pd.to_numeric(responses["response"], errors="coerce")
    responses = responses.dropna(subset=["subject_id", "item_id", "response"])
    if responses.empty:
        row["skip_reason"] = "empty_responses"
        return None, pd.DataFrame(), row

    allowed, reason = response_policy_allows(responses["response"], config["response_policy"])
    if not allowed:
        row["skip_reason"] = reason
        return None, pd.DataFrame(), row

    if responses.duplicated(subset=["subject_id", "item_id"]).any():
        responses = responses.groupby(["subject_id", "item_id"], as_index=False)["response"].mean()

    n_subjects = int(responses["subject_id"].nunique())
    n_items = int(responses["item_id"].nunique())
    n_rows = int(len(responses))
    row.update(
        {
            "subjects": n_subjects,
            "items": n_items,
            "responses": n_rows,
            "density": float(n_rows / max(n_subjects * n_items, 1)),
            "response_min": float(responses["response"].min()),
            "response_max": float(responses["response"].max()),
        }
    )
    if n_subjects < int(config["min_subjects"]):
        row["skip_reason"] = f"subjects_below_min_{n_subjects}"
        return None, pd.DataFrame(), row
    if n_items < int(config["min_items"]):
        row["skip_reason"] = f"items_below_min_{n_items}"
        return None, pd.DataFrame(), row

    col_map = {item_id: f"{dataset_name}.{item_id}" for item_id in responses["item_id"].unique()}
    responses["task_id"] = responses["item_id"].map(col_map)
    matrix = responses.pivot(index="subject_id", columns="task_id", values="response")
    matrix = matrix.sort_index().reindex(sorted(matrix.columns), axis=1)

    contents = pd.DataFrame({"benchmark.task_id": sorted(col_map.values())})
    contents["dataset"] = dataset_name
    reverse_map = {v: k for k, v in col_map.items()}
    contents["item_id"] = contents["benchmark.task_id"].map(reverse_map)
    if not items.empty and {"item_id", "content"}.issubset(items.columns):
        item_subset = items.copy()
        if "benchmark_id" in item_subset.columns:
            item_subset = item_subset[item_subset["benchmark_id"].astype(str) == dataset_name]
        item_subset["item_id"] = item_subset["item_id"].astype(str)
        item_subset = item_subset.drop_duplicates(subset=["item_id"])
        contents = contents.merge(item_subset[["item_id", "content"]], on="item_id", how="left")
    if "content" not in contents.columns:
        contents["content"] = ""
    contents["text_input"] = contents["content"].fillna("").astype(str)
    fallback = contents["benchmark.task_id"].astype(str)
    contents.loc[contents["text_input"].str.len() == 0, "text_input"] = fallback

    row["included"] = True
    return matrix, contents, row


def build_corpus(args: argparse.Namespace, result_root: Path) -> dict[str, Path | str]:
    config = corpus_config(args)
    slug = corpus_slug(config)
    corpus_path = result_root / "data_cache" / f"corpus_{slug}.parquet"
    item_content_path = result_root / "data_cache" / f"item_contents_{slug}.csv"
    inventory_path = result_root / "summaries" / f"dataset_inventory_{slug}.csv"
    config_path = corpus_path.with_suffix(corpus_path.suffix + ".config.json")

    if (
        corpus_path.exists()
        and item_content_path.exists()
        and inventory_path.exists()
        and config_matches(config_path, config)
        and not args.force
    ):
        return {
            "corpus_slug": slug,
            "corpus_path": corpus_path,
            "item_content_path": item_content_path,
            "dataset_inventory_path": inventory_path,
            "corpus_config_path": config_path,
            "corpus_status": "skipped_existing",
        }

    if args.source == "hf":
        available = list_hf_dataset_names(args.hf_repo)
        loader = lambda name: load_hf_frames(args.hf_repo, name)
    else:
        source_dir = Path(args.source_dir)
        available = list_source_dataset_names(source_dir)
        loader = lambda name: load_source_frames(source_dir, name)

    requested = parse_dataset_names(args.datasets)
    dataset_names = available if config["all_ready"] or requested is None else [name for name in requested if name in available]
    missing_requested = sorted(set(requested or []) - set(dataset_names) - (set(available) if requested else set()))

    matrices = []
    content_frames = []
    inventory_rows = []
    for missing in missing_requested:
        inventory_rows.append({"dataset": missing, "included": False, "skip_reason": "requested_missing"})

    for dataset_name in sorted(dataset_names):
        try:
            responses, items, _subjects, info = loader(dataset_name)
            matrix, contents, row = build_one_matrix(dataset_name, responses, items, info, config)
        except Exception as exc:
            matrix, contents, row = None, pd.DataFrame(), {
                "dataset": dataset_name,
                "included": False,
                "skip_reason": f"load_error_{type(exc).__name__}",
            }
        inventory_rows.append(row)
        if matrix is not None:
            matrices.append(matrix)
            content_frames.append(contents)

    if not matrices:
        raise RuntimeError(f"No datasets included for corpus {slug}. See inventory config: {config}")

    combined = pd.concat(matrices, axis=1, join="outer").sort_index()
    combined.to_parquet(corpus_path, index=True)
    item_contents = pd.concat(content_frames, ignore_index=True).drop_duplicates(subset=["benchmark.task_id"])
    item_contents = item_contents.sort_values("benchmark.task_id")
    item_contents.to_csv(item_content_path, index=False)
    inventory = pd.DataFrame(inventory_rows).sort_values("dataset")
    inventory.to_csv(inventory_path, index=False)

    metadata = dict(config)
    metadata.update(
        {
            "corpus_slug": slug,
            "corpus_path": str(corpus_path),
            "item_content_path": str(item_content_path),
            "dataset_inventory_path": str(inventory_path),
            "included_dataset_count": int(inventory["included"].fillna(False).astype(bool).sum()),
            "subject_count": int(combined.shape[0]),
            "item_count": int(combined.shape[1]),
            "observed_response_count": int(combined.notna().sum().sum()),
            "density": float(combined.notna().sum().sum() / max(combined.shape[0] * combined.shape[1], 1)),
            "dataset_names": sorted(inventory.loc[inventory["included"].fillna(False).astype(bool), "dataset"].astype(str).tolist()),
        }
    )
    write_json(config_path, metadata)
    return {
        "corpus_slug": slug,
        "corpus_path": corpus_path,
        "item_content_path": item_content_path,
        "dataset_inventory_path": inventory_path,
        "corpus_config_path": config_path,
        "corpus_status": "built",
    }


def generate_embeddings(args: argparse.Namespace, result_root: Path, corpus_info: dict[str, Path | str]) -> dict[str, Path | str]:
    config = embedding_config(
        str(corpus_info["corpus_slug"]),
        Path(corpus_info["corpus_path"]),
        Path(corpus_info["item_content_path"]),
        args,
    )
    slug = embedding_slug(config)
    output_path = result_root / "embeddings" / f"{slug}.pkl"
    config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    if output_path.exists() and config_matches(config_path, config) and not args.force:
        return {
            "embedding_slug": slug,
            "embedding_path": output_path,
            "embedding_config_path": config_path,
            "embedding_status": "skipped_existing",
        }

    from sentence_transformers import SentenceTransformer
    import torch

    contents = pd.read_csv(config["item_content_path"])
    if "benchmark.task_id" not in contents.columns or "text_input" not in contents.columns:
        raise ValueError(f"Item content file missing required columns: {config['item_content_path']}")
    texts = contents["text_input"].fillna("").astype(str).str.slice(0, int(args.max_chars)).tolist()
    model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_size = max(int(args.chunk_size), int(args.batch_size), 1)
    part_dir = output_path.with_suffix(output_path.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []
    for start in range(0, len(contents), chunk_size):
        end = min(start + chunk_size, len(contents))
        part_path = part_dir / f"part_{start:09d}_{end:09d}.pkl"
        part_paths.append(part_path)
        if part_path.exists() and not args.force:
            print(f"[embedding] skip existing part {start}:{end}", file=os.sys.stderr, flush=True)
            continue
        print(f"[embedding] encode part {start}:{end} of {len(contents)}", file=os.sys.stderr, flush=True)
        embeddings = model.encode(
            texts[start:end],
            batch_size=int(args.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            device=device_name,
            normalize_embeddings=False,
        )
        part_df = pd.DataFrame(
            {
                "benchmark.task_id": contents["benchmark.task_id"].iloc[start:end].astype(str).tolist(),
                "text_input": contents["text_input"].iloc[start:end].fillna("").astype(str).tolist(),
                "embedding": list(np.asarray(embeddings, dtype=np.float32)),
            }
        )
        part_df.to_pickle(part_path)

    out_df = pd.concat([pd.read_pickle(path) for path in part_paths], ignore_index=True)
    out_df.to_pickle(output_path)
    first_emb = np.asarray(out_df.iloc[0]["embedding"]) if not out_df.empty else np.asarray([])
    metadata = dict(config)
    metadata.update(
        {
            "embedding_slug": slug,
            "embedding_path": str(output_path),
            "embedding_count": int(len(out_df)),
            "embedding_dim": int(first_emb.shape[0]) if first_emb.ndim == 1 else 0,
            "device": device_name,
            "part_dir": str(part_dir),
            "part_count": int(len(part_paths)),
        }
    )
    write_json(config_path, metadata)
    return {
        "embedding_slug": slug,
        "embedding_path": output_path,
        "embedding_config_path": config_path,
        "embedding_status": "built",
    }


def shell_quote_value(value: Any) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def print_env(payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        env_key = str(key).upper()
        print(f"{env_key}={shell_quote_value(value)}")


def load_config_sidecar(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix == ".json":
        return read_json(p) or {}
    return read_json(p.with_suffix(p.suffix + ".config.json")) or {}


def result_complete(path: Path, kind: str) -> bool:
    if kind == "knn":
        try:
            import model.baseline_cache as bc

            df = bc.load_baseline_store(str(path))
        except Exception:
            df = pd.DataFrame()
    else:
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            df = pd.read_csv(path)
        except Exception:
            return False
    required = REQUIRED_RESULT_COLS.get(kind, [])
    if df.empty or any(col not in df.columns for col in required):
        return False
    for col in required:
        if df[col].isna().all():
            return False
    if kind == "araf":
        cfg = load_config_sidecar(str(path))
        tau_vals = [x for x in re.split(r"[,\s]+", str(cfg.get("lambda_tau", ""))) if x.strip()]
        seeds = [x for x in re.split(r"[,\s]+", str(cfg.get("seed", ""))) if x.strip()]
        expected = len(tau_vals) * len(seeds)
        if expected > 0 and len(df[["seed", "lambda_tau"]].drop_duplicates()) < expected:
            return False
    return True

def aggregate_results(result_root: Path) -> dict[str, Path]:
    summaries = result_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in sorted((result_root / "araf").glob("*.csv")):
        if not result_complete(path, "araf"):
            continue
        cfg = load_config_sidecar(path)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            row_cfg = dict(cfg)
            row_cfg["seed"] = row.get("seed", cfg.get("seed", ""))
            row_cfg["n_samples"] = row.get("n_samples", cfg.get("n_samples", ""))
            row_cfg["model_type"] = row.get("model_type", cfg.get("model_type", ""))
            rows.append(
                {
                    **common_metric_payload(row_cfg, path, "araf"),
                    "lambda_tau": row.get("lambda_tau", cfg.get("lambda_tau", np.nan)),
                    "epochs": cfg.get("epochs", np.nan),
                    "araf_latent_dim": row.get("araf_latent_dim", cfg.get("araf_latent_dim", np.nan)),
                    "araf_dropout": row.get("araf_dropout", cfg.get("araf_dropout", np.nan)),
                    "knn_k_grid": "",
                    "selected_knn_k": np.nan,
                    "val_auc": np.nan,
                    "val_rmse": np.nan,
                    "test_auc": row.get("auc_amortized", np.nan),
                    "test_rmse": row.get("rmse_amortized", np.nan),
                    "train_item_count": row.get("train_item_count", cfg.get("train_item_count", np.nan)),
                    "test_item_count": row.get("test_item_count", cfg.get("test_item_count", np.nan)),
                    "train_observed_count": row.get("train_observed_count", cfg.get("train_observed_count", np.nan)),
                    "test_observed_count": row.get("test_observed_count", cfg.get("test_observed_count", np.nan)),
                    "active_dims": row.get("active_dims", np.nan),
                    "status": "complete",
                }
            )

    baseline_paths = set((result_root / "baselines").glob("baseline_metrics*.csv"))
    for cfg_path in (result_root / "baselines").glob("baseline_metrics*.csv.config.json"):
        baseline_paths.add(Path(str(cfg_path).removesuffix(".config.json")))

    for path in sorted(baseline_paths):
        cfg = load_config_sidecar(path)
        try:
            import model.baseline_cache as bc

            df = bc.load_baseline_store(str(path))
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
        if df.empty:
            continue
        cfg_seed_tokens = [s for s in re.split(r"[,\s]+", str(cfg.get("seed", ""))) if s.strip()]
        cfg_seeds = {int(s) for s in cfg_seed_tokens} if cfg_seed_tokens else set()
        cfg_model_type = str(cfg.get("model_type", "")).strip().lower()
        cfg_n_samples = cfg.get("n_samples", None)
        if cfg_n_samples is not None:
            try:
                cfg_n_samples = int(cfg_n_samples)
            except Exception:
                cfg_n_samples = None
        if cfg_seeds:
            df = df[df["seed"].isin(cfg_seeds)]
        if cfg_model_type:
            df = df[df["model_type"].astype(str).str.lower() == cfg_model_type]
        if cfg_n_samples is not None:
            df = df[df["n_samples"] == cfg_n_samples]
        if df.empty:
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("auc_knn", np.nan)) or pd.isna(row.get("rmse_knn", np.nan)):
                continue
            row_cfg = dict(cfg)
            row_cfg["model_type"] = row.get("model_type", cfg.get("model_type", ""))
            row_cfg["seed"] = row.get("seed", cfg.get("seed", ""))
            row_cfg["n_samples"] = row.get("n_samples", cfg.get("n_samples", ""))
            rows.append(
                {
                    **common_metric_payload(row_cfg, path, "knn"),
                    "lambda_tau": np.nan,
                    "epochs": np.nan,
                    "araf_latent_dim": np.nan,
                    "araf_dropout": np.nan,
                    "knn_k_grid": cfg.get("knn_k_grid", ""),
                    "selected_knn_k": row.get("selected_knn_k", np.nan),
                    "val_auc": row.get("val_auc_knn", np.nan),
                    "val_rmse": row.get("val_rmse_knn", np.nan),
                    "test_auc": row.get("auc_knn", np.nan),
                    "test_rmse": row.get("rmse_knn", np.nan),
                    "active_dims": np.nan,
                    "status": "complete",
                }
            )

    metrics_long = pd.DataFrame(rows)
    metrics_long_path = summaries / "metrics_long.csv"
    manifest_path = summaries / "result_manifest.csv"
    metrics_wide_path = summaries / "metrics_wide.csv"
    if metrics_long.empty:
        metrics_long.to_csv(metrics_long_path, index=False)
        metrics_long.to_csv(manifest_path, index=False)
        metrics_long.to_csv(metrics_wide_path, index=False)
        return {
            "metrics_long_path": metrics_long_path,
            "metrics_wide_path": metrics_wide_path,
            "result_manifest_path": manifest_path,
        }

    metrics_long = metrics_long.sort_values(["method", "model_type", "seed", "araf_latent_dim", "araf_dropout", "lambda_tau"], na_position="last")
    metrics_long.to_csv(metrics_long_path, index=False)
    metrics_long.to_csv(manifest_path, index=False)

    join_cols = [
        "data_source",
        "dataset_selector",
        "corpus_slug",
        "embedding_slug",
        "embedding_type",
        "model_type",
        "seed",
        "test_size",
        "train_retention",
        "n_samples",
    ]
    wide_parts = []
    for method, sub in metrics_long.groupby("method"):
        if method == "araf":
            # Select best ARAF setup per unique corpus/seed configuration across tau/K/dropout.
            best_idx = sub.groupby(["corpus_slug", "seed"])["test_auc"].idxmax()
            sub = sub.loc[best_idx]
        
        keep = join_cols + ["test_auc", "test_rmse", "selected_knn_k", "lambda_tau", "epochs", "araf_latent_dim", "araf_dropout", "artifact_path"]
        renamed = sub[[col for col in keep if col in sub.columns]].copy()
        renamed = renamed.rename(
            columns={
                "test_auc": f"{method}_auc",
                "test_rmse": f"{method}_rmse",
                "artifact_path": f"{method}_artifact_path",
            }
        )
        wide_parts.append(renamed)
    if wide_parts:
        wide = wide_parts[0]
        for part in wide_parts[1:]:
            wide = wide.merge(part, on=[c for c in join_cols if c in wide.columns and c in part.columns], how="outer")
    else:
        wide = pd.DataFrame()
    wide.to_csv(metrics_wide_path, index=False)
    return {
        "metrics_long_path": metrics_long_path,
        "metrics_wide_path": metrics_wide_path,
        "result_manifest_path": manifest_path,
    }


def write_split_inventory(args: argparse.Namespace) -> dict[str, Path | int]:
    result_root = Path(args.result_root)
    summaries = result_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(args.corpus_path)
    corpus.columns = corpus.columns.astype(str)
    task_ids = corpus.columns.tolist()
    indices = np.arange(len(task_ids))
    np.random.seed(int(args.seed))
    np.random.shuffle(indices)
    n_test = int(float(args.test_size) * len(task_ids))
    test_pos = set(indices[:n_test].tolist())
    item_contents = pd.read_csv(args.item_content_path)
    embedded = set(item_contents["benchmark.task_id"].astype(str)) if "benchmark.task_id" in item_contents.columns else set()
    rows = []
    for pos, task_id in enumerate(task_ids):
        split = "test" if pos in test_pos else "train"
        dataset = str(task_id).split(".", 1)[0] if "." in str(task_id) else ""
        rows.append(
            {
                "item_id": task_id,
                "dataset": dataset,
                "split": split,
                "has_embedding": task_id in embedded,
                "observed_count": int(corpus[task_id].notna().sum()),
            }
        )
    df = pd.DataFrame(rows)
    safe_test = slugify(args.test_size)
    path = summaries / f"split_inventory_{args.corpus_slug}_seed{int(args.seed)}_test{safe_test}.csv"
    df.to_csv(path, index=False)
    return {
        "split_inventory_path": path,
        "train_item_count": int((df["split"] == "train").sum()),
        "test_item_count": int((df["split"] == "test").sum()),
        "train_observed_count": int(df.loc[df["split"] == "train", "observed_count"].sum()),
        "test_observed_count": int(df.loc[df["split"] == "test", "observed_count"].sum()),
    }


def common_metric_payload(config: dict[str, Any], artifact_path: Path, method: str) -> dict[str, Any]:
    corpus_cfg = load_config_sidecar(config.get("corpus_config_path", "")) if config.get("corpus_config_path") else {}
    embedding_cfg = load_config_sidecar(config.get("embedding_config_path", "")) if config.get("embedding_config_path") else {}
    log_path = config.get("log_path", "")
    return {
        "data_source": config.get("data_source", corpus_cfg.get("source", "")),
        "dataset_selector": config.get("dataset_selector", "all-ready" if corpus_cfg.get("all_ready") else ""),
        "dataset_names": "|".join(config.get("dataset_names", corpus_cfg.get("dataset_names", [])) or []),
        "corpus_slug": config.get("corpus_slug", corpus_cfg.get("corpus_slug", "")),
        "embedding_slug": config.get("embedding_slug", embedding_cfg.get("embedding_slug", "")),
        "method": method,
        "embedding_type": "raw",
        "model_type": config.get("model_type", ""),
        "seed": str(config.get("seed", "")),
        "test_size": config.get("test_size", 0.1),
        "train_retention": config.get("train_retention", 1.0),
        "n_samples": config.get("n_samples", 1),
        "dataset_count": corpus_cfg.get("included_dataset_count", config.get("dataset_count", np.nan)),
        "subject_count": corpus_cfg.get("subject_count", config.get("subject_count", np.nan)),
        "item_count": corpus_cfg.get("item_count", config.get("item_count", np.nan)),
        "observed_response_count": corpus_cfg.get("observed_response_count", config.get("observed_response_count", np.nan)),
        "density": corpus_cfg.get("density", config.get("density", np.nan)),
        "train_item_count": config.get("train_item_count", np.nan),
        "test_item_count": config.get("test_item_count", np.nan),
        "train_observed_count": config.get("train_observed_count", np.nan),
        "test_observed_count": config.get("test_observed_count", np.nan),
        "artifact_path": str(artifact_path),
        "config_path": str(artifact_path.with_suffix(artifact_path.suffix + ".config.json")),
        "log_path": log_path,
    }


def run_source_build(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", args.measurement_db_git, str(source_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(source_dir), "pull", "--ff-only"], check=True)
    env = os.environ.copy()
    env["NO_UPLOAD"] = "1"
    cmd = ["python", "reproduce.py", "--no-upload"]
    if args.datasets:
        cmd = ["python", "reproduce.py", *parse_dataset_names(args.datasets), "--no-upload"]
    subprocess.run(cmd, cwd=source_dir, env=env, check=True)


def cmd_prepare(args: argparse.Namespace) -> None:
    result_root = Path(args.result_root)
    ensure_result_dirs(result_root)
    if args.build_source:
        run_source_build(args)
    corpus_info = build_corpus(args, result_root)
    embedding_info = generate_embeddings(args, result_root, corpus_info)
    print_env({**corpus_info, **embedding_info, "result_root": result_root})


def cmd_aggregate(args: argparse.Namespace) -> None:
    result_root = Path(args.result_root)
    paths = aggregate_results(result_root)
    print_env(paths)


def cmd_split_inventory(args: argparse.Namespace) -> None:
    print_env(write_split_inventory(args))


def cmd_check_result(args: argparse.Namespace) -> None:
    complete = result_complete(Path(args.path), args.kind)
    raise SystemExit(0 if complete else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measurement-db RAW-only helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    prep.add_argument("--source", choices=["hf", "source"], default="hf")
    prep.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    prep.add_argument("--source-dir", default="")
    prep.add_argument("--measurement-db-git", default="https://github.com/aims-foundations/measurement-db.git")
    prep.add_argument("--build-source", action="store_true")
    prep.add_argument("--all-ready", action="store_true")
    prep.add_argument("--datasets", default=None)
    prep.add_argument("--min-subjects", type=int, default=4)
    prep.add_argument("--min-items", type=int, default=50)
    prep.add_argument("--response-policy", choices=["binary", "bounded01", "any"], default="bounded01")
    prep.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    prep.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    prep.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    prep.add_argument("--chunk-size", type=int, default=512)
    prep.add_argument("--quiet", action="store_true")
    prep.add_argument("--force", action="store_true")
    prep.set_defaults(func=cmd_prepare)

    agg = sub.add_parser("aggregate")
    agg.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    agg.set_defaults(func=cmd_aggregate)

    split = sub.add_parser("split-inventory")
    split.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    split.add_argument("--corpus-path", required=True)
    split.add_argument("--item-content-path", required=True)
    split.add_argument("--corpus-slug", required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--test-size", type=float, default=0.1)
    split.set_defaults(func=cmd_split_inventory)

    chk = sub.add_parser("check-result")
    chk.add_argument("--kind", choices=["araf", "knn"], required=True)
    chk.add_argument("--path", required=True)
    chk.set_defaults(func=cmd_check_result)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
