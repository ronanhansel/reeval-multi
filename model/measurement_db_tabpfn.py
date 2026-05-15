#!/usr/bin/env python3
"""TabPFN comparator for measurement-db RAW evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier
import tabpfn

# Add repo root to path for model.* imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utility.utils import compute_rmse, evaluate_auc
from model.measurement_db_raw import ensure_result_dirs, write_json, read_json


def load_split_safe_data(corpus_path: str, split_inventory_path: str):
    corpus = pd.read_parquet(corpus_path)
    corpus.columns = corpus.columns.astype(str)

    split_df = pd.read_csv(split_inventory_path)
    split_df["item_id"] = split_df["item_id"].astype(str)

    train_items = split_df[split_df["split"] == "train"]["item_id"].tolist()
    test_items = split_df[split_df["split"] == "test"]["item_id"].tolist()

    y_train_df = corpus[train_items]
    y_test_df = corpus[test_items]

    return y_train_df, y_test_df, train_items, test_items


def build_features(y_train_df: pd.DataFrame, y_test_df: pd.DataFrame, raw_embeddings: dict[str, np.ndarray], train_items: list[str], test_items: list[str], pca_dim: int = 64):
    # 1. Item PCA (fit on train items only)
    train_raw_embs = np.stack([raw_embeddings[item_id] for item_id in train_items])
    test_raw_embs = np.stack([raw_embeddings[item_id] for item_id in test_items])

    pca = PCA(n_components=min(pca_dim, len(train_items), train_raw_embs.shape[1]))
    pca.fit(train_raw_embs)

    train_item_pca = pca.transform(train_raw_embs)
    test_item_pca = pca.transform(test_raw_embs)

    # Pad if needed
    if train_item_pca.shape[1] < pca_dim:
        train_item_pca = np.pad(train_item_pca, ((0, 0), (0, pca_dim - train_item_pca.shape[1])))
        test_item_pca = np.pad(test_item_pca, ((0, 0), (0, pca_dim - test_item_pca.shape[1])))

    item_feat_map = {}
    for i, item_id in enumerate(train_items):
        item_feat_map[item_id] = train_item_pca[i]
    for i, item_id in enumerate(test_items):
        item_feat_map[item_id] = test_item_pca[i]

    # 2. User Stats (train only)
    user_mean = y_train_df.mean(axis=1)
    user_count = y_train_df.count(axis=1)
    user_var = y_train_df.var(axis=1).fillna(0)

    user_stats = pd.DataFrame({"user_mean": user_mean, "user_count": user_count, "user_var": user_var})

    # 3. Item Support Stats (train only)
    item_mean = y_train_df.mean(axis=0)
    item_count = y_train_df.count(axis=0)
    item_var = y_train_df.var(axis=0).fillna(0)

    # Global fallbacks for test items
    global_item_mean = float(y_train_df.stack().mean())
    global_item_var = float(y_train_df.stack().var())

    item_support_stats = {}
    for item_id in train_items:
        item_support_stats[item_id] = [item_mean[item_id], item_count[item_id], item_var[item_id]]
    for item_id in test_items:
        item_support_stats[item_id] = [global_item_mean, 0, global_item_var]

    return item_feat_map, user_stats, item_support_stats


def construct_table(y_df: pd.DataFrame, item_feat_map: dict[str, np.ndarray], user_stats: pd.DataFrame, item_support_stats: dict[str, list[float]], sample_n: int | None = None, seed: int = 42):
    obs = y_df.stack().reset_index()
    obs.columns = ["subject_id", "item_id", "response"]

    if sample_n and len(obs) > sample_n:
        obs = obs.sample(n=sample_n, random_state=seed)

    X_parts = []
    # User features
    u_feat = user_stats.loc[obs["subject_id"]].values
    X_parts.append(u_feat)

    # Item PCA features
    i_pca = np.stack([item_feat_map[iid] for iid in obs["item_id"]])
    X_parts.append(i_pca)

    # Item support features
    i_supp = np.stack([item_support_stats[iid] for iid in obs["item_id"]])
    X_parts.append(i_supp)

    X = np.concatenate(X_parts, axis=1)
    y = obs["response"].values

    return X, y


def main():
    parser = argparse.ArgumentParser(description="TabPFN Measurement-DB Comparator")
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--split-inventory-path", required=True)
    parser.add_argument("--result-root", default="model/result/measurement_db_raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows", type=int, default=50000)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--model-type", default="bernoulli")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    result_root = Path(args.result_root)
    ensure_result_dirs(result_root)

    # Build config and slug
    corpus_slug_val = Path(args.corpus_path).stem.replace("corpus_", "")
    corpus_config_path = str(args.corpus_path) + ".config.json"
    embedding_config_path = str(args.embeddings_path) + ".config.json"
    corpus_cfg = read_json(Path(corpus_config_path)) or {}
    embedding_cfg = read_json(Path(embedding_config_path)) or {}
    feature_set = "user_stats3_item_pca_item_train_stats3_v1"
    config = {
        "kind": "measurement_db_raw_tabpfn",
        "data_source": corpus_cfg.get("source", ""),
        "dataset_selector": "all-ready" if corpus_cfg.get("all_ready") else "",
        "corpus_slug": corpus_cfg.get("corpus_slug", corpus_slug_val),
        "corpus_config_path": corpus_config_path,
        "embedding_slug": embedding_cfg.get("embedding_slug", ""),
        "embedding_config_path": embedding_config_path,
        "corpus_path": str(args.corpus_path),
        "embeddings_path": str(args.embeddings_path),
        "split_inventory_path": str(args.split_inventory_path),
        "seed": args.seed,
        "train_rows": args.train_rows,
        "pca_dim": args.pca_dim,
        "model_type": args.model_type,
        "tabpfn_version": getattr(tabpfn, "__version__", "unknown"),
        "feature_set": feature_set,
    }

    (result_root / "tabpfn").mkdir(parents=True, exist_ok=True)
    output_path = result_root / "tabpfn" / f"tabpfn_raw_corpus-{corpus_slug_val}-seed-{args.seed}-rows-{args.train_rows}-pca-{args.pca_dim}.csv"
    config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    log_path = result_root / "logs" / f"tabpfn_corpus-{corpus_slug_val}-seed-{args.seed}.log"

    if output_path.exists() and config_path.exists() and not args.force:
        try:
            old_config = read_json(config_path) or {}
            if all(old_config.get(key) == value for key, value in config.items()):
                print(f"Skipping existing TabPFN result: {output_path}")
                return
        except Exception:
            pass

    print(f"Running TabPFN for seed {args.seed}...")
    start_time = time.time()

    # 1. Load Data
    y_train_df, y_test_df, train_items, test_items = load_split_safe_data(args.corpus_path, args.split_inventory_path)
    raw_emb_map = pd.read_pickle(args.embeddings_path)
    if isinstance(raw_emb_map, pd.DataFrame):
        id_col = "benchmark.task_id" if "benchmark.task_id" in raw_emb_map.columns else "task_id"
        raw_emb_dict = dict(zip(raw_emb_map[id_col].astype(str), raw_emb_map["embedding"]))
    else:
        raw_emb_dict = {str(k): v for k, v in raw_emb_map.items()}

    # 2. Build Features
    item_feat_map, user_stats, item_support_stats = build_features(y_train_df, y_test_df, raw_emb_dict, train_items, test_items, pca_dim=args.pca_dim)

    # 3. Construct Tables
    X_train, y_train = construct_table(y_train_df, item_feat_map, user_stats, item_support_stats, sample_n=args.train_rows, seed=args.seed)
    X_test, y_test = construct_table(y_test_df, item_feat_map, user_stats, item_support_stats)

    # Ensure binary for TabPFNClassifier if bernoulli
    if args.model_type == "bernoulli":
        y_train_bin = (y_train > 0.5).astype(int)
        y_test_bin = (y_test > 0.5).astype(int)
    else:
        y_train_bin = y_train.astype(int)
        y_test_bin = y_test.astype(int)

    # 4. Fit and Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = TabPFNClassifier(device=device, random_state=args.seed, ignore_pretraining_limits=True)
    classifier.fit(X_train, y_train_bin)

    # Batch predict if test is large
    batch_size = 10000
    p_test = []
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i : i + batch_size]
        p_test.append(classifier.predict_proba(batch_X)[:, 1])
    p_test = np.concatenate(p_test)

    # 5. Metrics
    p_test_t = torch.from_numpy(p_test)
    y_test_t = torch.from_numpy(y_test_bin)
    mask_t = torch.ones_like(y_test_t, dtype=torch.bool)
    auc = evaluate_auc(p_test_t, y_test_t, mask_t)
    rmse = compute_rmse(p_test, y_test_bin.astype(float), np.ones_like(y_test_bin, dtype=bool))

    elapsed = time.time() - start_time
    print(f"TabPFN seed {args.seed} done in {elapsed:.1f}s. AUC: {auc:.4f}, RMSE: {rmse:.4f}")

    # 6. Save
    res_df = pd.DataFrame(
        [
            {
                "seed": args.seed,
                "model_type": args.model_type,
                "test_auc": auc,
                "test_rmse": rmse,
                "tabpfn_train_rows": args.train_rows,
                "tabpfn_actual_train_rows": len(y_train),
                "tabpfn_item_pca_dim": args.pca_dim,
                "tabpfn_feature_set": feature_set,
                "tabpfn_package_version": getattr(tabpfn, "__version__", "unknown"),
                "elapsed_seconds": elapsed,
                "train_item_count": len(train_items),
                "test_item_count": len(test_items),
                "train_observed_count": int(y_train_df.notna().sum().sum()),
                "test_observed_count": len(y_test),
            }
        ]
    )
    res_df.to_csv(output_path, index=False)

    full_config = {**config, "log_path": str(log_path), "elapsed": elapsed, "auc": auc, "rmse": rmse}
    write_json(config_path, full_config)


if __name__ == "__main__":
    main()
