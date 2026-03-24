#!/usr/bin/env python3
"""
Standalone post-hoc calibration study for unseen-item prediction.

This script:
1. Scans existing result files to summarize the best observed AUC configurations for
   ARAF, kNN, Rasch, and MIRT.
2. Selects a fair common beta condition shared across methods.
3. Re-runs the chosen methods on a train / calibration / test unseen-item split.
4. Fits symmetric post-hoc calibrators on the calibration split only.
5. Saves per-example predictions, aggregated metrics, and summary figures.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

import amortized_irt as ai
from utils import compute_rmse, evaluate_auc


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "model" / "result"
DEFAULT_OUT_DIR = RESULT_ROOT / "calibration_study"

CAL_FRACTION = 0.10
TEST_FRACTION = 0.10
NOVELTY_ORDER = ["inlier", "moderate", "outlier"]
DEFAULT_RETENTIONS = [1.0, 0.5, 0.25, 0.1, 0.05]
MIN_ISOTONIC_POINTS = 200
EPS = 1e-6


@dataclass(frozen=True)
class Condition:
    model_type: str
    pre_revision: str
    j_percentage: float
    n_samples: int
    train_retention: float


def normalize_pre_revision(value) -> str:
    return ai.normalize_pre_revision(value)


def normalize_j_percentage(value) -> float:
    return ai.normalize_j_percentage(value)


def parse_amortized_path(path: Path) -> Optional[Dict]:
    name = path.name
    m = re.match(
        r"amortized_irt_(?P<embedding>[^_]+)_(?P<model>beta|bernoulli)"
        r"(?:_pre_(?P<pre>[^_]+))?"
        r"_n_(?P<n>max|\d+)"
        r"(?:_notau)?"
        r"(?:_j(?P<j>[0-9.]+))?"
        r"(?:_b(?P<baseline_emb>[^_]+)_k(?P<knn_k>\d+))?"
        r"\.csv$",
        name,
    )
    if not m:
        return None

    out = m.groupdict()
    out["embedding_type"] = out.pop("embedding")
    out["model_type"] = out.pop("model")
    out["pre_revision"] = normalize_pre_revision(out.pop("pre") or "none")
    n_token = out.pop("n")
    out["n_token"] = n_token
    out["j_percentage"] = normalize_j_percentage(float(out.pop("j") or 1.0))
    out["baseline_embedding_type"] = out.pop("baseline_emb")
    out["knn_k"] = int(out["knn_k"]) if out["knn_k"] is not None else None

    retention = 1.0
    for part in path.parts:
        if part.startswith("retain_"):
            try:
                retention = float(part.split("_", 1)[1])
            except ValueError:
                pass
    out["train_retention"] = retention
    return out


def parse_baseline_context(path: Path) -> Dict:
    retention = 1.0
    knn_k = 10
    path_str = str(path)
    for part in path.parts:
        if part.startswith("retain_"):
            try:
                retention = float(part.split("_", 1)[1])
            except ValueError:
                pass
    m = re.search(r"knn_(raw|pca|sae)_k(\d+)", path_str)
    if m:
        knn_k = int(m.group(2))
    return {"train_retention": retention, "knn_k": knn_k}


def scan_araf_candidates(root: Path, model_type: str = "beta") -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("amortized_irt_*.csv")):
        meta = parse_amortized_path(path)
        if meta is None or meta["model_type"] != model_type:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty or "auc_amortized" not in df.columns or "lambda_tau" not in df.columns:
            continue
        for col in ["seed", "lambda_tau", "auc_amortized", "rmse_amortized", "n_samples"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["seed", "lambda_tau", "auc_amortized", "rmse_amortized", "n_samples"])
        if df.empty:
            continue
        grouped = df.groupby("lambda_tau", as_index=False).agg(
            mean_auc=("auc_amortized", "mean"),
            mean_rmse=("rmse_amortized", "mean"),
            n_seeds=("seed", "nunique"),
            n_samples=("n_samples", "median"),
        )
        for _, row in grouped.iterrows():
            rows.append({
                "method": "ARAF",
                "source_path": str(path),
                "embedding_type": meta["embedding_type"],
                "lambda_tau": float(row["lambda_tau"]),
                "model_type": meta["model_type"],
                "pre_revision": meta["pre_revision"],
                "j_percentage": float(meta["j_percentage"]),
                "n_samples": int(row["n_samples"]),
                "train_retention": float(meta["train_retention"]),
                "mean_auc": float(row["mean_auc"]),
                "mean_rmse": float(row["mean_rmse"]),
                "n_seeds": int(row["n_seeds"]),
            })
    return pd.DataFrame(rows)


def _mode_or_default(values: pd.Series, default: int) -> int:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if vals.empty:
        return int(default)
    mode = vals.mode()
    if mode.empty:
        return int(default)
    return int(mode.iloc[0])


def scan_baseline_candidates(root: Path, model_type: str = "beta") -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("baseline_metrics.csv")):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty or "model_type" not in df.columns:
            continue
        ctx = parse_baseline_context(path)
        df["model_type"] = df["model_type"].astype(str).str.lower().str.strip()
        df = df[df["model_type"] == model_type]
        if df.empty:
            continue

        for col in [
            "seed", "n_samples", "j_percentage", "auc_knn", "rmse_knn", "auc_rasch", "rmse_rasch",
            "auc_mirt", "rmse_mirt", "selected_mirt_dim"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["pre_revision"] = df["pre_revision"].astype(str).map(normalize_pre_revision)
        if "baseline_embedding_type" not in df.columns:
            df["baseline_embedding_type"] = "raw"
        df["baseline_embedding_type"] = df["baseline_embedding_type"].astype(str).str.lower().str.strip()

        group_cols = ["pre_revision", "j_percentage", "n_samples", "baseline_embedding_type"]
        grouped = df.groupby(group_cols, as_index=False).agg(
            mean_auc_knn=("auc_knn", "mean"),
            mean_rmse_knn=("rmse_knn", "mean"),
            mean_auc_rasch=("auc_rasch", "mean"),
            mean_rmse_rasch=("rmse_rasch", "mean"),
            mean_auc_mirt=("auc_mirt", "mean"),
            mean_rmse_mirt=("rmse_mirt", "mean"),
            n_seeds=("seed", "nunique"),
        )
        for _, row in grouped.iterrows():
            base = {
                "source_path": str(path),
                "model_type": model_type,
                "pre_revision": normalize_pre_revision(row["pre_revision"]),
                "j_percentage": float(row["j_percentage"]),
                "n_samples": int(row["n_samples"]),
                "baseline_embedding_type": str(row["baseline_embedding_type"]),
                "train_retention": float(ctx["train_retention"]),
                "knn_k": int(ctx["knn_k"]),
                "n_seeds": int(row["n_seeds"]),
            }
            if not math.isnan(row["mean_auc_knn"]):
                rows.append({
                    **base,
                    "method": "kNN",
                    "mean_auc": float(row["mean_auc_knn"]),
                    "mean_rmse": float(row["mean_rmse_knn"]),
                    "selected_mirt_dim": np.nan,
                })
            if not math.isnan(row["mean_auc_rasch"]):
                rows.append({
                    **base,
                    "method": "Rasch",
                    "mean_auc": float(row["mean_auc_rasch"]),
                    "mean_rmse": float(row["mean_rmse_rasch"]),
                    "selected_mirt_dim": np.nan,
                })
            if not math.isnan(row["mean_auc_mirt"]):
                sub = df[
                    (df["pre_revision"] == base["pre_revision"]) &
                    (np.isclose(df["j_percentage"], base["j_percentage"], atol=1e-9)) &
                    (df["n_samples"].astype(int) == base["n_samples"]) &
                    (df["baseline_embedding_type"] == base["baseline_embedding_type"])
                ]
                rows.append({
                    **base,
                    "method": "MIRT",
                    "mean_auc": float(row["mean_auc_mirt"]),
                    "mean_rmse": float(row["mean_rmse_mirt"]),
                    "selected_mirt_dim": _mode_or_default(sub.get("selected_mirt_dim", pd.Series(dtype=float)), ai.K_MODEL),
                })
    return pd.DataFrame(rows)


def select_global_bests(araf_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    frames = [araf_df, baseline_df]
    frames = [df for df in frames if df is not None and not df.empty]
    all_candidates = pd.concat(frames, ignore_index=True)
    rows = []
    for method in ["ARAF", "kNN", "Rasch", "MIRT"]:
        sub = all_candidates[all_candidates["method"] == method]
        if sub.empty:
            continue
        pick = sub.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0]
        rows.append(pick.to_dict())
    return pd.DataFrame(rows)


def select_best_common_condition(araf_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Tuple[Condition, pd.DataFrame]:
    araf_conditions = araf_df[["model_type", "pre_revision", "j_percentage", "n_samples", "train_retention"]].drop_duplicates()
    base_conditions = baseline_df[["model_type", "pre_revision", "j_percentage", "n_samples", "train_retention"]].drop_duplicates()
    merged = araf_conditions.merge(base_conditions, on=["model_type", "pre_revision", "j_percentage", "n_samples", "train_retention"])
    if merged.empty:
        raise RuntimeError("No common beta evaluation condition found across ARAF and baselines.")

    scored_rows = []
    for _, cond_row in merged.iterrows():
        condition = Condition(
            model_type=str(cond_row["model_type"]),
            pre_revision=normalize_pre_revision(cond_row["pre_revision"]),
            j_percentage=float(cond_row["j_percentage"]),
            n_samples=int(cond_row["n_samples"]),
            train_retention=float(cond_row["train_retention"]),
        )
        row = {"condition": condition}
        ok = True

        sub_araf = araf_df[
            (araf_df["model_type"] == condition.model_type) &
            (araf_df["pre_revision"] == condition.pre_revision) &
            (np.isclose(araf_df["j_percentage"], condition.j_percentage, atol=1e-9)) &
            (araf_df["n_samples"].astype(int) == condition.n_samples) &
            (np.isclose(araf_df["train_retention"], condition.train_retention, atol=1e-9))
        ]
        if sub_araf.empty:
            ok = False
        else:
            pick = sub_araf.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0]
            row["ARAF_auc"] = float(pick["mean_auc"])

        for method in ["kNN", "Rasch", "MIRT"]:
            sub = baseline_df[
                (baseline_df["method"] == method) &
                (baseline_df["model_type"] == condition.model_type) &
                (baseline_df["pre_revision"] == condition.pre_revision) &
                (np.isclose(baseline_df["j_percentage"], condition.j_percentage, atol=1e-9)) &
                (baseline_df["n_samples"].astype(int) == condition.n_samples) &
                (np.isclose(baseline_df["train_retention"], condition.train_retention, atol=1e-9))
            ]
            if sub.empty:
                ok = False
                break
            pick = sub.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0]
            row[f"{method}_auc"] = float(pick["mean_auc"])

        if ok:
            row["score"] = np.mean([row["ARAF_auc"], row["kNN_auc"], row["Rasch_auc"], row["MIRT_auc"]])
            scored_rows.append(row)

    if not scored_rows:
        raise RuntimeError("No shared condition with all four methods available.")

    scored = pd.DataFrame(scored_rows).sort_values("score", ascending=False)
    best_cond = scored.iloc[0]["condition"]

    picks = []
    sub_araf = araf_df[
        (araf_df["model_type"] == best_cond.model_type) &
        (araf_df["pre_revision"] == best_cond.pre_revision) &
        (np.isclose(araf_df["j_percentage"], best_cond.j_percentage, atol=1e-9)) &
        (araf_df["n_samples"].astype(int) == best_cond.n_samples) &
        (np.isclose(araf_df["train_retention"], best_cond.train_retention, atol=1e-9))
    ]
    picks.append(sub_araf.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0].to_dict())
    for method in ["kNN", "Rasch", "MIRT"]:
        sub = baseline_df[
            (baseline_df["method"] == method) &
            (baseline_df["model_type"] == best_cond.model_type) &
            (baseline_df["pre_revision"] == best_cond.pre_revision) &
            (np.isclose(baseline_df["j_percentage"], best_cond.j_percentage, atol=1e-9)) &
            (baseline_df["n_samples"].astype(int) == best_cond.n_samples) &
            (np.isclose(baseline_df["train_retention"], best_cond.train_retention, atol=1e-9))
        ]
        picks.append(sub.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0].to_dict())

    return best_cond, pd.DataFrame(picks)


def select_configs_for_condition(araf_df: pd.DataFrame, baseline_df: pd.DataFrame, condition: Condition) -> pd.DataFrame:
    picks = []
    sub_araf = araf_df[
        (araf_df["model_type"] == condition.model_type) &
        (araf_df["pre_revision"] == condition.pre_revision) &
        (np.isclose(araf_df["j_percentage"], condition.j_percentage, atol=1e-9)) &
        (araf_df["n_samples"].astype(int) == condition.n_samples) &
        (np.isclose(araf_df["train_retention"], condition.train_retention, atol=1e-9))
    ]
    if sub_araf.empty:
        raise RuntimeError(f"No ARAF candidates found for forced condition: {condition}")
    picks.append(sub_araf.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0].to_dict())

    for method in ["kNN", "Rasch", "MIRT"]:
        sub = baseline_df[
            (baseline_df["method"] == method) &
            (baseline_df["model_type"] == condition.model_type) &
            (baseline_df["pre_revision"] == condition.pre_revision) &
            (np.isclose(baseline_df["j_percentage"], condition.j_percentage, atol=1e-9)) &
            (baseline_df["n_samples"].astype(int) == condition.n_samples) &
            (np.isclose(baseline_df["train_retention"], condition.train_retention, atol=1e-9))
        ]
        if sub.empty:
            raise RuntimeError(f"No {method} candidates found for forced condition: {condition}")
        picks.append(sub.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0].to_dict())
    return pd.DataFrame(picks)


def set_all_seeds(seed: int):
    ai.RANDOM_SEED = int(seed)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def prepare_calibration_data(
    all_dfs, global_shared_indices, raw_embs_map,
    embedding_type: str, j_percentage: float, seed: int,
    cal_fraction: float = CAL_FRACTION, test_fraction: float = TEST_FRACTION,
):
    set_all_seeds(seed)

    all_columns = sorted(list(set().union(*[df.columns for df in all_dfs])))
    oracle_dfs_filtered = [df.reindex(index=global_shared_indices, columns=all_columns) for df in all_dfs]
    oracle_stacked = np.array([df.values for df in oracle_dfs_filtered], dtype=float)
    oracle_matrix = np.nanmean(oracle_stacked, axis=0)
    oracle_df = pd.DataFrame(oracle_matrix, index=global_shared_indices, columns=all_columns)

    N, J_full = oracle_df.shape
    if j_percentage < 1.0:
        n_keep = max(10, int(j_percentage * J_full))
        rng = np.random.default_rng(seed + 999)
        sampled_j_indices = np.sort(rng.choice(np.arange(J_full), size=n_keep, replace=False))
        oracle_df = oracle_df.iloc[:, sampled_j_indices]

    N, J = oracle_df.shape
    sampled_columns = oracle_df.columns.tolist()
    all_dfs_filtered = [df.reindex(columns=sampled_columns) for df in all_dfs]

    item_indices = np.arange(J)
    rng = np.random.default_rng(seed)
    rng.shuffle(item_indices)
    n_test = max(1, int(round(test_fraction * J)))
    n_cal = max(1, int(round(cal_fraction * J)))
    if n_test + n_cal >= J:
        n_test = max(1, int(math.floor(0.1 * J)))
        n_cal = max(1, min(int(math.floor(0.1 * J)), J - n_test - 1))

    test_idx = item_indices[:n_test]
    cal_idx = item_indices[n_test:n_test + n_cal]
    train_idx = item_indices[n_test + n_cal:]

    oracle_values_clean = np.nan_to_num(oracle_df.values, nan=0.5)
    y_oracle = torch.from_numpy(oracle_values_clean.astype(np.float32)).to(ai.device)

    train_mask = np.zeros_like(oracle_df.values, dtype=bool)
    cal_mask = np.zeros_like(oracle_df.values, dtype=bool)
    test_mask = np.zeros_like(oracle_df.values, dtype=bool)
    train_mask[:, train_idx] = ~np.isnan(oracle_df.values)[:, train_idx]
    cal_mask[:, cal_idx] = ~np.isnan(oracle_df.values)[:, cal_idx]
    test_mask[:, test_idx] = ~np.isnan(oracle_df.values)[:, test_idx]

    task_ids = oracle_df.columns.tolist()
    if embedding_type in ["ones", "rasch_2pl", "nonamortised_mirt"]:
        embeddings = np.ones((len(task_ids), 1), dtype=np.float32)
    else:
        embs = []
        for task_id in task_ids:
            emb = raw_embs_map.get(str(task_id))
            if emb is None and str(task_id).startswith("colbench."):
                suffix = str(task_id).split(".")[-1]
                emb = raw_embs_map.get(f"colbench_backend_programming.{suffix}")
            if emb is None:
                sample_emb = next(iter(raw_embs_map.values()))
                emb = np.zeros(len(sample_emb), dtype=np.float32)
            elif isinstance(emb, str):
                emb = ai.ast.literal_eval(emb)
            embs.append(np.array(emb, dtype=np.float32))
        embeddings = np.stack(embs)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    x_j = torch.tensor(embeddings, dtype=torch.float32).to(ai.device)

    return {
        "all_dfs": all_dfs_filtered,
        "y_oracle": y_oracle,
        "x_j": x_j,
        "N": N,
        "J": J,
        "embedding_dim": int(x_j.shape[1]),
        "train_idx": train_idx,
        "cal_idx": cal_idx,
        "test_idx": test_idx,
        "train_mask_t": torch.from_numpy(train_mask).to(ai.device),
        "cal_mask_t": torch.from_numpy(cal_mask).to(ai.device),
        "test_mask_t": torch.from_numpy(test_mask).to(ai.device),
        "train_mask": train_mask,
        "cal_mask": cal_mask,
        "test_mask": test_mask,
        "task_ids": task_ids,
        "agent_ids": list(global_shared_indices),
    }


def train_amortized_no_leakage(model, y_train, train_mask_t, model_type: str, beta_phi: float, lambda_tau: float, quiet: bool = True):
    fit_mask_t, val_mask_t = ai.build_mirt_validation_masks(y_train, train_mask_t)
    optimizer = torch.optim.AdamW([
        {"params": [model.theta, model.theta_bias], "lr": ai.LR_THETA, "weight_decay": ai.WD_THETA},
        {"params": [model.W, model.global_bias], "lr": ai.LR_GLOBAL, "weight_decay": ai.WD_W},
        {"params": model.difficulty_proj.parameters(), "lr": ai.LR_GLOBAL},
    ])
    optimizer_tau = torch.optim.SGD([model.tau_raw], lr=0.05)

    best_score = float("inf")
    best_state = None
    eps = 1e-6

    for epoch in range(ai.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        optimizer_tau.zero_grad()
        probs = model()
        p = probs[fit_mask_t].clamp(eps, 1 - eps)
        if model_type == "beta":
            y = y_train[fit_mask_t].clamp(eps, 1 - eps)
            dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
            loss_fit = -dist.log_prob(y).mean()
        else:
            y = y_train[fit_mask_t]
            dist = torch.distributions.Bernoulli(probs=p)
            loss_fit = -dist.log_prob(y).mean()

        if epoch < ai.TAU_WARMUP:
            current_lambda = 0.0
        elif epoch < ai.TAU_WARMUP + ai.RAMP_EPOCHS:
            progress = (epoch - ai.TAU_WARMUP) / ai.RAMP_EPOCHS
            current_lambda = lambda_tau * progress
        else:
            current_lambda = lambda_tau

        loss = loss_fit + current_lambda * torch.sum(model.get_tau())
        loss.backward()
        optimizer.step()
        optimizer_tau.step()

        if epoch > ai.TAU_WARMUP + 50 and epoch % 10 == 0:
            with torch.no_grad():
                active_mask = model.get_tau() > ai.SNAPPING_THRESHOLD
                for k in range(ai.K_MODEL):
                    if not active_mask[k]:
                        model.tau_raw[k] = ai.DEAD_ZONE_VALUE

        if epoch % ai.EVAL_EVERY == 0 or epoch == ai.EPOCHS:
            model.eval()
            with torch.no_grad():
                p_eval = model()
                eval_mask = val_mask_t if val_mask_t is not None else fit_mask_t
                prob = ai.compute_prob_metrics(p_eval, y_train, eval_mask)
                score = prob["log_loss"] if np.isfinite(prob["log_loss"]) else prob["brier"]
                if score < best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(ai.device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        probs = model().detach().clone()
    return probs, best_state


def train_mirt_no_leakage(N, J, y_train, train_mask_t, model_type: str, beta_phi: float, mirt_dim: int):
    fit_mask_t, val_mask_t = ai.build_mirt_validation_masks(y_train, train_mask_t)
    model = ai.MIRTModel(N, J, mirt_dim).to(ai.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)

    best_score = float("inf")
    best_state = None
    for _ in range(ai.EPOCHS // 2):
        optimizer.zero_grad()
        probs = model()
        p = probs[fit_mask_t].clamp(EPS, 1 - EPS)
        y = y_train[fit_mask_t]
        if model_type == "beta":
            y = y.clamp(EPS, 1 - EPS)
            dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
            loss = -dist.log_prob(y).mean()
        else:
            dist = torch.distributions.Bernoulli(probs=p)
            loss = -dist.log_prob(y).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p_eval = model()
            eval_mask = val_mask_t if val_mask_t is not None else fit_mask_t
            prob = ai.compute_prob_metrics(p_eval, y_train, eval_mask)
            score = prob["log_loss"] if np.isfinite(prob["log_loss"]) else prob["brier"]
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(ai.device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        probs = model().detach().clone()
    return probs, best_state


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class RawCalibrator:
    name = "raw"

    def fit(self, score, y):
        return self

    def predict(self, score, prob=None):
        if prob is not None:
            return np.clip(prob, EPS, 1 - EPS)
        return 1.0 / (1.0 + np.exp(-score))


class SigmoidCalibrator:
    name = "sigmoid"

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, score, y):
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=float)

        def objective(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * score + b)))
            p = np.clip(p, EPS, 1 - EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize(objective, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        if res.success:
            self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict(self, score, prob=None):
        score = np.asarray(score, dtype=float)
        return np.clip(1.0 / (1.0 + np.exp(-(self.a * score + self.b))), EPS, 1 - EPS)


class TemperatureCalibrator:
    name = "temperature"

    def __init__(self):
        self.temperature = 1.0

    def fit(self, score, y):
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=float)

        def objective(log_t):
            t = np.exp(log_t[0])
            p = 1.0 / (1.0 + np.exp(-(score / max(t, EPS))))
            p = np.clip(p, EPS, 1 - EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B")
        if res.success:
            self.temperature = float(np.exp(res.x[0]))
        return self

    def predict(self, score, prob=None):
        score = np.asarray(score, dtype=float)
        return np.clip(1.0 / (1.0 + np.exp(-(score / max(self.temperature, EPS)))), EPS, 1 - EPS)


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, score, y):
        self.model.fit(np.asarray(score, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, score, prob=None):
        return np.clip(self.model.predict(np.asarray(score, dtype=float)), EPS, 1 - EPS)


def masked_arrays(prob_tensor: torch.Tensor, y_tensor: torch.Tensor, mask_t: torch.Tensor):
    valid = mask_t.detach()
    y = y_tensor[valid].detach().cpu().numpy().astype(float)
    p = prob_tensor[valid].detach().cpu().numpy().astype(float)
    return y, p


def build_novelty_masks(x_j: torch.Tensor, train_idx, item_names: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, str]]:
    train_idx = np.asarray(train_idx, dtype=int)
    all_idx = np.arange(x_j.shape[0])
    heldout_idx = np.setdiff1d(all_idx, train_idx)
    if train_idx.size == 0 or heldout_idx.size == 0:
        empty = torch.zeros((x_j.shape[0],), dtype=torch.bool, device=x_j.device)
        return {"inlier": empty, "moderate": empty, "outlier": empty}, {"q50": np.nan, "q80": np.nan}, {name: "inlier" for name in item_names}

    x_norm = F.normalize(x_j, dim=1)
    sims = x_norm[heldout_idx] @ x_norm[torch.as_tensor(train_idx, device=x_j.device)].T
    d_min = 1.0 - sims.max(dim=1).values.detach().cpu().numpy()
    q50 = float(np.quantile(d_min, 0.50))
    q80 = float(np.quantile(d_min, 0.80))

    masks = {}
    label_map = {}
    for name, pred in [
        ("inlier", d_min <= q50),
        ("moderate", (d_min > q50) & (d_min <= q80)),
        ("outlier", d_min > q80),
    ]:
        mask = torch.zeros((x_j.shape[0],), dtype=torch.bool, device=x_j.device)
        selected = heldout_idx[pred]
        if selected.size:
            mask[selected] = True
            for idx in selected:
                label_map[item_names[int(idx)]] = name
        masks[name] = mask
    return masks, {"q50": q50, "q80": q80}, label_map


def fit_calibrators(method: str, cal_probs: np.ndarray, cal_y: np.ndarray):
    score = logit_np(cal_probs)
    calibrators = [RawCalibrator(), SigmoidCalibrator()]
    if method in {"ARAF", "Rasch", "MIRT"}:
        calibrators.append(TemperatureCalibrator())
    if cal_y.size >= MIN_ISOTONIC_POINTS:
        calibrators.append(IsotonicCalibrator())

    fitted = []
    for calibrator in calibrators:
        calibrator.fit(score, cal_y)
        fitted.append(calibrator)
    return fitted


def metric_row(method, calibration_type, seed, retention, observed_pairs, novelty_bin, split_name, prob_tensor, y_tensor, mask_t):
    mask_np = mask_t.detach().cpu().numpy().astype(bool)
    return {
        "method": method,
        "calibration": calibration_type,
        "seed": int(seed),
        "train_retention": float(retention),
        "observed_train_pairs": int(observed_pairs),
        "novelty_bin": novelty_bin,
        "split": split_name,
        "auc": float(evaluate_auc(prob_tensor, y_tensor, mask_t)),
        "rmse": float(compute_rmse(prob_tensor.detach().cpu().numpy(), y_tensor.detach().cpu().numpy(), mask_np)),
        **ai.compute_prob_metrics(prob_tensor, y_tensor, mask_t),
    }


def select_config_for_retention(method: str, common_configs: pd.DataFrame, araf_df: pd.DataFrame, baseline_df: pd.DataFrame, condition: Condition, retention: float) -> Dict:
    if method == "ARAF":
        sub = araf_df[
            (araf_df["model_type"] == condition.model_type) &
            (araf_df["pre_revision"] == condition.pre_revision) &
            (np.isclose(araf_df["j_percentage"], condition.j_percentage, atol=1e-9)) &
            (araf_df["n_samples"].astype(int) == condition.n_samples) &
            (np.isclose(araf_df["train_retention"], retention, atol=1e-9))
        ]
        if sub.empty:
            sub = common_configs[common_configs["method"] == "ARAF"]
        return sub.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0].to_dict()

    sub = baseline_df[
        (baseline_df["method"] == method) &
        (baseline_df["model_type"] == condition.model_type) &
        (baseline_df["pre_revision"] == condition.pre_revision) &
        (np.isclose(baseline_df["j_percentage"], condition.j_percentage, atol=1e-9)) &
        (baseline_df["n_samples"].astype(int) == condition.n_samples) &
        (np.isclose(baseline_df["train_retention"], retention, atol=1e-9))
    ]
    if sub.empty:
        sub = common_configs[common_configs["method"] == method]
    return sub.sort_values(["mean_auc", "mean_rmse"], ascending=[False, True]).iloc[0].to_dict()


def run_method_predictions(method: str, config: Dict, seed: int, retention: float, data_cache: Dict[Tuple[str, str], Tuple], beta_phi: float):
    pre_revision = config["pre_revision"]
    model_type = str(config["model_type"])
    n_samples = int(config["n_samples"])
    j_percentage = float(config["j_percentage"])

    if method == "ARAF":
        emb_type = str(config["embedding_type"])
    elif method == "kNN":
        emb_type = str(config["baseline_embedding_type"])
    else:
        emb_type = "raw"

    cache_key = (pre_revision, emb_type)
    if cache_key not in data_cache:
        data_cache[cache_key] = ai.load_data(embedding_type=emb_type, pre_revision=pre_revision)
    all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = data_cache[cache_key]

    cal_data = prepare_calibration_data(
        all_dfs, global_shared_indices, raw_embs_map, actual_emb_type,
        j_percentage=j_percentage, seed=seed, cal_fraction=CAL_FRACTION, test_fraction=TEST_FRACTION,
    )
    N, J, y_train, train_mask_t = ai.build_training_targets(
        n_samples, all_dfs, global_shared_indices, cal_data,
        model_type=model_type, quiet=True, train_retention=retention,
    )
    observed_pairs = int(train_mask_t.sum().item())

    if method == "ARAF":
        ai.LAMBDA_TAU = float(config["lambda_tau"])
        model = ai.AmortizedIRTModel(N, J, ai.K_MODEL, cal_data["embedding_dim"], cal_data["x_j"], dropout=0.5, no_tau=False).to(ai.device)
        probs, _ = train_amortized_no_leakage(
            model, y_train, train_mask_t, model_type=model_type, beta_phi=beta_phi, lambda_tau=float(config["lambda_tau"])
        )
    elif method == "kNN":
        probs, _ = ai.compute_knn_predictions(
            y_train, train_mask_t, cal_data["x_j"], cal_data["test_mask"], knn_k=int(config.get("knn_k", 10))
        )
    elif method == "Rasch":
        probs = ai.train_rasch(N, J, y_train, train_mask_t)
    elif method == "MIRT":
        probs, _ = train_mirt_no_leakage(
            N, J, y_train, train_mask_t, model_type=model_type, beta_phi=beta_phi,
            mirt_dim=int(config.get("selected_mirt_dim", ai.K_MODEL))
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return probs, cal_data, y_train, train_mask_t, observed_pairs


def build_prediction_rows(method, calibration_name, seed, retention, observed_pairs, task_ids, agent_ids, novelty_label_map, split_name, raw_probs, calibrated_probs, y_true, mask_t):
    valid = mask_t.detach().cpu().numpy().astype(bool)
    rows = []
    for i_idx, j_idx in np.argwhere(valid):
        item_id = str(task_ids[int(j_idx)])
        rows.append({
            "seed": int(seed),
            "train_retention": float(retention),
            "observed_train_pairs": int(observed_pairs),
            "method": method,
            "calibration": calibration_name,
            "split": split_name,
            "agent_id": str(agent_ids[int(i_idx)]),
            "item_id": item_id,
            "benchmark": item_id.split(".")[0] if "." in item_id else item_id,
            "novelty_bin": novelty_label_map.get(item_id, "unknown"),
            "true_label": float(y_true[int(i_idx), int(j_idx)]),
            "raw_probability": float(raw_probs[int(i_idx), int(j_idx)]),
            "raw_score": float(logit_np(raw_probs[int(i_idx), int(j_idx)])),
            "calibrated_probability": float(calibrated_probs[int(i_idx), int(j_idx)]),
        })
    return rows


def reliability_curve(y_true, y_prob, n_bins=10):
    y_true_bin = (np.asarray(y_true) > 0.5).astype(float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    order = np.argsort(y_prob)
    y_prob = y_prob[order]
    y_true_bin = y_true_bin[order]
    bins = np.array_split(np.arange(y_prob.size), min(n_bins, y_prob.size))
    xs, ys = [], []
    for idx in bins:
        if idx.size == 0:
            continue
        xs.append(float(np.mean(y_prob[idx])))
        ys.append(float(np.mean(y_true_bin[idx])))
    return np.array(xs), np.array(ys)


def save_reliability_plots(pred_df: pd.DataFrame, out_dir: Path):
    if pred_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    settings = [("overall", pred_df), ("outlier", pred_df[pred_df["novelty_bin"] == "outlier"])]
    methods = ["ARAF", "kNN", "Rasch", "MIRT"]

    for ax, (title, subdf) in zip(axes, settings):
        for method, color in zip(methods, ["steelblue", "orange", "tan", "gray"]):
            for calibration, style in [("raw", "-"), ("sigmoid", "--")]:
                cur = subdf[(subdf["method"] == method) & (subdf["calibration"] == calibration) & (subdf["split"] == "test")]
                if cur.empty:
                    continue
                xs, ys = reliability_curve(cur["true_label"].to_numpy(), cur["calibrated_probability"].to_numpy())
                label = f"{method} ({calibration})"
                ax.plot(xs, ys, linestyle=style, color=color, linewidth=1.3, label=label)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, alpha=0.4)
        ax.set_title(f"Reliability: {title}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical accuracy")
        ax.grid(linestyle=":", alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=True)
    out_path = out_dir / "reliability_overall.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def print_summary_tables(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    overall = metrics_df[(metrics_df["split"] == "test") & (metrics_df["novelty_bin"] == "overall")]
    summary = overall.groupby(["method", "calibration"], as_index=False).agg(
        auc=("auc", "mean"),
        rmse=("rmse", "mean"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
        ece=("ece", "mean"),
    )
    raw = summary[summary["calibration"] == "raw"].sort_values("auc", ascending=False)
    post = summary[summary["calibration"] != "raw"].sort_values(["method", "calibration"])
    print("\n=== Overall Raw Metrics ===")
    print(raw.to_string(index=False))
    print("\n=== Overall Calibrated Metrics ===")
    print(post.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Standalone calibration study for unseen-item prediction.")
    parser.add_argument("--model-type", type=str, default="beta", choices=["beta", "bernoulli"])
    parser.add_argument("--seeds", type=str, default="0-49")
    parser.add_argument("--retentions", type=str, default="1.0,0.5,0.25,0.1,0.05")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--beta-phi", type=float, default=ai.BETA_PHI)
    parser.add_argument("--force-pre-revision", type=str, default=None)
    parser.add_argument("--force-j-percentage", type=float, default=None)
    parser.add_argument("--force-n-samples", type=int, default=None)
    parser.add_argument("--force-train-retention", type=float, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = ai.parse_explicit_n_samples(args.seeds)
    if seeds is None:
        seeds = ai.parse_n_samples(args.seeds, 50)
    retentions = [float(x) for x in args.retentions.split(",") if x.strip()]

    araf_df = scan_araf_candidates(RESULT_ROOT, model_type=args.model_type)
    baseline_df = scan_baseline_candidates(RESULT_ROOT, model_type=args.model_type)
    if araf_df.empty or baseline_df.empty:
        raise RuntimeError("Could not discover enough candidate configurations from existing result files.")

    global_best = select_global_bests(araf_df, baseline_df)
    global_best.to_csv(out_dir / "global_best_configs.csv", index=False)

    if any(v is not None for v in [args.force_pre_revision, args.force_j_percentage, args.force_n_samples, args.force_train_retention]):
        common_condition = Condition(
            model_type=args.model_type,
            pre_revision=normalize_pre_revision(args.force_pre_revision if args.force_pre_revision is not None else "none"),
            j_percentage=normalize_j_percentage(args.force_j_percentage if args.force_j_percentage is not None else 1.0),
            n_samples=int(args.force_n_samples if args.force_n_samples is not None else 54),
            train_retention=float(args.force_train_retention if args.force_train_retention is not None else 1.0),
        )
        common_configs = select_configs_for_condition(araf_df, baseline_df, common_condition)
    else:
        common_condition, common_configs = select_best_common_condition(araf_df, baseline_df)
    common_configs.to_csv(out_dir / "common_condition_best_configs.csv", index=False)

    print("=== Global Best Configurations Across Available Runs ===")
    print(global_best[["method", "mean_auc", "mean_rmse", "pre_revision", "j_percentage", "n_samples", "train_retention", "embedding_type", "baseline_embedding_type", "knn_k", "lambda_tau", "selected_mirt_dim"]].fillna("").to_string(index=False))
    print("\n=== Fair Common Condition Used For Calibration ===")
    print(f"model_type={common_condition.model_type}, pre_revision={common_condition.pre_revision}, "
          f"j_percentage={common_condition.j_percentage}, n_samples={common_condition.n_samples}, "
          f"train_retention={common_condition.train_retention}")
    print(common_configs[["method", "mean_auc", "mean_rmse", "embedding_type", "baseline_embedding_type", "knn_k", "lambda_tau", "selected_mirt_dim"]].fillna("").to_string(index=False))

    prediction_rows = []
    metric_rows = []
    data_cache: Dict[Tuple[str, str], Tuple] = {}

    for seed in seeds:
        set_all_seeds(int(seed))
        for retention in retentions:
            method_configs = {
                method: select_config_for_retention(method, common_configs, araf_df, baseline_df, common_condition, retention)
                for method in ["ARAF", "kNN", "Rasch", "MIRT"]
            }

            # Use raw embeddings as the canonical novelty space.
            raw_cache_key = (common_condition.pre_revision, "raw")
            if raw_cache_key not in data_cache:
                data_cache[raw_cache_key] = ai.load_data(embedding_type="raw", pre_revision=common_condition.pre_revision)
            raw_all_dfs, raw_shared_idx, raw_emb_map, raw_emb_type = data_cache[raw_cache_key]
            novelty_data = prepare_calibration_data(
                raw_all_dfs, raw_shared_idx, raw_emb_map, raw_emb_type,
                j_percentage=common_condition.j_percentage, seed=int(seed),
                cal_fraction=CAL_FRACTION, test_fraction=TEST_FRACTION,
            )
            novelty_masks, _, novelty_label_map = build_novelty_masks(
                novelty_data["x_j"], novelty_data["train_idx"], novelty_data["task_ids"]
            )

            for method, config in method_configs.items():
                probs, cal_data, y_train, train_mask_t, observed_pairs = run_method_predictions(
                    method, config, int(seed), retention, data_cache, beta_phi=args.beta_phi
                )

                cal_y, cal_prob = masked_arrays(probs, cal_data["y_oracle"], cal_data["cal_mask_t"])
                calibrators = fit_calibrators(method, cal_prob, cal_y)

                raw_probs_np = probs.detach().cpu().numpy()
                raw_scores_np = logit_np(raw_probs_np)
                y_oracle_np = cal_data["y_oracle"].detach().cpu().numpy()

                for calibrator in calibrators:
                    cal_pred = calibrator.predict(raw_scores_np, prob=raw_probs_np)
                    cal_pred_t = torch.tensor(cal_pred, dtype=torch.float32, device=ai.device)

                    prediction_rows.extend(build_prediction_rows(
                        method, calibrator.name, int(seed), retention, observed_pairs,
                        cal_data["task_ids"], cal_data["agent_ids"], novelty_label_map,
                        "calibration", raw_probs_np, cal_pred, y_oracle_np, cal_data["cal_mask_t"]
                    ))
                    prediction_rows.extend(build_prediction_rows(
                        method, calibrator.name, int(seed), retention, observed_pairs,
                        cal_data["task_ids"], cal_data["agent_ids"], novelty_label_map,
                        "test", raw_probs_np, cal_pred, y_oracle_np, cal_data["test_mask_t"]
                    ))

                    metric_rows.append(metric_row(
                        method, calibrator.name, int(seed), retention, observed_pairs, "overall", "test",
                        cal_pred_t, cal_data["y_oracle"], cal_data["test_mask_t"]
                    ))
                    metric_rows.append(metric_row(
                        method, calibrator.name, int(seed), retention, observed_pairs, "overall", "calibration",
                        cal_pred_t, cal_data["y_oracle"], cal_data["cal_mask_t"]
                    ))

                    for novelty_bin, item_mask in novelty_masks.items():
                        test_mask_t = cal_data["test_mask_t"].clone()
                        cal_mask_t = cal_data["cal_mask_t"].clone()
                        test_mask_t[:, ~item_mask] = False
                        cal_mask_t[:, ~item_mask] = False
                        metric_rows.append(metric_row(
                            method, calibrator.name, int(seed), retention, observed_pairs, novelty_bin, "test",
                            cal_pred_t, cal_data["y_oracle"], test_mask_t
                        ))
                        metric_rows.append(metric_row(
                            method, calibrator.name, int(seed), retention, observed_pairs, novelty_bin, "calibration",
                            cal_pred_t, cal_data["y_oracle"], cal_mask_t
                        ))

    pred_df = pd.DataFrame(prediction_rows)
    metrics_df = pd.DataFrame(metric_rows)
    pred_df.to_csv(out_dir / "calibration_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "calibration_metrics.csv", index=False)
    save_reliability_plots(pred_df, fig_dir)
    print_summary_tables(metrics_df)


if __name__ == "__main__":
    main()
