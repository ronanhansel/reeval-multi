#!/usr/bin/env python3
"""
Diagnose pre- vs post-revision response-matrix stability.

This script follows the data-loading conventions from ``amortized_irt.py``:

- Pre-revision uses the aggregated matrix from ``_load_pre_revision_response_matrix``.
- Post-revision uses the averaged beta/oracle matrix from
  ``_load_post_revision_response_matrices`` and binarizes it with ``> 0.5``.
- Matched-size pre-revision sampling mirrors the balanced benchmark-wise row
  sampling used by the amortized IRT pre-revision path.

The main requested outputs are:
1. Full-matrix comparison of average variance (pre vs post).
2. Matched-size repeated sampling of pre-revision rows to the post row count,
   repeated multiple times, with standard errors.

Additional summaries are included to help compare stability/quality:
- Mean score / pass rate
- Per-agent average variance
- Fraction of zero-variance (consensus) items
- Mean binary item entropy
- Benchmark-level item-variance breakdown
- Optional sensitivity check where pre is also binarized
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from model.amortized_irt import (
    RANDOM_SEED,
    _load_post_revision_response_matrices,
    _load_pre_revision_response_matrix,
)


BENCHMARKS = [
    "colbench_backend_programming",
    "corebench_hard",
    "scicode",
    "scienceagentbench",
]


@dataclass
class MatrixBundle:
    name: str
    frame: pd.DataFrame

    @property
    def values(self) -> np.ndarray:
        return self.frame.to_numpy(dtype=float)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.frame.shape


def binarize_observed(frame: pd.DataFrame) -> pd.DataFrame:
    """Threshold observed entries at 0.5 while preserving missing values."""
    binary = pd.DataFrame(
        np.where(frame.isna(), np.nan, (frame.to_numpy(dtype=float) > 0.5).astype(float)),
        index=frame.index,
        columns=frame.columns,
    )
    return binary


def compute_post_binary_matrix() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return the post beta/oracle matrix, its binarized version, and the aligned post stack."""
    post_dfs, post_indices = _load_post_revision_response_matrices()
    post_columns = sorted(list(set().union(*[df.columns for df in post_dfs])))
    post_filtered = [df.reindex(index=post_indices, columns=post_columns) for df in post_dfs]
    post_stack = np.array([df.values for df in post_filtered], dtype=float)
    post_beta = pd.DataFrame(
        np.nanmean(post_stack, axis=0),
        index=post_indices,
        columns=post_columns,
    )
    post_binary = binarize_observed(post_beta)
    return post_beta, post_binary, post_stack


def align_columns(
    pre_df: pd.DataFrame,
    post_beta_df: pd.DataFrame,
    post_binary_df: pd.DataFrame,
    column_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align matrices to a common set of columns."""
    pre_cols = set(pre_df.columns)
    post_cols = set(post_beta_df.columns)
    if column_mode == "intersection":
        columns = sorted(pre_cols & post_cols)
    else:
        columns = sorted(pre_cols | post_cols)

    if not columns:
        raise ValueError("No shared item columns were found between pre and post matrices.")

    return (
        pre_df.reindex(columns=columns),
        post_beta_df.reindex(columns=columns),
        post_binary_df.reindex(columns=columns),
    )


def to_binary(frame: pd.DataFrame) -> pd.DataFrame:
    """Binarize a score matrix with the same thresholding convention as post."""
    return binarize_observed(frame)


def nanvar_axis(values: np.ndarray, axis: int) -> np.ndarray:
    """Variance with ddof=1, returning NaN when fewer than 2 observations exist."""
    counts = np.sum(~np.isnan(values), axis=axis)
    out_len = values.shape[1 - axis]
    variances = np.full(out_len, np.nan, dtype=float)
    valid = counts >= 2
    if axis == 0:
        if np.any(valid):
            variances[valid] = np.nanvar(values[:, valid], axis=0, ddof=1)
    else:
        if np.any(valid):
            variances[valid] = np.nanvar(values[valid, :], axis=1, ddof=1)
    return variances


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def safe_se(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def binary_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def ci95(mean: float, se: float) -> Tuple[float, float]:
    if np.isnan(mean) or np.isnan(se):
        return float("nan"), float("nan")
    return float(mean - 1.96 * se), float(mean + 1.96 * se)


def relative_change_pct(reference: float, new_value: float) -> float:
    if np.isnan(reference) or np.isnan(new_value) or abs(reference) < 1e-12:
        return float("nan")
    return float(100.0 * (new_value - reference) / reference)


def relative_reduction_pct(reference: float, new_value: float) -> float:
    if np.isnan(reference) or np.isnan(new_value) or abs(reference) < 1e-12:
        return float("nan")
    return float(100.0 * (reference - new_value) / reference)


def z_score_from_sampling(reference_mean: float, reference_se: float, observed: float) -> float:
    if np.isnan(reference_mean) or np.isnan(reference_se) or np.isnan(observed) or reference_se <= 0:
        return float("nan")
    return float((observed - reference_mean) / reference_se)


def empirical_two_sided_pvalue(sample_values: Sequence[float], reference: float) -> float:
    arr = np.asarray(sample_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0 or np.isnan(reference):
        return float("nan")
    lower = np.mean(arr <= reference)
    upper = np.mean(arr >= reference)
    return float(2.0 * min(lower, upper))


def nanquantiles(values: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {f"q{int(q * 100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q * 100)}"] = float(np.nanquantile(arr, q))
    return out


def item_variances(frame: pd.DataFrame) -> np.ndarray:
    return nanvar_axis(frame.to_numpy(dtype=float), axis=0)


def benchmark_item_variance_macro(frame: pd.DataFrame) -> float:
    per_benchmark = benchmark_item_variance(frame)
    if not per_benchmark:
        return float("nan")
    return float(np.nanmean(list(per_benchmark.values())))


def observed_count_per_item(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    return np.sum(~np.isnan(values), axis=0).astype(float)


def bootstrap_item_metric_gap(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    metric_fn,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Bootstrap over aligned item columns to quantify uncertainty in item-level gaps.
    This is useful because the comparison target is 'matrix/item behavior', not paired agents.
    """
    if list(pre_df.columns) != list(post_df.columns):
        raise ValueError("pre_df and post_df must have identical aligned columns for item bootstrap.")

    n_cols = pre_df.shape[1]
    if n_cols == 0:
        return {
            "observed_gap": float("nan"),
            "bootstrap_mean_gap": float("nan"),
            "bootstrap_se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    observed_gap = float(metric_fn(post_df) - metric_fn(pre_df))
    gaps = []
    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n_cols, size=n_cols)
        cols = [pre_df.columns[i] for i in boot_idx]
        pre_boot = pre_df.loc[:, cols]
        post_boot = post_df.loc[:, cols]
        gaps.append(float(metric_fn(post_boot) - metric_fn(pre_boot)))

    gaps_arr = np.asarray(gaps, dtype=float)
    mean_gap = float(np.nanmean(gaps_arr))
    se_gap = safe_se(gaps_arr)
    ci_lower, ci_upper = ci95(mean_gap, se_gap)
    return {
        "observed_gap": observed_gap,
        "bootstrap_mean_gap": mean_gap,
        "bootstrap_se": se_gap,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def permutation_test_item_metric_gap(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    metric_fn,
    n_permutations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Permute item labels between pre and post at the column level.
    This tests whether the observed item-level gap is unusually large
    relative to a null where 'pre/post assignment' of aligned columns does not matter.
    """
    if list(pre_df.columns) != list(post_df.columns):
        raise ValueError("pre_df and post_df must have identical aligned columns for permutation test.")

    observed_gap = float(metric_fn(post_df) - metric_fn(pre_df))
    n_cols = pre_df.shape[1]
    null_gaps = []

    # Column swapping requires equal row counts. For permutation-null construction
    # only, downsample each matrix to a shared row count when needed.
    if pre_df.shape[0] != post_df.shape[0]:
        n_rows = min(pre_df.shape[0], post_df.shape[0])
        pre_index = rng.choice(pre_df.index.to_numpy(), size=n_rows, replace=False)
        post_index = rng.choice(post_df.index.to_numpy(), size=n_rows, replace=False)
        pre_perm_source = pre_df.loc[pre_index]
        post_perm_source = post_df.loc[post_index]
    else:
        pre_perm_source = pre_df
        post_perm_source = post_df

    pre_values = pre_perm_source.to_numpy(dtype=float)
    post_values = post_perm_source.to_numpy(dtype=float)

    for _ in range(n_permutations):
        swap_mask = rng.random(n_cols) < 0.5
        perm_pre = pre_values.copy()
        perm_post = post_values.copy()

        perm_pre[:, swap_mask] = post_values[:, swap_mask]
        perm_post[:, swap_mask] = pre_values[:, swap_mask]

        perm_pre_df = pd.DataFrame(
            perm_pre,
            index=pre_perm_source.index,
            columns=pre_perm_source.columns,
        )
        perm_post_df = pd.DataFrame(
            perm_post,
            index=post_perm_source.index,
            columns=post_perm_source.columns,
        )
        null_gaps.append(float(metric_fn(perm_post_df) - metric_fn(perm_pre_df)))

    null_arr = np.asarray(null_gaps, dtype=float)
    p_two_sided = float(np.mean(np.abs(null_arr) >= abs(observed_gap)))
    return {
        "observed_gap": observed_gap,
        "null_mean_gap": float(np.nanmean(null_arr)),
        "null_se_gap": safe_se(null_arr),
        "p_value_two_sided": p_two_sided,
    }

def summarize_matrix(frame: pd.DataFrame, *, entropy_on_binary: bool = False) -> Dict[str, float]:
    values = frame.to_numpy(dtype=float)
    item_var = nanvar_axis(values, axis=0)
    agent_var = nanvar_axis(values, axis=1)
    observed_mask = ~np.isnan(values)
    item_means = np.nanmean(values, axis=0)
    item_obs = np.sum(~np.isnan(values), axis=0)

    summary = {
        "rows": int(frame.shape[0]),
        "cols": int(frame.shape[1]),
        "observed_fraction": float(observed_mask.mean()),
        "overall_mean": safe_mean(values),
        "avg_item_variance": safe_mean(item_var),
        "avg_agent_variance": safe_mean(agent_var),
        "median_item_variance": float(np.nanmedian(item_var)),
        "median_agent_variance": float(np.nanmedian(agent_var)),
        "zero_variance_item_fraction": float(np.nanmean(np.isclose(item_var, 0.0, atol=1e-12))),
        "mean_item_score": safe_mean(item_means),
        "macro_benchmark_avg_item_variance": benchmark_item_variance_macro(frame),
        "mean_observed_per_item": safe_mean(item_obs.astype(float)),
    }

    summary.update(nanquantiles(item_var, [0.10, 0.25, 0.50, 0.75, 0.90]))
    summary.update({
        f"agent_var_{k}": v for k, v in nanquantiles(agent_var, [0.10, 0.25, 0.50, 0.75, 0.90]).items()
    })

    if entropy_on_binary:
        summary["mean_item_entropy"] = safe_mean(binary_entropy(item_means[~np.isnan(item_means)]))

    return summary


def summarize_post_repeatability(post_stack: np.ndarray) -> Dict[str, float]:
    """Summarize how often repeated post runs disagree before hard thresholding."""
    if post_stack.size == 0:
        return {
            "observed_cell_count": 0,
            "mean_within_cell_variance": float("nan"),
            "fraction_cells_with_any_disagreement": float("nan"),
            "fraction_majority_fail_with_some_success": float("nan"),
            "fraction_majority_pass_with_some_failure": float("nan"),
            "fraction_unanimous_fail": float("nan"),
            "fraction_unanimous_pass": float("nan"),
            "fraction_exactly_at_threshold": float("nan"),
        }

    post_mean = np.nanmean(post_stack, axis=0)
    observed_mask = ~np.isnan(post_mean)
    observed_means = post_mean[observed_mask]
    within_cell_var = np.nanvar(post_stack, axis=0)
    observed_var = within_cell_var[observed_mask]

    return {
        "observed_cell_count": int(observed_means.size),
        "mean_within_cell_variance": safe_mean(observed_var),
        "fraction_cells_with_any_disagreement": float(np.mean((observed_means > 0.0) & (observed_means < 1.0))),
        "fraction_majority_fail_with_some_success": float(np.mean((observed_means > 0.0) & (observed_means <= 0.5))),
        "fraction_majority_pass_with_some_failure": float(np.mean((observed_means > 0.5) & (observed_means < 1.0))),
        "fraction_unanimous_fail": float(np.mean(np.isclose(observed_means, 0.0))),
        "fraction_unanimous_pass": float(np.mean(np.isclose(observed_means, 1.0))),
        "fraction_exactly_at_threshold": float(np.mean(np.isclose(observed_means, 0.5))),
    }


def benchmark_item_variance(frame: pd.DataFrame) -> Dict[str, float]:
    """Average item variance within each benchmark's columns."""
    out: Dict[str, float] = {}
    for benchmark in BENCHMARKS:
        cols = [col for col in frame.columns if str(col).startswith(f"{benchmark}.")]
        if not cols:
            continue
        values = frame[cols].to_numpy(dtype=float)
        out[benchmark] = safe_mean(nanvar_axis(values, axis=0))
    return out


def retained_items_per_benchmark(columns: Sequence[str]) -> Dict[str, int]:
    out = {benchmark: 0 for benchmark in BENCHMARKS}
    for col in columns:
        for benchmark in BENCHMARKS:
            if str(col).startswith(f"{benchmark}."):
                out[benchmark] += 1
                break
    return out


def difficulty_restricted_metrics(
    sample_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    include_entropy: bool,
    low: float,
    high: float,
    min_items_per_benchmark: int,
) -> Dict[str, object]:
    if list(sample_df.columns) != list(post_df.columns):
        raise ValueError("sample_df and post_df must have aligned identical columns.")

    sample_means = np.nanmean(sample_df.to_numpy(dtype=float), axis=0)
    post_means = np.nanmean(post_df.to_numpy(dtype=float), axis=0)
    keep_mask = (
        (sample_means >= low)
        & (sample_means <= high)
        & (post_means >= low)
        & (post_means <= high)
    )

    kept_cols = [col for col, keep in zip(sample_df.columns, keep_mask) if keep]
    sample_kept = sample_df.loc[:, kept_cols]
    post_kept = post_df.loc[:, kept_cols]

    metric_keys = [
        "avg_item_variance",
        "macro_benchmark_avg_item_variance",
        "zero_variance_item_fraction",
    ]
    if include_entropy:
        metric_keys.append("mean_item_entropy")

    sample_summary = summarize_matrix(sample_kept, entropy_on_binary=include_entropy)
    post_summary = summarize_matrix(post_kept, entropy_on_binary=include_entropy)

    per_metric: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        sample_val = float(sample_summary.get(key, np.nan))
        post_val = float(post_summary.get(key, np.nan))
        per_metric[key] = {
            "sample": sample_val,
            "post": post_val,
            "delta_post_minus_sample": float(post_val - sample_val),
            "relative_reduction_pct": relative_reduction_pct(sample_val, post_val),
        }

    retained_counts = retained_items_per_benchmark(kept_cols)
    too_few = {
        benchmark: int(count)
        for benchmark, count in retained_counts.items()
        if count < min_items_per_benchmark
    }

    return {
        "retained_item_count": int(len(kept_cols)),
        "retained_item_fraction": float(len(kept_cols) / sample_df.shape[1]) if sample_df.shape[1] > 0 else float("nan"),
        "retained_items_per_benchmark": retained_counts,
        "too_few_items_per_benchmark": too_few,
        "min_items_per_benchmark": int(min_items_per_benchmark),
        "per_metric": per_metric,
    }


def allocation_like_amortized(n_total: int) -> Dict[str, int]:
    """Balanced per-benchmark row allocation used in amortized_irt pre subsampling."""
    base = n_total // len(BENCHMARKS)
    remainder = n_total % len(BENCHMARKS)
    return {
        benchmark: base + (1 if i < remainder else 0)
        for i, benchmark in enumerate(BENCHMARKS)
    }


def sample_pre_rows_like_amortized(
    pre_df: pd.DataFrame,
    n_total: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample pre rows benchmark-wise, mirroring amortized_irt pre sampling."""
    allocation = allocation_like_amortized(n_total)
    sampled_rows: List[str] = []
    for benchmark in BENCHMARKS:
        n_keep = allocation[benchmark]
        if n_keep == 0:
            continue
        benchmark_rows = [idx for idx in pre_df.index if str(idx).startswith(f"{benchmark}.")]
        if not benchmark_rows:
            continue
        if len(benchmark_rows) > n_keep:
            sampled = rng.choice(benchmark_rows, size=n_keep, replace=False).tolist()
        else:
            sampled = list(benchmark_rows)
        sampled_rows.extend(sampled)
    return pre_df.loc[sampled_rows]


def agent_strength_series(frame: pd.DataFrame) -> pd.Series:
    """Simple ability proxy: mean observed score per row."""
    return pd.Series(
        np.nanmean(frame.to_numpy(dtype=float), axis=1),
        index=frame.index,
        dtype=float,
    )


def columns_for_benchmark(columns: Iterable[str], benchmark: str) -> List[str]:
    return [col for col in columns if str(col).startswith(f"{benchmark}.")]


def row_strength_within_benchmark(frame: pd.DataFrame, benchmark: str) -> pd.Series:
    cols = columns_for_benchmark(frame.columns, benchmark)
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame.loc[:, cols].to_numpy(dtype=float)
    return pd.Series(np.nanmean(values, axis=1), index=frame.index, dtype=float)


def benchmark_row_ids(frame: pd.DataFrame, benchmark: str) -> List[object]:
    return [idx for idx in frame.index if row_group_from_index(idx) == benchmark]


def row_group_from_index(row_id: object) -> str:
    row_str = str(row_id)

    # Normalize known benchmark name variants (e.g., dot/underscore separators)
    # back to canonical BENCHMARKS labels.
    for benchmark in BENCHMARKS:
        aliases = {benchmark, benchmark.replace("_", ".")}
        if benchmark == "colbench_backend_programming":
            aliases.add("colbench")
        if benchmark == "corebench_hard":
            aliases.add("corebench")
        for alias in aliases:
            if row_str.startswith(f"{alias}.") or row_str.startswith(f"{alias}_"):
                return benchmark

    if "." in row_str:
        return row_str.split(".", 1)[0]
    return row_str


def row_group_ids(frame: pd.DataFrame, group: str) -> List[object]:
    return [idx for idx in frame.index if row_group_from_index(idx) == group]


def row_group_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for idx in frame.index:
        group = row_group_from_index(idx)
        counts[group] = counts.get(group, 0) + 1
    return counts


def post_benchmark_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = {benchmark: 0 for benchmark in BENCHMARKS}
    for idx in frame.index:
        group = row_group_from_index(idx)
        if group in counts:
            counts[group] += 1
    return counts


def sample_pre_rows_benchmark_constrained_ability_matched(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    strength_jitter: float = 1e-8,
) -> pd.DataFrame:
    """
    Sample pre rows to match post row-count and post benchmark composition,
    with nearest-neighbor matching on benchmark-local row strength.
    """
    sampled_rows: List[object] = []
    post_group_counts = row_group_counts(post_df)

    for group, n_required in post_group_counts.items():
        if n_required == 0:
            continue

        post_rows = row_group_ids(post_df, group)
        pre_rows = row_group_ids(pre_df, group)
        if len(post_rows) < n_required:
            raise ValueError(f"Post rows for group {group} are fewer than required count.")
        if len(pre_rows) < n_required:
            raise ValueError(
                f"Insufficient pre rows for group {group}: have {len(pre_rows)}, need {n_required}."
            )

        if group in BENCHMARKS:
            post_strength = row_strength_within_benchmark(post_df.loc[post_rows], group)
            pre_strength = row_strength_within_benchmark(pre_df.loc[pre_rows], group)
        else:
            post_strength = agent_strength_series(post_df.loc[post_rows])
            pre_strength = agent_strength_series(pre_df.loc[pre_rows])

        post_order = rng.permutation(post_rows)
        available_ids = list(pre_rows)

        for post_id in post_order:
            if not available_ids:
                break

            p_strength = float(post_strength.loc[post_id])
            available_strength = pre_strength.loc[available_ids].to_numpy(dtype=float)

            if np.isnan(p_strength):
                distances = np.where(np.isnan(available_strength), 0.0, np.inf)
            else:
                distances = np.abs(available_strength - p_strength)
                distances[np.isnan(distances)] = np.inf

            finite_mask = np.isfinite(distances)
            if not np.any(finite_mask):
                chosen_pos = int(rng.integers(0, len(available_ids)))
            else:
                finite_distances = distances[finite_mask]
                jitter = rng.uniform(0.0, strength_jitter, size=finite_distances.shape[0])
                finite_positions = np.where(finite_mask)[0]
                chosen_pos = int(finite_positions[int(np.argmin(finite_distances + jitter))])

            chosen_id = available_ids[chosen_pos]
            sampled_rows.append(chosen_id)
            del available_ids[chosen_pos]

        if len([row for row in sampled_rows if row_group_from_index(row) == group]) != n_required:
            raise ValueError(
                f"Failed to sample required rows for group {group}: "
                f"expected {n_required}."
            )

    return pre_df.loc[sampled_rows]


def empirical_percentile(sample_values: Sequence[float], reference: float) -> float:
    arr = np.asarray(sample_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0 or np.isnan(reference):
        return float("nan")
    return float(np.mean(arr <= reference))


def sampling_match_diagnostics(sample_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict[str, float]:
    sample_strength = agent_strength_series(sample_df).to_numpy(dtype=float)
    post_strength = agent_strength_series(post_df).to_numpy(dtype=float)

    return {
        "sample_mean_strength": float(np.nanmean(sample_strength)),
        "post_mean_strength": float(np.nanmean(post_strength)),
        "delta_mean_strength": float(np.nanmean(sample_strength) - np.nanmean(post_strength)),
        "sample_sd_strength": float(np.nanstd(sample_strength)),
        "post_sd_strength": float(np.nanstd(post_strength)),
        "delta_sd_strength": float(np.nanstd(sample_strength) - np.nanstd(post_strength)),
    }


def sampling_match_diagnostics_by_benchmark(sample_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for benchmark in BENCHMARKS:
        sample_rows = benchmark_row_ids(sample_df, benchmark)
        post_rows = benchmark_row_ids(post_df, benchmark)
        if not post_rows:
            continue

        sample_strength = row_strength_within_benchmark(sample_df.loc[sample_rows], benchmark).to_numpy(dtype=float)
        post_strength = row_strength_within_benchmark(post_df.loc[post_rows], benchmark).to_numpy(dtype=float)
        out[benchmark] = {
            "sample_mean_strength": float(np.nanmean(sample_strength)),
            "post_mean_strength": float(np.nanmean(post_strength)),
            "delta_mean_strength": float(np.nanmean(sample_strength) - np.nanmean(post_strength)),
            "sample_sd_strength": float(np.nanstd(sample_strength)),
            "post_sd_strength": float(np.nanstd(post_strength)),
            "delta_sd_strength": float(np.nanstd(sample_strength) - np.nanstd(post_strength)),
        }
    return out


def validate_sampled_pre_matrix(
    sample_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    observed_per_item_ratio_floor: float = 0.9,
    enforce_post_benchmark_counts: bool = True,
) -> Tuple[bool, str, Dict[str, object]]:
    sample_counts = post_benchmark_counts(sample_df)
    post_counts = post_benchmark_counts(post_df)
    sample_group_counts = row_group_counts(sample_df)
    post_group_counts = row_group_counts(post_df)

    sample_summary = summarize_matrix(sample_df, entropy_on_binary=False)
    post_summary = summarize_matrix(post_df, entropy_on_binary=False)
    sample_mean_obs = float(sample_summary["mean_observed_per_item"])
    post_mean_obs = float(post_summary["mean_observed_per_item"])
    observed_ratio = float(sample_mean_obs / post_mean_obs) if post_mean_obs > 0 else float("nan")

    required_benchmarks = [benchmark for benchmark, count in post_counts.items() if count > 0]
    sample_variance_keys = set(benchmark_item_variance(sample_df).keys())
    missing_benchmarks = [benchmark for benchmark in required_benchmarks if benchmark not in sample_variance_keys]

    diagnostics: Dict[str, object] = {
        "sample_benchmark_counts": sample_counts,
        "post_benchmark_counts": post_counts,
        "sample_group_counts": sample_group_counts,
        "post_group_counts": post_group_counts,
        "sample_mean_observed_per_item": sample_mean_obs,
        "post_mean_observed_per_item": post_mean_obs,
        "observed_per_item_ratio": observed_ratio,
        "required_benchmarks": required_benchmarks,
        "benchmark_variance_keys": sorted(list(sample_variance_keys)),
    }

    if sample_df.shape[0] != post_df.shape[0]:
        return False, "sample/post row count mismatch", diagnostics
    if enforce_post_benchmark_counts and sample_group_counts != post_group_counts:
        return False, "sample/post group counts mismatch", diagnostics
    if np.isnan(observed_ratio) or observed_ratio < observed_per_item_ratio_floor:
        return False, "sample observed coverage too low", diagnostics
    if missing_benchmarks:
        diagnostics["missing_benchmarks"] = missing_benchmarks
        return False, "missing benchmarks in benchmark-wise variance", diagnostics
    return True, "ok", diagnostics


def repeated_sampling_report(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    n_repeats: int,
    rng_seed: int,
    include_entropy: bool,
    *,
    sampler_name: str = "benchmark_balanced",
    max_attempts: int = 100,
    enable_difficulty_restricted: bool = False,
    difficulty_low: float = 0.2,
    difficulty_high: float = 0.8,
    difficulty_min_items_per_benchmark: int = 5,
    observed_per_item_ratio_floor: float = 0.9,
) -> Dict[str, object]:
    """Run repeated matched-size pre sampling and compare with post."""
    target_rows = post_df.shape[0]
    rng = np.random.default_rng(rng_seed)
    post_summary = summarize_matrix(post_df, entropy_on_binary=include_entropy)
    summaries: List[Dict[str, float]] = []
    benchmark_rows: List[Dict[str, float]] = []
    match_rows: List[Dict[str, float]] = []
    match_rows_by_benchmark: Dict[str, List[Dict[str, float]]] = {benchmark: [] for benchmark in BENCHMARKS}
    sample_validity_rows: List[Dict[str, object]] = []
    difficulty_rows: List[Dict[str, object]] = []
    invalid_attempts_total = 0

    for _ in range(n_repeats):
        sample_df = None
        sample_valid = False
        last_reason = ""
        last_diag: Dict[str, object] = {}

        for attempt in range(max_attempts):
            if sampler_name == "benchmark_balanced":
                candidate = sample_pre_rows_like_amortized(pre_df, target_rows, rng)
            elif sampler_name == "benchmark_constrained_ability_matched":
                candidate = sample_pre_rows_benchmark_constrained_ability_matched(pre_df, post_df, rng)
            else:
                raise ValueError(f"Unknown sampler_name={sampler_name!r}")

            sample_valid, last_reason, last_diag = validate_sampled_pre_matrix(
                candidate,
                post_df,
                observed_per_item_ratio_floor=observed_per_item_ratio_floor,
                enforce_post_benchmark_counts=(sampler_name == "benchmark_constrained_ability_matched"),
            )
            if sample_valid:
                sample_df = candidate
                break
            invalid_attempts_total += 1

        if not sample_valid or sample_df is None:
            raise ValueError(
                f"Unable to produce a valid sampled pre matrix for sampler={sampler_name!r} "
                f"after {max_attempts} attempts. Last reason: {last_reason}. Diagnostics: {last_diag}"
            )

        summaries.append(summarize_matrix(sample_df, entropy_on_binary=include_entropy))
        benchmark_rows.append(benchmark_item_variance(sample_df))
        match_rows.append(sampling_match_diagnostics(sample_df, post_df))
        per_benchmark_diag = sampling_match_diagnostics_by_benchmark(sample_df, post_df)
        for benchmark in BENCHMARKS:
            if benchmark in per_benchmark_diag:
                match_rows_by_benchmark[benchmark].append(per_benchmark_diag[benchmark])
        sample_validity_rows.append(last_diag)
        if enable_difficulty_restricted:
            difficulty_rows.append(
                difficulty_restricted_metrics(
                    sample_df,
                    post_df,
                    include_entropy=include_entropy,
                    low=difficulty_low,
                    high=difficulty_high,
                    min_items_per_benchmark=difficulty_min_items_per_benchmark,
                )
            )

    metric_keys = [
        "overall_mean",
        "avg_item_variance",
        "avg_agent_variance",
        "zero_variance_item_fraction",
        "macro_benchmark_avg_item_variance",
        "mean_observed_per_item",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
    ]
    if include_entropy:
        metric_keys.append("mean_item_entropy")

    aggregate: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [row.get(key, np.nan) for row in summaries]
        mean_val = float(np.nanmean(values))
        se_val = safe_se(values)
        ci_lower, ci_upper = ci95(mean_val, se_val)
        post_val = float(post_summary.get(key, np.nan))
        p_two_sided = empirical_two_sided_pvalue(values, post_val)
        if not np.isnan(p_two_sided):
            p_two_sided = min(1.0, max(0.0, p_two_sided))

        aggregate[key] = {
            "mean": mean_val,
            "se": se_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "post": post_val,
            "delta_post_minus_sampled_pre": float(post_val - mean_val),
            "relative_change_pct": relative_change_pct(mean_val, post_val),
            "relative_reduction_pct": relative_reduction_pct(mean_val, post_val),
            "z_score_vs_sampled_pre": z_score_from_sampling(mean_val, se_val, post_val),
            "empirical_percentile_of_post": empirical_percentile(values, post_val),
            "empirical_two_sided_pvalue": p_two_sided,
        }

    benchmark_aggregate: Dict[str, Dict[str, float]] = {}
    post_benchmark = benchmark_item_variance(post_df)
    for benchmark in BENCHMARKS:
        values = [row.get(benchmark, np.nan) for row in benchmark_rows]
        if np.all(np.isnan(values)):
            continue
        mean_val = float(np.nanmean(values))
        se_val = safe_se(values)
        ci_lower, ci_upper = ci95(mean_val, se_val)
        post_val = float(post_benchmark.get(benchmark, float("nan")))
        p_two_sided = empirical_two_sided_pvalue(values, post_val)
        if not np.isnan(p_two_sided):
            p_two_sided = min(1.0, max(0.0, p_two_sided))

        benchmark_aggregate[benchmark] = {
            "mean": mean_val,
            "se": se_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "post": post_val,
            "delta_post_minus_sampled_pre": float(post_val - mean_val),
            "relative_change_pct": relative_change_pct(mean_val, post_val),
            "relative_reduction_pct": relative_reduction_pct(mean_val, post_val),
            "z_score_vs_sampled_pre": z_score_from_sampling(mean_val, se_val, post_val),
            "empirical_percentile_of_post": empirical_percentile(values, post_val),
            "empirical_two_sided_pvalue": p_two_sided,
        }

    match_aggregate: Dict[str, Dict[str, float]] = {}
    for key in [
        "sample_mean_strength",
        "post_mean_strength",
        "delta_mean_strength",
        "sample_sd_strength",
        "post_sd_strength",
        "delta_sd_strength",
    ]:
        values = [row.get(key, np.nan) for row in match_rows]
        mean_val = float(np.nanmean(values))
        se_val = safe_se(values)
        ci_lower, ci_upper = ci95(mean_val, se_val)
        match_aggregate[key] = {
            "mean": mean_val,
            "se": se_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    benchmark_strength_aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for benchmark in BENCHMARKS:
        rows = match_rows_by_benchmark.get(benchmark, [])
        if not rows:
            continue
        benchmark_strength_aggregate[benchmark] = {}
        for key in [
            "sample_mean_strength",
            "post_mean_strength",
            "delta_mean_strength",
            "sample_sd_strength",
            "post_sd_strength",
            "delta_sd_strength",
        ]:
            values = [row.get(key, np.nan) for row in rows]
            mean_val = float(np.nanmean(values))
            se_val = safe_se(values)
            ci_lower, ci_upper = ci95(mean_val, se_val)
            benchmark_strength_aggregate[benchmark][key] = {
                "mean": mean_val,
                "se": se_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }

    count_diagnostics: Dict[str, object] = {}
    if sample_validity_rows:
        count_diagnostics = {
            "post_benchmark_counts": sample_validity_rows[0]["post_benchmark_counts"],
            "sample_benchmark_counts": sample_validity_rows[0]["sample_benchmark_counts"],
            "sample_mean_observed_per_item": float(np.nanmean([row["sample_mean_observed_per_item"] for row in sample_validity_rows])),
            "post_mean_observed_per_item": float(np.nanmean([row["post_mean_observed_per_item"] for row in sample_validity_rows])),
            "observed_per_item_ratio": float(np.nanmean([row["observed_per_item_ratio"] for row in sample_validity_rows])),
        }

    difficulty_payload: Dict[str, object] = {}
    if enable_difficulty_restricted:
        retained_counts = [row["retained_item_count"] for row in difficulty_rows]
        retained_fracs = [row["retained_item_fraction"] for row in difficulty_rows]

        per_benchmark_retained: Dict[str, Dict[str, float]] = {}
        per_benchmark_too_few_rate: Dict[str, float] = {}
        for benchmark in BENCHMARKS:
            bench_counts = [row["retained_items_per_benchmark"].get(benchmark, 0) for row in difficulty_rows]
            per_benchmark_retained[benchmark] = {
                "mean": float(np.nanmean(bench_counts)),
                "min": float(np.nanmin(bench_counts)),
                "max": float(np.nanmax(bench_counts)),
            }
            per_benchmark_too_few_rate[benchmark] = float(
                np.mean([
                    benchmark in row["too_few_items_per_benchmark"]
                    for row in difficulty_rows
                ])
            )

        metric_keys = list(difficulty_rows[0]["per_metric"].keys()) if difficulty_rows else []
        per_metric_aggregate: Dict[str, Dict[str, float]] = {}
        for key in metric_keys:
            sample_vals = [row["per_metric"][key]["sample"] for row in difficulty_rows]
            post_vals = [row["per_metric"][key]["post"] for row in difficulty_rows]
            delta_vals = [row["per_metric"][key]["delta_post_minus_sample"] for row in difficulty_rows]
            per_metric_aggregate[key] = {
                "sample_mean": float(np.nanmean(sample_vals)),
                "sample_se": safe_se(sample_vals),
                "post_mean": float(np.nanmean(post_vals)),
                "post_se": safe_se(post_vals),
                "delta_mean": float(np.nanmean(delta_vals)),
                "delta_se": safe_se(delta_vals),
                "relative_reduction_pct": relative_reduction_pct(
                    float(np.nanmean(sample_vals)),
                    float(np.nanmean(post_vals)),
                ),
            }

        too_few_any = sorted(
            [
                benchmark
                for benchmark, rate in per_benchmark_too_few_rate.items()
                if rate > 0
            ]
        )

        difficulty_payload = {
            "enabled": True,
            "difficulty_low": float(difficulty_low),
            "difficulty_high": float(difficulty_high),
            "retained_item_count": {
                "mean": float(np.nanmean(retained_counts)) if retained_counts else float("nan"),
                "se": safe_se(retained_counts),
                "min": float(np.nanmin(retained_counts)) if retained_counts else float("nan"),
                "max": float(np.nanmax(retained_counts)) if retained_counts else float("nan"),
            },
            "retained_item_fraction": {
                "mean": float(np.nanmean(retained_fracs)) if retained_fracs else float("nan"),
                "se": safe_se(retained_fracs),
                "min": float(np.nanmin(retained_fracs)) if retained_fracs else float("nan"),
                "max": float(np.nanmax(retained_fracs)) if retained_fracs else float("nan"),
            },
            "retained_items_per_benchmark": per_benchmark_retained,
            "min_items_per_benchmark": int(difficulty_min_items_per_benchmark),
            "too_few_benchmark_rate": per_benchmark_too_few_rate,
            "benchmarks_with_any_too_few": too_few_any,
            "per_metric": per_metric_aggregate,
        }

    return {
        "target_rows": int(target_rows),
        "n_repeats": int(n_repeats),
        "sampler_name": sampler_name,
        "max_attempts": int(max_attempts),
        "invalid_attempts_total": int(invalid_attempts_total),
        "per_metric": aggregate,
        "per_benchmark_avg_item_variance": benchmark_aggregate,
        "strength_match_diagnostics": match_aggregate,
        "strength_match_diagnostics_per_benchmark": benchmark_strength_aggregate,
        "benchmark_count_diagnostics": count_diagnostics,
        "difficulty_restricted_sensitivity": difficulty_payload,
    }


def build_report(args: argparse.Namespace) -> Dict[str, object]:
    pre_dfs, _ = _load_pre_revision_response_matrix(args.pre_revision)
    pre_full = pre_dfs[0]
    post_beta, post_binary, post_stack = compute_post_binary_matrix()
    pre_full, post_beta, post_binary = align_columns(pre_full, post_beta, post_binary, args.column_mode)
    pre_binary = to_binary(pre_full)

    pre_full_summary = summarize_matrix(pre_full, entropy_on_binary=False)
    post_beta_summary = summarize_matrix(post_beta, entropy_on_binary=False)
    post_binary_summary = summarize_matrix(post_binary, entropy_on_binary=True)
    pre_binary_summary = summarize_matrix(pre_binary, entropy_on_binary=True)
    post_repeatability_summary = summarize_post_repeatability(post_stack)

    infer_rng = np.random.default_rng(args.seed + 1000)

    def avg_item_var_metric(df: pd.DataFrame) -> float:
        return summarize_matrix(df, entropy_on_binary=False)["avg_item_variance"]

    def macro_item_var_metric(df: pd.DataFrame) -> float:
        return summarize_matrix(df, entropy_on_binary=False)["macro_benchmark_avg_item_variance"]

    item_bootstrap_raw = bootstrap_item_metric_gap(
        pre_full,
        post_beta,
        metric_fn=avg_item_var_metric,
        n_bootstrap=args.n_bootstrap,
        rng=infer_rng,
    )
    item_bootstrap_raw_hard = bootstrap_item_metric_gap(
        pre_full,
        post_binary,
        metric_fn=avg_item_var_metric,
        n_bootstrap=args.n_bootstrap,
        rng=infer_rng,
    )
    item_bootstrap_macro = bootstrap_item_metric_gap(
        pre_full,
        post_beta,
        metric_fn=macro_item_var_metric,
        n_bootstrap=args.n_bootstrap,
        rng=infer_rng,
    )
    item_bootstrap_macro_hard = bootstrap_item_metric_gap(
        pre_full,
        post_binary,
        metric_fn=macro_item_var_metric,
        n_bootstrap=args.n_bootstrap,
        rng=infer_rng,
    )

    item_permutation_raw = permutation_test_item_metric_gap(
        pre_full,
        post_beta,
        metric_fn=avg_item_var_metric,
        n_permutations=args.n_permutations,
        rng=infer_rng,
    )
    item_permutation_raw_hard = permutation_test_item_metric_gap(
        pre_full,
        post_binary,
        metric_fn=avg_item_var_metric,
        n_permutations=args.n_permutations,
        rng=infer_rng,
    )
    item_permutation_macro = permutation_test_item_metric_gap(
        pre_full,
        post_beta,
        metric_fn=macro_item_var_metric,
        n_permutations=args.n_permutations,
        rng=infer_rng,
    )
    item_permutation_macro_hard = permutation_test_item_metric_gap(
        pre_full,
        post_binary,
        metric_fn=macro_item_var_metric,
        n_permutations=args.n_permutations,
        rng=infer_rng,
    )

    report = {
        "config": {
            "pre_revision": args.pre_revision,
            "column_mode": args.column_mode,
            "n_repeats": args.n_repeats,
            "seed": args.seed,
            "post_binarization_rule": "post_binary = where(isnan(post_beta), nan, (post_beta > 0.5).astype(float))",
        },
        "matrices": {
            "pre_full_raw": pre_full_summary,
            "post_beta": post_beta_summary,
            "post_binary": post_binary_summary,
            "pre_full_binary_sensitivity": pre_binary_summary,
        },
        "post_repeatability": post_repeatability_summary,
        "full_comparison": {
            "raw_pre_vs_soft_post": {
                "avg_item_variance_pre": pre_full_summary["avg_item_variance"],
                "avg_item_variance_post": post_beta_summary["avg_item_variance"],
                "delta_post_minus_pre": post_beta_summary["avg_item_variance"] - pre_full_summary["avg_item_variance"],
                "avg_agent_variance_pre": pre_full_summary["avg_agent_variance"],
                "avg_agent_variance_post": post_beta_summary["avg_agent_variance"],
                "mean_score_pre": pre_full_summary["overall_mean"],
                "mean_score_post_soft": post_beta_summary["overall_mean"],
            },
            "raw_pre_vs_binary_post": {
                "avg_item_variance_pre": pre_full_summary["avg_item_variance"],
                "avg_item_variance_post": post_binary_summary["avg_item_variance"],
                "delta_post_minus_pre": post_binary_summary["avg_item_variance"] - pre_full_summary["avg_item_variance"],
                "avg_agent_variance_pre": pre_full_summary["avg_agent_variance"],
                "avg_agent_variance_post": post_binary_summary["avg_agent_variance"],
                "mean_score_pre": pre_full_summary["overall_mean"],
                "mean_pass_rate_post": post_binary_summary["overall_mean"],
            },
            "binary_sensitivity_pre_vs_soft_post": {
                "avg_item_variance_pre_binary": pre_binary_summary["avg_item_variance"],
                "avg_item_variance_post_soft": post_beta_summary["avg_item_variance"],
                "delta_post_minus_pre_binary": post_beta_summary["avg_item_variance"] - pre_binary_summary["avg_item_variance"],
                "mean_item_entropy_pre_binary": pre_binary_summary["mean_item_entropy"],
            },
            "binary_sensitivity_pre_vs_post": {
                "avg_item_variance_pre_binary": pre_binary_summary["avg_item_variance"],
                "avg_item_variance_post_binary": post_binary_summary["avg_item_variance"],
                "delta_post_minus_pre_binary": post_binary_summary["avg_item_variance"] - pre_binary_summary["avg_item_variance"],
                "mean_item_entropy_pre_binary": pre_binary_summary["mean_item_entropy"],
                "mean_item_entropy_post_binary": post_binary_summary["mean_item_entropy"],
            },
            "soft_vs_hard_post_binarization_effect": {
                "mean_score_post_soft": post_beta_summary["overall_mean"],
                "mean_score_post_hard": post_binary_summary["overall_mean"],
                "delta_hard_minus_soft": post_binary_summary["overall_mean"] - post_beta_summary["overall_mean"],
                "avg_item_variance_post_soft": post_beta_summary["avg_item_variance"],
                "avg_item_variance_post_hard": post_binary_summary["avg_item_variance"],
                "delta_item_variance_hard_minus_soft": post_binary_summary["avg_item_variance"] - post_beta_summary["avg_item_variance"],
                "zero_variance_item_fraction_post_soft": post_beta_summary["zero_variance_item_fraction"],
                "zero_variance_item_fraction_post_hard": post_binary_summary["zero_variance_item_fraction"],
            },
            "per_benchmark_avg_item_variance": {
                "pre_full_raw": benchmark_item_variance(pre_full),
                "post_beta": benchmark_item_variance(post_beta),
                "post_binary": benchmark_item_variance(post_binary),
                "pre_full_binary_sensitivity": benchmark_item_variance(pre_binary),
            },
        },
        "matched_sampling": {
            "benchmark_constrained_ability_matched_raw_pre_vs_soft_post": repeated_sampling_report(
                pre_full,
                post_beta,
                n_repeats=args.n_repeats,
                rng_seed=args.seed,
                include_entropy=False,
                sampler_name="benchmark_constrained_ability_matched",
                enable_difficulty_restricted=True,
                difficulty_low=args.difficulty_low,
                difficulty_high=args.difficulty_high,
                difficulty_min_items_per_benchmark=args.difficulty_min_items_per_benchmark,
            ),
            "benchmark_constrained_ability_matched_binary_sensitivity_pre_vs_soft_post": repeated_sampling_report(
                pre_binary,
                post_beta,
                n_repeats=args.n_repeats,
                rng_seed=args.seed,
                include_entropy=True,
                sampler_name="benchmark_constrained_ability_matched",
                enable_difficulty_restricted=True,
                difficulty_low=args.difficulty_low,
                difficulty_high=args.difficulty_high,
                difficulty_min_items_per_benchmark=args.difficulty_min_items_per_benchmark,
            ),
            "benchmark_balanced_raw_pre_vs_binary_post": repeated_sampling_report(
                pre_full,
                post_binary,
                n_repeats=args.n_repeats,
                rng_seed=args.seed,
                include_entropy=False,
                sampler_name="benchmark_balanced",
                observed_per_item_ratio_floor=0.8,
                difficulty_low=args.difficulty_low,
                difficulty_high=args.difficulty_high,
                difficulty_min_items_per_benchmark=args.difficulty_min_items_per_benchmark,
            ),
            "benchmark_balanced_binary_sensitivity_pre_vs_post": repeated_sampling_report(
                pre_binary,
                post_binary,
                n_repeats=args.n_repeats,
                rng_seed=args.seed,
                include_entropy=True,
                sampler_name="benchmark_balanced",
                observed_per_item_ratio_floor=0.8,
                difficulty_low=args.difficulty_low,
                difficulty_high=args.difficulty_high,
                difficulty_min_items_per_benchmark=args.difficulty_min_items_per_benchmark,
            ),
            "benchmark_constrained_ability_matched_raw_pre_vs_binary_post": repeated_sampling_report(
                pre_full,
                post_binary,
                n_repeats=args.n_repeats,
                rng_seed=args.seed,
                include_entropy=False,
                sampler_name="benchmark_constrained_ability_matched",
                enable_difficulty_restricted=True,
                difficulty_low=args.difficulty_low,
                difficulty_high=args.difficulty_high,
                difficulty_min_items_per_benchmark=args.difficulty_min_items_per_benchmark,
            ),
            "benchmark_constrained_ability_matched_binary_sensitivity_pre_vs_post": repeated_sampling_report(
                pre_binary,
                post_binary,
                n_repeats=args.n_repeats,
                rng_seed=args.seed,
                include_entropy=True,
                sampler_name="benchmark_constrained_ability_matched",
                enable_difficulty_restricted=True,
                difficulty_low=args.difficulty_low,
                difficulty_high=args.difficulty_high,
                difficulty_min_items_per_benchmark=args.difficulty_min_items_per_benchmark,
            ),
        },
        "item_level_inference": {
            "raw_pre_vs_soft_post": {
                "avg_item_variance_bootstrap": item_bootstrap_raw,
                "macro_benchmark_avg_item_variance_bootstrap": item_bootstrap_macro,
                "avg_item_variance_permutation": item_permutation_raw,
                "macro_benchmark_avg_item_variance_permutation": item_permutation_macro,
            },
            "raw_pre_vs_binary_post": {
                "avg_item_variance_bootstrap": item_bootstrap_raw_hard,
                "macro_benchmark_avg_item_variance_bootstrap": item_bootstrap_macro_hard,
                "avg_item_variance_permutation": item_permutation_raw_hard,
                "macro_benchmark_avg_item_variance_permutation": item_permutation_macro_hard,
            }
        },
    }
    return report


def format_metric_line(label: str, value: float) -> str:
    if np.isnan(value):
        return f"- {label}: nan"
    if float(value).is_integer():
        return f"- {label}: {int(value)}"
    return f"- {label}: {value:.6f}"


def format_sampling_block(title: str, payload: Dict[str, object]) -> List[str]:
    lines = [f"### {title}"]
    lines.append(f"- Target pre sample size: {payload['target_rows']}")
    lines.append(f"- Repeats: {payload['n_repeats']}")
    if "sampler_name" in payload:
        lines.append(f"- Sampler: {payload['sampler_name']}")
    if "max_attempts" in payload:
        lines.append(f"- Max resampling attempts per repeat: {payload['max_attempts']}")
    if "invalid_attempts_total" in payload:
        lines.append(f"- Rejected samples during validation: {payload['invalid_attempts_total']}")
    if payload.get("benchmark_count_diagnostics"):
        count_diag = payload["benchmark_count_diagnostics"]
        lines.append("- Benchmark count diagnostics:")
        lines.append(f"  - sample counts: {count_diag['sample_benchmark_counts']}")
        lines.append(f"  - post counts: {count_diag['post_benchmark_counts']}")
        lines.append(
            f"  - mean_observed_per_item(sample)={count_diag['sample_mean_observed_per_item']:.6f}, "
            f"mean_observed_per_item(post)={count_diag['post_mean_observed_per_item']:.6f}, "
            f"ratio={count_diag['observed_per_item_ratio']:.6f}"
        )
    for metric, stats in payload["per_metric"].items():
        lines.append(
            f"- {metric}: sampled pre mean={stats['mean']:.6f}, "
            f"SE={stats['se']:.6f}, 95% CI=[{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}], "
            f"post={stats['post']:.6f}, delta(post-pre)={stats['delta_post_minus_sampled_pre']:.6f}, "
            f"reduction={stats['relative_reduction_pct']:.2f}%, "
            f"z={stats['z_score_vs_sampled_pre']:.3f}, "
            f"post percentile in pre samples={stats['empirical_percentile_of_post']:.3f}, "
            f"empirical two-sided p={stats['empirical_two_sided_pvalue']:.3f}"
        )
    if payload["per_benchmark_avg_item_variance"]:
        lines.append("- Benchmark-wise avg item variance:")
        for benchmark, stats in payload["per_benchmark_avg_item_variance"].items():
            lines.append(
                f"  - {benchmark}: sampled pre mean={stats['mean']:.6f}, "
                f"SE={stats['se']:.6f}, 95% CI=[{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}], "
                f"post={stats['post']:.6f}, delta(post-pre)={stats['delta_post_minus_sampled_pre']:.6f}, "
                f"reduction={stats['relative_reduction_pct']:.2f}%, "
                f"z={stats['z_score_vs_sampled_pre']:.3f}, "
                f"post percentile={stats['empirical_percentile_of_post']:.3f}, "
                f"empirical two-sided p={stats['empirical_two_sided_pvalue']:.3f}"
            )
    if payload.get("strength_match_diagnostics"):
        lines.append("- Strength-match diagnostics:")
        for key, stats in payload["strength_match_diagnostics"].items():
            lines.append(
                f"  - {key}: mean={stats['mean']:.6f}, "
                f"SE={stats['se']:.6f}, 95% CI=[{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]"
            )
    if payload.get("strength_match_diagnostics_per_benchmark"):
        lines.append("- Strength-match diagnostics by benchmark:")
        for benchmark, benchmark_stats in payload["strength_match_diagnostics_per_benchmark"].items():
            lines.append(f"  - {benchmark}:")
            for key, stats in benchmark_stats.items():
                lines.append(
                    f"    - {key}: mean={stats['mean']:.6f}, "
                    f"SE={stats['se']:.6f}, 95% CI=[{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]"
                )
    if payload.get("difficulty_restricted_sensitivity", {}).get("enabled", False):
        diff = payload["difficulty_restricted_sensitivity"]
        lines.append("- Difficulty-restricted sensitivity:")
        lines.append(
            f"  - retained where sample/post pass rates both in "
            f"[{diff['difficulty_low']:.3f}, {diff['difficulty_high']:.3f}]"
        )
        lines.append(
            f"  - retained_item_count: mean={diff['retained_item_count']['mean']:.3f}, "
            f"SE={diff['retained_item_count']['se']:.3f}, "
            f"min={diff['retained_item_count']['min']:.0f}, max={diff['retained_item_count']['max']:.0f}"
        )
        lines.append(
            f"  - retained_item_fraction: mean={diff['retained_item_fraction']['mean']:.6f}, "
            f"SE={diff['retained_item_fraction']['se']:.6f}, "
            f"min={diff['retained_item_fraction']['min']:.6f}, max={diff['retained_item_fraction']['max']:.6f}"
        )
        lines.append(f"  - min_items_per_benchmark={diff['min_items_per_benchmark']}")
        lines.append("  - retained items per benchmark:")
        for benchmark, stats in diff["retained_items_per_benchmark"].items():
            too_few_rate = diff["too_few_benchmark_rate"].get(benchmark, float("nan"))
            lines.append(
                f"    - {benchmark}: mean={stats['mean']:.3f}, min={stats['min']:.0f}, max={stats['max']:.0f}, "
                f"too_few_rate={too_few_rate:.3f}"
            )
        if diff.get("benchmarks_with_any_too_few"):
            lines.append(
                f"  - explicitly flagged benchmarks with too-few retained items: "
                f"{diff['benchmarks_with_any_too_few']}"
            )
        lines.append("  - filtered metric comparison (post minus sampled pre):")
        for metric, stats in diff["per_metric"].items():
            lines.append(
                f"    - {metric}: sample={stats['sample_mean']:.6f} (SE={stats['sample_se']:.6f}), "
                f"post={stats['post_mean']:.6f} (SE={stats['post_se']:.6f}), "
                f"delta={stats['delta_mean']:.6f} (SE={stats['delta_se']:.6f}), "
                f"reduction={stats['relative_reduction_pct']:.2f}%"
            )
    return lines


def render_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    config = report["config"]
    matrices = report["matrices"]
    post_repeatability = report.get("post_repeatability", {})
    full = report["full_comparison"]
    matched = report["matched_sampling"]

    lines.append("# Pre/Post Stability Diagnosis")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Pre-revision setting: `{config['pre_revision']}`")
    lines.append(f"- Column alignment: `{config['column_mode']}`")
    lines.append(f"- Repeated samples: {config['n_repeats']}")
    lines.append(f"- Seed: {config['seed']}")
    lines.append(f"- Post binarization: `{config['post_binarization_rule']}`")
    lines.append("")
    lines.append("## Matrix Summary")
    for name, stats in matrices.items():
        lines.append(f"### {name}")
        for key in [
            "rows",
            "cols",
            "observed_fraction",
            "overall_mean",
            "avg_item_variance",
            "avg_agent_variance",
            "zero_variance_item_fraction",
        ]:
            lines.append(format_metric_line(key, float(stats[key])))
        if "mean_item_entropy" in stats:
            lines.append(format_metric_line("mean_item_entropy", float(stats["mean_item_entropy"])))
    if post_repeatability:
        lines.append("")
        lines.append("## Post Repeatability")
        for key, value in post_repeatability.items():
            lines.append(format_metric_line(key, float(value)))
    lines.append("")
    lines.append("## Full Comparison")
    for section_name, stats in full.items():
        lines.append(f"### {section_name}")
        if isinstance(stats, dict) and stats and all(isinstance(v, dict) for v in stats.values()):
            for benchmark, benchmark_stats in stats.items():
                lines.append(f"- {benchmark}:")
                for key, value in benchmark_stats.items():
                    lines.append(f"  - {key}: {value:.6f}")
        else:
            for key, value in stats.items():
                lines.append(format_metric_line(key, float(value)))
    lines.append("")
    lines.append("## Matched Sampling")
    for section_name, payload in matched.items():
        lines.extend(format_sampling_block(section_name, payload))
        lines.append("")
    if "item_level_inference" in report:
        lines.append("## Item-Level Inference")
        for section_name, payload in report["item_level_inference"].items():
            lines.append(f"### {section_name}")
            for test_name, stats in payload.items():
                lines.append(f"- {test_name}:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        lines.append(format_metric_line(f"  {key}", value))
                    else:
                        lines.append(f"  - {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pre- vs post-revision matrix stability.")
    parser.add_argument(
        "--pre-revision",
        type=str,
        default="max",
        help='Pre-revision subset to load via amortized_irt.py logic. Use "max" for the full matrix.',
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=50,
        help="Number of matched-size pre subsamples to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for repeated matched-size pre subsampling.",
    )
    parser.add_argument(
        "--column-mode",
        choices=["intersection", "union"],
        default="intersection",
        help="Whether to compare on the shared item set only or the union of item columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the Markdown report.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to save the raw report as JSON.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of item-level bootstrap resamples for pre/post gap inference.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of item-level permutations for null testing of pre/post gap.",
    )
    parser.add_argument(
        "--difficulty-low",
        type=float,
        default=0.2,
        help="Lower pass-rate bound for difficulty-restricted item filtering.",
    )
    parser.add_argument(
        "--difficulty-high",
        type=float,
        default=0.8,
        help="Upper pass-rate bound for difficulty-restricted item filtering.",
    )
    parser.add_argument(
        "--difficulty-min-items-per-benchmark",
        type=int,
        default=5,
        help="Minimum retained item count per benchmark before flagging too-few items.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    markdown = render_markdown(report)
    print(markdown)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(markdown)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
