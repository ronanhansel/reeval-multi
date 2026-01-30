#!/usr/bin/env python3
"""
Split pre-revision resmat into per-benchmark matrices and align rubric labels.

Outputs:
  - data-reeval-multi/pre-revision/<benchmark>/benchmark.csv
  - data-reeval-multi/pre-revision/<benchmark>/benchmark_rubric.csv (if rubric exists)
  - data-reeval-multi/post-revision/<benchmark>/benchmark_rubric.csv (if rubric exists)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data-reeval-multi"
PRE_DIR = DATA_ROOT / "pre-revision"
POST_DIR = DATA_ROOT / "post-revision"
RUBRICS_DIR = DATA_ROOT / "rubrics" / "post-revision"


def _normalize_value(val) -> Optional[int]:
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        if val in (0, 1):
            return int(val)
        return int(val > 0)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1"):
            return 1
        if v in ("false", "0"):
            return 0
    return None


def load_rubrics(rubrics_dir: Path) -> Dict[str, Dict[int, Dict[str, int]]]:
    """Return mapping: benchmark -> revision_index -> task_id(str) -> satisfies_rubric(int)."""
    out: Dict[str, Dict[int, Dict[str, int]]] = {}
    if not rubrics_dir.exists():
        return out

    for path in rubrics_dir.glob("*.csv"):
        name = path.stem
        if "_" not in name:
            continue
        bench, idx_str = name.rsplit("_", 1)
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        df = pd.read_csv(path)
        if "task_id" not in df.columns or "satisfies_rubric" not in df.columns:
            continue
        task_map: Dict[str, int] = {}
        for _, row in df.iterrows():
            task_id = str(row["task_id"])
            val = _normalize_value(row["satisfies_rubric"])
            if val is None:
                continue
            task_map[task_id] = val
        out.setdefault(bench, {})[idx] = task_map
    return out


def _extract_task_id(col: str) -> str:
    if "." in col:
        return col.split(".", 1)[1]
    return col


def build_rubric_row(
    columns: List[str],
    rubric_maps: Dict[str, Dict[int, Dict[str, int]]],
    rubric_bench: str,
    revision_indices: Iterable[int],
) -> List[Optional[int]]:
    revisions = [rubric_maps[rubric_bench][i] for i in revision_indices if rubric_bench in rubric_maps and i in rubric_maps[rubric_bench]]
    if not revisions:
        return [None for _ in columns]

    row: List[Optional[int]] = []
    for col in columns:
        task_id = _extract_task_id(col)
        vals: List[int] = []
        for rev in revisions:
            v = rev.get(task_id)
            if v is not None:
                vals.append(v)
        if not vals:
            row.append(None)
        else:
            row.append(min(vals))
    return row


def write_rubric_csv(path: Path, columns: List[str], row: List[Optional[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerow(["" if v is None else v for v in row])


def split_pre_revision(rubric_maps: Dict[str, Dict[int, Dict[str, int]]]) -> None:
    resmat_path = PRE_DIR / "resmat.csv"
    if not resmat_path.exists():
        print(f"Missing pre-revision resmat: {resmat_path}")
        return

    df = pd.read_csv(resmat_path)
    id_col = "test_taker_id" if "test_taker_id" in df.columns else None
    columns = [c for c in df.columns if c != id_col]
    by_benchmark: Dict[str, List[str]] = {}
    for col in columns:
        if "." not in col:
            continue
        bench = col.split(".", 1)[0]
        by_benchmark.setdefault(bench, []).append(col)

    rubric_bench_map = {
        "assistantbench": "assistantbench",
        "colbench_backend_programming": "colbench_backend_programming",
        "corebench_hard": "corebench",
        "scienceagentbench": "scienceagentbench",
        "scicode": "scicode",
        "swebench_verified_mini": "swebench",
    }

    for bench, cols in by_benchmark.items():
        out_dir = PRE_DIR / bench
        out_dir.mkdir(parents=True, exist_ok=True)
        out_cols = [id_col] + cols if id_col is not None else cols
        df[out_cols].to_csv(out_dir / "benchmark.csv", index=False)

        rubric_bench = rubric_bench_map.get(bench)
        if rubric_bench is None:
            continue
        row = build_rubric_row(cols, rubric_maps, rubric_bench, [1])
        write_rubric_csv(out_dir / "benchmark_rubric.csv", cols, row)


def post_revision_rubrics(rubric_maps: Dict[str, Dict[int, Dict[str, int]]]) -> None:
    if not POST_DIR.exists():
        print(f"Missing post-revision directory: {POST_DIR}")
        return

    rubric_bench_map = {
        "assistantbench": "assistantbench",
        "colbench": "colbench_backend_programming",
        "corebench": "corebench",
        "scienceagentbench": "scienceagentbench",
        "scicode": "scicode",
        "swebench": "swebench",
    }

    for bench_dir in sorted([p for p in POST_DIR.iterdir() if p.is_dir()]):
        bench_name = bench_dir.name
        resmat_files = sorted(bench_dir.glob("resmat_*.csv"))
        if not resmat_files:
            continue
        with resmat_files[0].open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        if not header:
            continue
        columns = header
        if columns[0] in ("agent", "test_taker_id"):
            columns = columns[1:]
        rubric_bench = rubric_bench_map.get(bench_name, bench_name)
        row = build_rubric_row(columns, rubric_maps, rubric_bench, [i for i in range(2, 10)])
        if all(v is None for v in row):
            continue
        write_rubric_csv(bench_dir / "benchmark_rubric.csv", columns, row)


def main() -> None:
    rubric_maps = load_rubrics(RUBRICS_DIR)
    split_pre_revision(rubric_maps)
    post_revision_rubrics(rubric_maps)


if __name__ == "__main__":
    main()
