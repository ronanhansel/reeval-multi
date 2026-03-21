#!/usr/bin/env python3
"""
Summarize invalid-item coverage for all pre-revision benchmarks.

For each benchmark directory under the pre-revision response matrix, this script
reports:
  - total number of items
  - number of agents
  - number of rubric items matched to benchmark items
  - number of invalid items (rubric value == 1)
  - percentage invalid

Benchmarks without a benchmark_rubric.csv are still included, with rubric-based
fields shown as ---.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_PRE_REVISION_DIR = Path(
    "/Users/ronan/Developer/agent-eval/item-editor/eval_response_matrix/pre-revision"
)


def _format_int(value: int | None) -> str:
    return "---" if value is None else str(value)


def _format_pct(value: float | None) -> str:
    return "---" if value is None else f"{value:.2f}%"


def _load_benchmark_summary(benchmark_dir: Path) -> Dict[str, int | float | str | None]:
    benchmark_csv = benchmark_dir / "benchmark.csv"
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"Missing benchmark.csv for {benchmark_dir.name}: {benchmark_csv}")

    benchmark_df = pd.read_csv(benchmark_csv)
    item_columns = [col for col in benchmark_df.columns if col not in {"agent", "test_taker_id"}]
    total_items = len(item_columns)
    agent_count = len(benchmark_df.index)

    summary: Dict[str, int | float | str | None] = {
        "benchmark": benchmark_dir.name,
        "total_items": total_items,
        "agents": agent_count,
        "matched_rubric_items": None,
        "invalid_items": None,
        "invalid_pct": None,
    }

    rubric_csv = benchmark_dir / "benchmark_rubric.csv"
    if not rubric_csv.exists():
        return summary

    rubric_df = pd.read_csv(rubric_csv)
    if rubric_df.empty:
        return summary

    rubric_row = rubric_df.iloc[0]
    matched_columns = [col for col in item_columns if col in rubric_row.index and pd.notna(rubric_row[col])]
    invalid_items = 0
    for col in matched_columns:
        value = rubric_row[col]
        if value == 1 or value == 1.0:
            invalid_items += 1

    summary["matched_rubric_items"] = len(matched_columns)
    summary["invalid_items"] = invalid_items
    summary["invalid_pct"] = (invalid_items / total_items * 100.0) if total_items else 0.0
    return summary


def build_markdown(rows: List[Dict[str, int | float | str | None]]) -> str:
    lines = [
        "# Pre-Revision Rubric Invalid-Item Summary",
        "",
        "| Benchmark | Total Items | Agents | Matched Rubric Items | Invalid Items | % Invalid |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {benchmark} | {total_items} | {agents} | {matched_rubric_items} | {invalid_items} | {invalid_pct} |".format(
                benchmark=row["benchmark"],
                total_items=_format_int(row["total_items"]),
                agents=_format_int(row["agents"]),
                matched_rubric_items=_format_int(row["matched_rubric_items"]),
                invalid_items=_format_int(row["invalid_items"]),
                invalid_pct=_format_pct(row["invalid_pct"]),
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pre_revision_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PRE_REVISION_DIR,
        help="Directory containing the 10 pre-revision benchmark folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the markdown summary.",
    )
    args = parser.parse_args()

    benchmark_dirs = sorted(path for path in args.pre_revision_dir.iterdir() if path.is_dir())
    rows = [_load_benchmark_summary(benchmark_dir) for benchmark_dir in benchmark_dirs]
    markdown = build_markdown(rows)

    print(markdown, end="")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"\nWrote summary to {args.output}")


if __name__ == "__main__":
    main()
