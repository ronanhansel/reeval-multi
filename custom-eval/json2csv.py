#!/usr/bin/env python3
"""Create a response matrix (model × question) of SQuAD-style F1 scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from qa_eval_utils import (
    compute_question_f1,
    emit_warnings,
    load_gold_answers,
    load_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce a CSV response matrix where each row is a predictions file (model) "
            "and each column is a question id populated with the question-level F1 score."
        )
    )
    default_dataset = Path(__file__).parent / "data" / "squad_validation_lite.csv"
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"CSV file containing validation data (default: {default_dataset}).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to either a single predictions JSON file or a folder containing multiple *.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination CSV file or directory for the response matrix. "
            "If omitted, defaults to ./output/response_matrix.csv."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places for the F1 scores (default: 4).",
    )
    return parser.parse_args()


def resolve_output_path(raw_output: Optional[Path]) -> Path:
    """Resolve the desired CSV destination, treating bare paths as directories."""

    default_dir = Path("./output")
    default_name = "response_matrix.csv"

    if raw_output is None:
        return default_dir / default_name

    # If the user points to an existing directory or a path without an extension, treat as a folder.
    if raw_output.is_dir() or raw_output.suffix.lower() != ".csv":
        return raw_output / default_name

    return raw_output


def collect_prediction_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json")
        if not files:
            raise FileNotFoundError("No *.json files found in the provided predictions folder.")
        return files
    raise FileNotFoundError(f"Predictions path {path} does not exist.")


def write_response_matrix(
    question_ids: Sequence[str],
    rows: Sequence[Tuple[str, Mapping[str, float]]],
    output_path: Path,
    precision: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model", *question_ids])
        for label, f1_by_qid in rows:
            row = [label]
            for qid in question_ids:
                score = f1_by_qid.get(qid, 0.0)
                row.append(f"{score:.{precision}f}")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    gold_answers = load_gold_answers(args.dataset)
    if not gold_answers:
        print("Error: dataset did not yield any gold answers.", file=sys.stderr)
        raise SystemExit(2)

    question_ids = sorted(gold_answers.keys())
    prediction_files = collect_prediction_files(args.predictions)

    rows: List[Tuple[str, Dict[str, float]]] = []
    for predictions_path in prediction_files:
        label = predictions_path.stem
        predictions = load_predictions(predictions_path)
        per_question, missing_ids, extra_ids = compute_question_f1(gold_answers, predictions)
        emit_warnings(label, missing_ids, extra_ids)
        rows.append((label, per_question))

    output_path = resolve_output_path(args.output)

    write_response_matrix(question_ids, rows, output_path, args.precision)
    print(
        "Wrote response matrix with "
        f"{len(rows)} model(s) and {len(question_ids)} question(s) to {output_path}"
    )


if __name__ == "__main__":
    main()
