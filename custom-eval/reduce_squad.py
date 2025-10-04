"""Utilities for producing a reduced SQuAD validation split."""

from __future__ import annotations

import argparse
import csv
import random
from collections import OrderedDict
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence


def _group_rows_by_title_and_question(
	rows: Iterable[Dict[str, str]]
) -> "OrderedDict[str, OrderedDict[str, List[Dict[str, str]]]]":
	grouped: "OrderedDict[str, OrderedDict[str, List[Dict[str, str]]]]" = OrderedDict()
	for row in rows:
		title = row["title"]
		question = row["question"]
		if title not in grouped:
			grouped[title] = OrderedDict()
		if question not in grouped[title]:
			grouped[title][question] = []
		grouped[title][question].append(row)
	return grouped


def reduce_squad_file(
	input_csv: Path,
	output_csv: Path,
	*,
	max_questions_per_title: int = 10,
	rng: Random | None = None,
) -> int:
	"""Create a reduced SQuAD validation file.

	Args:
		input_csv: Path to the original SQuAD validation CSV.
		output_csv: Path where the reduced CSV will be written.
		max_questions_per_title: Maximum number of unique questions to keep per title.

	Returns:
		Number of rows written to the output CSV.
	"""

	if max_questions_per_title <= 0:
		raise ValueError("max_questions_per_title must be a positive integer")

	if not input_csv.exists():
		raise FileNotFoundError(f"Input CSV not found: {input_csv}")

	with input_csv.open("r", newline="", encoding="utf-8") as infile:
		reader = csv.DictReader(infile)
		fieldnames = reader.fieldnames
		if not fieldnames:
			raise ValueError("Input CSV appears to be missing a header row.")

		grouped_rows = _group_rows_by_title_and_question(reader)

	if rng is None:
		rng = random.Random()

	output_rows: List[Dict[str, str]] = []

	for title, question_map in grouped_rows.items():
		question_items = list(question_map.items())
		rng.shuffle(question_items)
		for question, question_rows in question_items[:max_questions_per_title]:
			output_rows.extend(question_rows)

	output_csv.parent.mkdir(parents=True, exist_ok=True)

	with output_csv.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(output_rows)

	return len(output_rows)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Create a lite version of the SQuAD validation split by keeping a "
			"fixed number of unique questions per title while preserving all "
			"gold answers for the selected questions."
		)
	)
	parser.add_argument(
		"input_csv",
		type=Path,
		help="Path to the original SQuAD validation CSV file.",
	)
	parser.add_argument(
		"output_csv",
		type=Path,
		help="Path where the lite SQuAD CSV will be saved.",
	)
	parser.add_argument(
		"--max-questions-per-title",
		type=int,
		default=20,
		help="Maximum number of unique questions to keep for each title (default: 10).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional random seed for reproducible question sampling.",
	)
	return parser.parse_args(args=args)


def main(cli_args: Sequence[str] | None = None) -> None:
	args = parse_args(cli_args)
	rng = random.Random(args.seed) if args.seed is not None else random.Random()
	rows_written = reduce_squad_file(
		args.input_csv,
		args.output_csv,
		max_questions_per_title=args.max_questions_per_title,
		rng=rng,
	)
	print(
		f"Wrote {rows_written} rows to {args.output_csv} "
		f"(max {args.max_questions_per_title} random questions per title)."
	)


if __name__ == "__main__":
	main()
