import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from qa_eval_utils import emit_warnings, evaluate, load_gold_answers, load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SQuAD predictions produced by run_eval.py.")
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
        help="JSON file with model predictions (list of {id, answer} objects or id->answer mapping).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Treat --predictions as a folder and evaluate every *.json file inside it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the evaluation metrics as JSON.",
    )
    return parser.parse_args()


def evaluate_predictions_path(
    predictions_path: Path,
    gold_answers: Dict[str, List[str]],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    predictions = load_predictions(predictions_path)
    return evaluate(gold_answers, predictions)


def main() -> None:
    args = parse_args()
    gold_answers = load_gold_answers(args.dataset)
    if args.all:
        if not args.predictions.is_dir():
            print("Error: --all expects --predictions to point to a folder.", file=sys.stderr)
            raise SystemExit(2)

        prediction_files = sorted(
            path for path in args.predictions.iterdir() if path.is_file() and path.suffix.lower() == ".json"
        )

        if not prediction_files:
            print("Error: no *.json files found in the provided predictions folder.", file=sys.stderr)
            raise SystemExit(2)

        aggregated_results = {}
        for predictions_path in prediction_files:
            metrics, missing_ids, extra_ids = evaluate_predictions_path(predictions_path, gold_answers)
            aggregated_results[predictions_path.name] = {
                "metrics": metrics,
                "missing_ids": missing_ids,
                "extra_ids": extra_ids,
            }

            print(f"## {predictions_path.name}")
            print(json.dumps(metrics, indent=2))
            emit_warnings(predictions_path.name, missing_ids, extra_ids)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as metrics_file:
                json.dump(aggregated_results, metrics_file, indent=2)
    else:
        metrics, missing_ids, extra_ids = evaluate_predictions_path(args.predictions, gold_answers)

        print(json.dumps(metrics, indent=2))

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)

        emit_warnings("", missing_ids, extra_ids)


if __name__ == "__main__":
    main()
