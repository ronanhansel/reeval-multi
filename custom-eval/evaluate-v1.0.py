import argparse
import csv
import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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


def normalize_answer(text: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    def remove_special_tokens(t: str) -> str:
        return t.replace("</s>", "").replace("<|eot_id|>", "")

    return white_space_fix(remove_articles(remove_punc(remove_special_tokens(text.lower()))))

def get_tokens(text: str) -> List[str]:
    return normalize_answer(text).split() if text else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gold_tokens = get_tokens(gold)
    pred_tokens = get_tokens(pred)
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if not gold_tokens or not pred_tokens:
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def load_gold_answers(csv_path: Path) -> Dict[str, List[str]]:
    gold_answers: Dict[str, List[str]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            qid = row["id"].strip()
            answer = (row.get("answer_text") or "").strip()
            if answer not in gold_answers[qid]:
                gold_answers[qid].append(answer)
    return dict(gold_answers)


def load_predictions(predictions_path: Path) -> Dict[str, str]:
    with predictions_path.open(encoding="utf-8") as predictions_file:
        payload = json.load(predictions_file)

    if isinstance(payload, dict):
        return {str(qid): str(answer).strip() for qid, answer in payload.items()}

    if isinstance(payload, list):
        predictions: Dict[str, str] = {}
        for entry in payload:
            if not isinstance(entry, dict) or "id" not in entry:
                raise ValueError("Prediction list entries must be dicts containing at least an 'id' key.")
            qid = str(entry["id"])
            answer = str(entry.get("answer", "")).strip()
            predictions[qid] = answer
        return predictions

    raise ValueError("Predictions JSON must be either an object (id->answer) or a list of {id, answer} objects.")


def evaluate_predictions_path(
    predictions_path: Path,
    gold_answers: Dict[str, List[str]],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    predictions = load_predictions(predictions_path)
    return evaluate(gold_answers, predictions)


def emit_warnings(label: str, missing_ids: List[str], extra_ids: List[str]) -> None:
    prefix = f"[{label}] " if label else ""
    if missing_ids:
        print(f"{prefix}Warning: missing predictions for {len(missing_ids)} ids.", file=sys.stderr)
    if extra_ids:
        print(f"{prefix}Warning: {len(extra_ids)} predictions did not match any dataset ids.", file=sys.stderr)


def evaluate(
    gold_answers: Dict[str, List[str]],
    predictions: Dict[str, str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    total = len(gold_answers)
    exact_sum = 0.0
    f1_sum = 0.0
    missing_ids: List[str] = []

    for qid, answers in gold_answers.items():
        if qid not in predictions:
            missing_ids.append(qid)
            predicted_answer = ""
        else:
            predicted_answer = predictions[qid]

        candidate_answers = answers or [""]
        exact_sum += max(compute_exact(gold, predicted_answer) for gold in candidate_answers)
        f1_sum += max(compute_f1(gold, predicted_answer) for gold in candidate_answers)

    extra_ids = sorted(set(predictions.keys()) - set(gold_answers.keys()))
    metrics = {
        "total": total,
        "exact": 100.0 * exact_sum / total if total else 0.0,
        "f1": 100.0 * f1_sum / total if total else 0.0,
        "missing_predictions": len(missing_ids),
        "extra_predictions": len(extra_ids),
    }
    return metrics, missing_ids, extra_ids


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
