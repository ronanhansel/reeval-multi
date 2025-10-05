"""Shared utilities for evaluating SQuAD-style predictions."""

from __future__ import annotations

import csv
import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "normalize_answer",
    "get_tokens",
    "compute_exact",
    "compute_f1",
    "load_gold_answers",
    "load_predictions",
    "evaluate",
    "compute_question_f1",
    "emit_warnings",
]


def normalize_answer(text: str) -> str:
    """Lowercase, trim punctuation/articles/special tokens, and collapse whitespace."""

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
    """Tokenize normalized text."""
    return normalize_answer(text).split() if text else []


def compute_exact(gold: str, pred: str) -> int:
    """Return 1 if the normalized gold and prediction match exactly."""
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    """Standard SQuAD-style F1 between a gold answer and a prediction."""
    gold_tokens = get_tokens(gold)
    pred_tokens = get_tokens(pred)
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def load_gold_answers(csv_path: Path) -> Dict[str, List[str]]:
    """Parse a dataset CSV into a mapping of question id -> list of gold answers."""
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
    """Load a predictions JSON file into a mapping of question id -> answer."""
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


def evaluate(
    gold_answers: Mapping[str, Sequence[str]],
    predictions: Mapping[str, str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Compute dataset-level metrics and record missing/extra ids."""
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


def compute_question_f1(
    gold_answers: Mapping[str, Sequence[str]],
    predictions: Mapping[str, str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Return per-question F1 scores plus any missing/extra ids."""
    per_question: Dict[str, float] = {}
    missing_ids: List[str] = []

    for qid, answers in gold_answers.items():
        predicted_answer = predictions.get(qid)
        if predicted_answer is None:
            missing_ids.append(qid)
            predicted_answer = ""

        candidate_answers = answers or [""]
        per_question[qid] = max(compute_f1(gold, predicted_answer) for gold in candidate_answers)

    extra_ids = sorted(set(predictions.keys()) - set(gold_answers.keys()))
    return per_question, missing_ids, extra_ids


def emit_warnings(label: str, missing_ids: Iterable[str], extra_ids: Iterable[str]) -> None:
    """Print consistent warnings for missing/extra predictions."""
    prefix = f"[{label}] " if label else ""
    missing_list = list(missing_ids)
    extra_list = list(extra_ids)
    if missing_list:
        print(f"{prefix}Warning: missing predictions for {len(missing_list)} ids.", file=sys.stderr)
    if extra_list:
        print(f"{prefix}Warning: {len(extra_list)} predictions did not match any dataset ids.", file=sys.stderr)
