import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loader import load_model


def run_answer(context: str, question: str, model_name: str = "google/gemma-3-270m-it") -> str:

    system= """You are an extraction model. 
Answer with ONLY the exact text from the context. 
No introductions. No explanations. No extra words.
Your task is to answer the question based *only* on the provided context.

    Follow these rules strictly:
    1. The answer must be a direct quote or a concise phrase extracted verbatim from the context.
    2. Answer only based on the information within the text. Do not use any external knowledge.
    3. If the answer cannot be found in the context, respond with an empty string and nothing else.
    4. Do not add any explanation or preamble like "The answer is...". Provide only the extracted text.
    
    """

    prompt = f"""Context:
    {context}

    Question:
    {question}"""
    
    shots = [
        {
            "role": "user",
            "content": """Context: The Eiffel Tower is one of the most famous landmarks in the world. It was constructed in Paris, France for the 1889 World’s Fair and stands over 300 meters tall. Today, it attracts millions of visitors every year.

Question: In which city is the Eiffel Tower located?""",
        },
        {"role": "assistant", "content": "Paris"},
        {
            "role": "user",
            "content": """Context: Water boils at 100 degrees Celsius at sea level, but the boiling point decreases at higher altitudes because of lower air pressure. For example, in the city of Denver, which is about one mile above sea level, water boils at around 95 degrees Celsius.

Question: At what temperature does water boil at sea level?""",
        },
        {"role": "assistant", "content": "100 degrees Celsius"},
        {"role": "user",
    "content": """Context: The Wright brothers, Orville and Wilbur, are credited with inventing and flying the first successful powered airplane in 1903. Their historic flight took place in Kitty Hawk, North Carolina, which offered steady winds and soft sand for landing. This event marked the beginning of modern aviation and changed the course of human history.

Question: In which state did the Wright brothers' first flight take place?""",
},
{"role": "assistant", "content": "North Carolina"},
    ]

    response = load_model([prompt], system=system, shots=shots, model_name=model_name)
    if isinstance(response, (list, tuple)):
        result = response[0]
    else:
        result = response
    return str(result).strip()


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def load_dataset_rows(dataset_path: Path) -> List[dict]:
    with dataset_path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        return list(reader)


def read_existing_output(output_file: Path) -> List[dict]:
    if not output_file.exists():
        return []
    with output_file.open(encoding="utf-8") as existing_file:
        try:
            data = json.load(existing_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {output_file}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {output_file}.")
    return data


def write_output_rows(output_file: Path, rows: List[dict]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=output_file.parent,
        prefix=f"{output_file.name}.tmp.",
    ) as tmp_file:
        json.dump(rows, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = Path(tmp_file.name)
    os.replace(temp_path, output_file)


def update_progress(current: int, total: int, current_id: str, prefix: str = "") -> None:
    bar_width = 40
    filled = int(bar_width * current / max(total, 1))
    bar = "#" * filled + "-" * (bar_width - filled)
    prefix_str = f"{prefix} " if prefix else ""
    print(f"\r{prefix_str}[{bar}] {current}/{total} | id: {current_id}", end="", flush=True)
    if current >= total:
        print()


def _select_output_file(output_dir: Path, model_name: str, *, resume: bool, fresh_run: bool) -> Path:
    base_pattern = f"squad_{sanitize_model_name(model_name)}"
    pattern = f"{base_pattern}_*.json"
    candidates = [path for path in output_dir.glob(pattern) if path.is_file()]
    legacy_path = output_dir / f"{base_pattern}.json"
    if legacy_path.exists() and legacy_path.is_file():
        candidates.append(legacy_path)
    candidates.sort(key=lambda path: path.stat().st_mtime)

    if fresh_run or not candidates:
        if resume and not candidates:
            raise FileNotFoundError(
                f"No existing output files found for model '{model_name}' to resume from."
            )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return output_dir / f"{base_pattern}_{timestamp}.json"

    return candidates[-1]


def evaluate_dataset(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    resume_id: Optional[str] = None,
    fresh_run: bool = False,
) -> None:
    rows = load_dataset_rows(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = _select_output_file(
        output_dir,
        model_name,
        resume=resume_id is not None,
        fresh_run=fresh_run,
    )
    if output_file.exists():
        print(f"Resuming predictions using {output_file}")
    else:
        print(f"Writing predictions to {output_file}")

    rows_to_write = read_existing_output(output_file)
    start_index = len(rows_to_write)

    if resume_id is not None:
        matches = [idx for idx, row in enumerate(rows) if row["id"] == resume_id]
        if not matches:
            raise ValueError(f"resume_id '{resume_id}' not found in dataset.")
        resume_index = next((idx for idx in matches if idx >= start_index), matches[-1])
        if start_index > resume_index:
            rows_to_write = rows_to_write[:resume_index]
            write_output_rows(output_file, rows_to_write)
            start_index = resume_index
        elif start_index < resume_index:
            raise ValueError(
                f"Cannot resume from id '{resume_id}' because previous rows are incomplete "
                f"(have {start_index} completed, need {resume_index})."
            )

    existing_total = len(rows_to_write)
    if existing_total:
        print("Reading existing outputs to identify completed ids...")
        for index, row in enumerate(rows_to_write, start=1):
            current_id = row.get("id", "<unknown>")
            update_progress(index, existing_total, f"{current_id} (skipped)", prefix="Reading existing outputs:")

    start_index = existing_total

    if not output_file.exists():
        write_output_rows(output_file, rows_to_write)

    total = len(rows)
    completed = start_index

    if completed and total:
        update_progress(completed, total, rows_to_write[-1]["id"], prefix="Processing dataset:")

    if start_index >= len(rows):
        return

    for index in range(start_index, len(rows)):
        sample = rows[index]
        update_progress(completed, total, f"{sample['id']} (running)", prefix="Processing dataset:")
        answer = run_answer(sample["context"], sample["question"], model_name=model_name)
        rows_to_write.append({"id": sample["id"], "answer": answer})
        write_output_rows(output_file, rows_to_write)
        completed = len(rows_to_write)
        update_progress(completed, total, sample["id"], prefix="Processing dataset:")


def main() -> None:
    default_dataset = Path(__file__).parent / "data" / "squad_validation_lite.csv"
    parser = argparse.ArgumentParser(description="Run model answers over a dataset.")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Path to the validation dataset.")
    parser.add_argument("--model-name", default="google/gemma-3-270m-it", help="Model identifier to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Directory for prediction outputs.")
    parser.add_argument(
        "--resume-id",
        default=None,
        help="Optional dataset id to resume from; must already have prior rows completed.",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Force creation of a new output file instead of resuming any previous runs.",
    )
    args = parser.parse_args()

    evaluate_dataset(
        args.dataset,
        args.model_name,
        args.output_dir,
        resume_id=args.resume_id,
        fresh_run=args.fresh_run,
    )


if __name__ == "__main__":
    main()
