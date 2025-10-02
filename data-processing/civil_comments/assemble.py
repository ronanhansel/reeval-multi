import pandas as pd
from datasets import load_dataset
import difflib
from collections import defaultdict

import sys
sys.path.append('../../')
from string_utils import *

dataset = load_dataset("google/civil_comments")
resmat = pd.read_pickle('../../data/resmat.pkl')
r_cv = resmat.loc[:, resmat.columns.get_level_values("scenario") == "civil_comments"]
r_cv.columns.get_level_values('input.text').tolist()

civil_texts = pd.Index(r_cv.columns.get_level_values('input.text')).unique()

norm_to_texts = defaultdict(list)
for original_text in civil_texts:
    normalized = normalize_text(original_text)
    if normalized:
        norm_to_texts[normalized].append(original_text)

target_norms = set(norm_to_texts.keys())
feature_columns = [feature for feature in dataset['train'].column_names if feature != 'text']

matched_records = {}
missing_texts = set(civil_texts)

# First pass: exact normalized matches
for split_name in dataset.keys():
    split = dataset[split_name]
    filtered = split.filter(lambda example: normalize_text(example['text']) in target_norms, load_from_cache_file=False)
    for row in filtered:
        normalized = normalize_text(row['text'])
        for candidate_text in norm_to_texts.get(normalized, []):
            if candidate_text not in matched_records:
                matched_records[candidate_text] = {feature: row[feature] for feature in feature_columns}
                missing_texts.discard(candidate_text)
    if not missing_texts:
        break

approx_match_count = 0
approx_records = {}

# Second pass: approximate matches via leading-word keys and high similarity
if missing_texts:
    target_keys = {make_key(text) for text in missing_texts}
    key_to_candidates: dict[str, list[dict]] = defaultdict(list)
    for split_name in dataset.keys():
        split = dataset[split_name]
        for row in split:
            key = make_key(row['text'])
            if key in target_keys:
                key_to_candidates[key].append({
                    "clean": clean_text(row['text']),
                    "text": row['text'],
                    "features": {feature: row[feature] for feature in feature_columns}
                })

    similarity_threshold = 0.83
    for original_text in list(missing_texts):
        key = make_key(original_text)
        candidates = key_to_candidates.get(key, [])
        target_clean = clean_text(original_text)
        best_entry = None
        best_ratio = 0.0
        for entry in candidates:
            ratio = difflib.SequenceMatcher(None, target_clean, entry["clean"]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_entry = entry
        approx_records[original_text] = {
            "best_ratio": best_ratio,
            "candidate_text": best_entry["text"] if best_entry else None
        }
        if best_entry and best_ratio >= similarity_threshold:
            matched_records[original_text] = best_entry["features"]
            missing_texts.discard(original_text)
            approx_match_count += 1

ordered_texts = [text for text in civil_texts if text in matched_records]
result_civil_comment = pd.DataFrame(matched_records, index=feature_columns)[ordered_texts]
result_civil_comment.to_pickle('./result_civil_comments.pkl')

missing_civil_comment = sorted(missing_texts)
print(f"Matched {result_civil_comment.shape[1]} of {len(civil_texts)} civil comment prompts.")
if approx_match_count:
    print(f"  - Added {approx_match_count} matches via approximate string similarity.")
if missing_civil_comment:
    print(f"Unmatched prompts: {len(missing_civil_comment)}")

unmatched_overview = pd.DataFrame([
    {
        "preview": text[:160].replace("\n", " "),
        "best_ratio": round(approx_records.get(text, {}).get("best_ratio", 0.0), 3),
        "candidate_preview": ((approx_records.get(text, {}).get("candidate_text") or "")[:160].replace("\n", " ")
        )
    }
    for text in missing_civil_comment[:10]
])
unmatched_overview