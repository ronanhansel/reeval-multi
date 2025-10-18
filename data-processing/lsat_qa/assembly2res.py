import re
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path
import pandas as pd


data_dir = Path('../../data-reeval-multi/lsat_qa')

# Load resmat and keep only lsat_qa scenario columns
resmat = pd.read_pickle('../../data-reeval-multi/resmat.pkl')
if not isinstance(resmat.columns, pd.MultiIndex) or 'scenario' not in resmat.columns.names:
    raise ValueError('Expected resmat columns to be a MultiIndex with a "scenario" level.')

scenario_mask = resmat.columns.get_level_values('scenario') == 'lsat_qa'
resmat = resmat.loc[:, scenario_mask]

if resmat.empty:
    raise ValueError('No columns found for scenario "lsat_qa" in resmat.')

resmat_columns = resmat.columns
if isinstance(resmat_columns, pd.MultiIndex) and 'input.text' in resmat_columns.names:
    resmat_texts = resmat_columns.get_level_values('input.text')
else:
    resmat_texts = pd.Index(resmat_columns)

# Load combined dataset prepared by assembly.py
combined_path_candidates = [
    Path('./lsat_qa_combined.pkl'),
    data_dir / 'lsat_qa_combined.pkl'
]

combined_path = next((path for path in combined_path_candidates if path.exists()), None)
if combined_path is None:
    raise FileNotFoundError('Could not locate lsat_qa_combined.pkl. Run assembly.py first.')

combined_df = pd.read_pickle(combined_path)

# Build full prompt text to align with resmat formatting
combined_df['full_input'] = (
    combined_df['input.text'].fillna('').astype(str).str.strip() +
    '\nQuestion: ' + combined_df['question'].fillna('').astype(str).str.strip()
)

REMOVE_TOKENS = (
    'description', 'question', 'prompt', 'text', 'analysis', 'facts', 'fact',
    'issue', 'holding', 'conclusion', 'rule', 'citation', 'passage', 'context'
)


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text).lower()
    for token in REMOVE_TOKENS:
        text = text.replace(f'{token}:', ' ')
        text = text.replace(token, ' ')
    text = re.sub(r'\d+', ' ', text)
    return re.sub(r'[^a-z0-9]', '', text)


combined_df['normalized'] = combined_df['full_input'].map(normalize_text)
if 'type' not in combined_df.columns:
    combined_df['type'] = ''

normalized_map = combined_df.drop_duplicates('normalized').set_index('normalized')[['full_input', 'answer', 'type']]

if '' in normalized_map.index:
    normalized_map = normalized_map.drop(index='')

if not normalized_map.index.is_unique:
    raise ValueError('Duplicate normalized keys found in combined dataset.')

norm_keys = [key for key in normalized_map.index if key]

matches = {}
match_records = []
unmatched = {}

for idx, text in enumerate(resmat_texts):
    norm = normalize_text(text)
    if norm and norm in normalized_map.index:
        row = normalized_map.loc[norm]
        matches[idx] = {'answer': row['answer'], 'type': row['type']}
        match_records.append({
            'column_index': idx,
            'original_text': text,
            'matched_text': row['full_input'],
            'method': 'exact_alphanum',
            'score': 1.0
        })
    else:
        unmatched[idx] = {'norm': norm, 'original_text': text}

if unmatched:
    for cutoff in [0.98, 0.95, 0.9, 0.85, 0.8, 0.75]:
        if not unmatched:
            break
        for idx, info in list(unmatched.items()):
            candidate_key = info['norm'] or normalize_text(info['original_text'])
            if not candidate_key:
                continue
            candidates = get_close_matches(candidate_key, norm_keys, n=3, cutoff=cutoff)
            if candidates:
                cand = max(candidates, key=lambda key: SequenceMatcher(None, candidate_key, key).ratio())
                score = SequenceMatcher(None, candidate_key, cand).ratio()
                row = normalized_map.loc[cand]
                matches[idx] = {'answer': row['answer'], 'type': row['type']}
                match_records.append({
                    'column_index': idx,
                    'original_text': info['original_text'],
                    'matched_text': row['full_input'],
                    'method': f'fuzzy_{cutoff}',
                    'score': score
                })
                unmatched.pop(idx)

if unmatched:
    for idx, info in list(unmatched.items()):
        candidate_key = info['norm'] or normalize_text(info['original_text'])
        if not candidate_key:
            continue
        best_key = None
        best_score = 0
        for key in norm_keys:
            score = SequenceMatcher(None, candidate_key, key).ratio()
            if score > best_score:
                best_score = score
                best_key = key
        if best_key and best_score >= 0.65:
            row = normalized_map.loc[best_key]
            matches[idx] = {'answer': row['answer'], 'type': row['type']}
            match_records.append({
                'column_index': idx,
                'original_text': info['original_text'],
                'matched_text': row['full_input'],
                'method': f'fuzzy_best_{best_score:.2f}',
                'score': best_score
            })
            unmatched.pop(idx)

if unmatched:
    missing_count = len(unmatched)
    sample = list(unmatched.values())[:5]
    raise ValueError(f'Unmatched columns remaining ({missing_count}). Sample entries: {sample}')

answers = [matches[idx]['answer'] for idx in range(len(resmat_columns))]
types = [matches[idx]['type'] for idx in range(len(resmat_columns))]

result_resmat = pd.DataFrame(
    [answers, types],
    index=['answer', 'type'],
    columns=resmat_columns
)

match_summary = pd.DataFrame(match_records)
method_counts = match_summary['method'].value_counts()

print(f'Matched columns: {len(matches)} / {resmat.shape[1]}')
print('Match methods used:')
print(method_counts)

result_path = Path('./lsat_qa_result.pkl')
result_resmat.to_pickle(result_path)

print(f'Saved results to {result_path.resolve()}')
