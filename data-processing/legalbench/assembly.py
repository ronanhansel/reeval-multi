import pandas as pd
from pathlib import Path
from typing import Iterable


def _join_parts(parts: Iterable[str]) -> str:
    cleaned = [str(part).strip() for part in parts if pd.notna(part) and str(part).strip()]
    return "\n\n".join(cleaned)

data_dir = Path('../../data-reeval-multi/legalbench')

# Abercrombie dataset
abercrombie_df = pd.read_csv(data_dir / 'data_abercrombie_test.txt', sep='\t')
abercrombie_df = abercrombie_df.rename(columns={'text': 'input.text'})
abercrombie_df['type'] = 'abercrombie'
abercrombie_df = abercrombie_df[['input.text', 'answer', 'type']]

# Corporate lobbying dataset
corporate_df = pd.read_csv(data_dir / 'data_corporate_lobbying_test.txt', sep='\t')
corporate_df['input.text'] = corporate_df.apply(
    lambda row: _join_parts([row.get('bill_title'), row.get('bill_summary'), row.get('company_name'), row.get('company_description')]),
    axis=1
)
corporate_df['type'] = 'corporate_lobbying'
corporate_df = corporate_df[['input.text', 'answer', 'type']]

# Function of decision section dataset
function_df = pd.read_csv(data_dir / 'data_function_of_decision_section_test.txt', sep='\t')
function_df['input.text'] = function_df.apply(
    lambda row: _join_parts([row.get('Paragraph'), f"Citation: {row.get('Citation')}" if pd.notna(row.get('Citation')) else None]),
    axis=1
)
function_df['type'] = 'function_of_decision_section'
function_df = function_df[['input.text', 'answer', 'type']]

# International citizenship dataset
international_df = pd.read_csv(data_dir / 'data_international_citizenship_questions_test.txt', sep='\t')
international_df = international_df.rename(columns={'question': 'input.text'})
international_df['type'] = 'international_citizenship'
international_df = international_df[['input.text', 'answer', 'type']]

# PROA dataset
proa_df = pd.read_csv(data_dir / 'data_proa_test.txt', sep='\t')
proa_df = proa_df.rename(columns={'text': 'input.text'})
proa_df['type'] = 'proa'
proa_df = proa_df[['input.text', 'answer', 'type']]

combined_df = pd.concat([
    abercrombie_df,
    corporate_df,
    function_df,
    international_df,
    proa_df
], ignore_index=True)

output_path = data_dir / 'legalbench_combined.pkl'
output_path.parent.mkdir(parents=True, exist_ok=True)
combined_df.to_pickle(output_path)