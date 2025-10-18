import pandas as pd
from pathlib import Path
import json


data_dir = Path('../../data-reeval-multi/lsat_qa')

# Get all JSON files in the directory
json_files = list(data_dir.glob('*.json'))

# Read and combine all JSON files
all_data = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        file_type = json_file.stem.split('_')[-1]
        for entry in data:
            entry['type'] = file_type
        all_data.extend(data)

# Convert to DataFrame
lsat_df = pd.DataFrame(all_data)

# Rename 'context' to 'input.text' to match the required format
lsat_df['input.text'] = lsat_df['context']

# Extract the correct answer from answers using label
lsat_df['answer'] = lsat_df.apply(
    lambda row: row['answers'][row['label']] if pd.notna(row['label']) and row['label'] < len(row['answers']) else None,
    axis=1
)

# Select only the required columns
lsat_df = lsat_df[['input.text', 'question', 'answer', 'type']]

# Save to pickle file
output_path = Path('./lsat_qa_combined.pkl')
output_path.parent.mkdir(parents=True, exist_ok=True)
lsat_df.to_pickle(output_path)

print(f"Successfully processed {len(lsat_df)} rows from {len(json_files)} JSON files")
print(f"Saved to {output_path}")
