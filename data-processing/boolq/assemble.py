import pandas as pd
import json
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read train.jsonl
train_data = []
with open(script_dir / 'train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        data['split'] = 'train'
        train_data.append(data)

print(f"Loaded {len(train_data)} train examples")

# Read dev.jsonl
dev_data = []
with open(script_dir / 'dev.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        data['split'] = 'dev'
        dev_data.append(data)

print(f"Loaded {len(dev_data)} dev examples")

# Combine into a single DataFrame
all_data = train_data + dev_data
df = pd.DataFrame(all_data)

print(f"\nTotal examples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDataFrame shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Save as pickle
output_path = script_dir / 'boolq.pkl'
df.to_pickle(output_path)
print(f"\nSaved DataFrame to {output_path}")
