import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Directories
base_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal'
cache_dir = os.path.join(base_dir, '.cache/huggingface')
result_dir = os.path.join(base_dir, 'result')
data_dir = os.path.join(base_dir, 'data')

# Model settings
batch_size = 8
max_chars = 20000  # Truncate extremely long inputs to prevent OOM
truncate_dim = 512 # Matryoshka embedding dimension

# Setup environment
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

# --- 1. Load Data ---
print("Loading input data from CSV files...")
benchmarks = ['scicode', 'gaia', 'taubench_airline', 'scienceagentbench', 'corebench_hard', 'assistantbench',
              'usaco', 'online_mind2web', 'swebench_verified_mini', 'colbench_backend_programming']
dfs = []

for benchmark in benchmarks:
    csv_file = f"{data_dir}/{benchmark}_inputs.csv"
    if os.path.exists(csv_file):
        temp_df = pd.read_csv(csv_file)
        temp_df['benchmark'] = benchmark
        dfs.append(temp_df)
    else:
        print(f"  ⚠️  Warning: {csv_file} not found")

if not dfs:
    print("❌ No input files found!")
    exit(1)

# Combine and Clean
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} total task inputs.")

if 'task_id' not in df.columns or 'text_input' not in df.columns:
    print("❌ Error: CSV files must have 'task_id' and 'text_input' columns")
    exit(1)

# --- 2. Pre-process Text ---
print("Pre-processing texts (truncating)...")
# Ensure inputs are strings and truncate to max_chars to prevent OOM errors
# (Qwen supports 32k tokens, but 20k chars is a safe limit for the 4B model)
texts = df['text_input'].astype(str).tolist()
truncated_texts = [t[:max_chars] for t in texts]

# --- 3. Embed ---
print("Loading model...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", cache_folder=cache_dir, trust_remote_code=True)

print(f"Generating embeddings (Batch size: {batch_size})...")
try:
    # We rely on the library's internal batching now since we don't need to resume
    embeddings = model.encode(
        truncated_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        normalize_embeddings=False,
        truncate_dim=truncate_dim 
    )
except Exception as e:
    print(f"❌ Error during embedding: {e}")
    exit(1)

# --- 4. Save Results ---
print("Formatting and saving results...")

# Assign embeddings back to dataframe
df['embedding'] = list(embeddings)

# Create unique ID key
df['benchmark.task_id'] = df['benchmark'] + '.' + df['task_id'].astype(str)

# Select and reorder columns
final_df = df[['benchmark.task_id', 'text_input', 'embedding']]

# Save
output_file = f"{result_dir}/all_benchmarks_embeddings.pkl"
final_df.to_pickle(output_file)

print(f"✅ Saved final result to: {output_file}")
print(f"   Shape: {final_df.shape}")
print(f"   Embedding Dim: {len(final_df.iloc[0]['embedding'])}")