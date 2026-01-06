import pandas as pd
import numpy as np
import os
import glob
import torch
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description='Embed task inputs with optional resume capability')
parser.add_argument('--check', type=str, default=None, 
                    help='Path to resmat pickle file to check which items need embedding')
args = parser.parse_args()

# Setup directories
cache_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal/.cache/huggingface'
temp_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal/temp'
result_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal/result'

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

# Load input data
print("Loading input data...")
df = pd.read_pickle('data/all_benchmarks_inputs.pkl')

# Check for existing embeddings (resume capability)
output_file = f"{result_dir}/all_benchmarks_with_embeddings.pkl"
existing_embeddings_df = None

if os.path.exists(output_file):
    print(f"Found existing embeddings file: {output_file}")
    existing_embeddings_df = pd.read_pickle(output_file)
    print(f"  - Existing embeddings: {len(existing_embeddings_df)} rows")
    
    # Check if embeddings column exists
    if 'embeddings' in existing_embeddings_df.columns:
        # Count how many have embeddings
        has_embedding = existing_embeddings_df['embeddings'].notna().sum()
        print(f"  - Rows with embeddings: {has_embedding}")
    else:
        print(f"  - No 'embeddings' column found")
        existing_embeddings_df = None

# If --check flag is provided, compare with resmat to find what's needed
indices_to_process = None
if args.check:
    print(f"\n📋 Checking against resmat file: {args.check}")
    resmat_df = pd.read_pickle(args.check)
    
    # Extract unique (task_id, benchmark) pairs from resmat columns
    # Resmat columns are MultiIndex: (task_id, text_input, benchmark)
    resmat_items = set()
    for col in resmat_df.columns:
        task_id = col[0]
        benchmark = col[2]
        resmat_items.add((task_id, benchmark))
    
    print(f"  - Resmat has {len(resmat_items)} unique (task_id, benchmark) pairs")
    print(f"  - Resmat has {len(resmat_df.columns)} total columns")
    
    # Find which rows in df match these items
    # Create (task_id, benchmark) column in df for matching
    df['_temp_key'] = df.apply(lambda row: (str(row['task_id']), str(row['benchmark_id'])), axis=1)
    
    # Find indices that match resmat items
    matching_mask = df['_temp_key'].isin(resmat_items)
    indices_to_process = df[matching_mask].index.tolist()
    
    print(f"  - Found {len(indices_to_process)} rows in inputs matching resmat items")
    
    # If we have existing embeddings, filter out what's already done
    if existing_embeddings_df is not None and 'embeddings' in existing_embeddings_df.columns:
        # Check which of the matching indices already have embeddings
        already_embedded = existing_embeddings_df[
            (existing_embeddings_df['embeddings'].notna()) &
            (existing_embeddings_df.index.isin(indices_to_process))
        ].index.tolist()
        
        # Remove already embedded indices
        indices_to_process = [idx for idx in indices_to_process if idx not in already_embedded]
        
        print(f"  - After checking existing embeddings: {len(indices_to_process)} rows still need embedding")
        print(f"  - Already have embeddings: {len(already_embedded)} rows")
    
    df.drop('_temp_key', axis=1, inplace=True)
    
    if len(indices_to_process) == 0:
        print("\n✅ All required items already have embeddings!")
        print("Nothing to process. Exiting.")
        exit(0)

# Determine which rows to process
if indices_to_process is not None:
    # Process only specific indices
    print(f"\n🎯 Processing {len(indices_to_process)} specific rows")
    task_inputs = df.loc[indices_to_process, 'task_input'].tolist()
    processing_indices = indices_to_process
else:
    # Process all rows (legacy behavior)
    print(f"\n📦 Processing all {len(df)} rows")
    task_inputs = df['task_input'].tolist()
    processing_indices = df.index.tolist()

total_samples = len(task_inputs)

# Configuration
batch_size = 8  # Reduced from 32 to prevent OOM
num_batches = (total_samples + batch_size - 1) // batch_size

print(f"Total samples: {total_samples}")
print(f"Batch size: {batch_size}")
print(f"Total batches: {num_batches}")

# Check for existing batch files and validate batch size (resume capability)
config_file = f"{temp_dir}/config.txt"
existing_batches = set()
batch_files = glob.glob(f"{temp_dir}/batch_*.npy")

if batch_files:
    # Check if batch size has changed
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            saved_batch_size = int(f.read().strip())
        
        if saved_batch_size != batch_size:
            print(f"⚠️  Batch size changed from {saved_batch_size} to {batch_size}")
            print(f"Clearing existing {len(batch_files)} batches and starting fresh...")
            import shutil
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        else:
            # Same batch size, can resume
            for batch_file in batch_files:
                batch_num = int(batch_file.split('_')[-1].split('.')[0])
                existing_batches.add(batch_num)
            print(f"Found {len(existing_batches)} existing batches. Resuming from where we left off...")
    else:
        # Config file missing, can't validate - clear to be safe
        print(f"Config file missing. Clearing existing batches and starting fresh...")
        import shutil
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
else:
    print("Starting fresh...")

# Save current batch size to config
with open(config_file, 'w') as f:
    f.write(str(batch_size))

# Load the model
print("Loading model...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", cache_folder=cache_dir)

# Process batches
for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
    # Skip if already processed
    if batch_idx in existing_batches:
        continue
    
    # Get batch data
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, total_samples)
    batch = task_inputs[start_idx:end_idx]
    
    # Truncate extremely long inputs to prevent OOM
    # Qwen3 models support ~32k tokens, but we'll limit to 20k chars (~5k tokens) to be safe
    MAX_CHARS = 20000
    truncated_batch = []
    for i, text in enumerate(batch):
        text_str = str(text)
        text_len = len(text_str)
        if text_len > MAX_CHARS:
            actual_idx = start_idx + i
            print(f"\n⚠️  Input {actual_idx}: {text_len:,} chars → truncating to {MAX_CHARS:,} chars")
            truncated_batch.append(text_str[:MAX_CHARS])
        else:
            truncated_batch.append(text_str)
    
    try:
        # Encode batch
        batch_embeddings = model.encode(
            truncated_batch, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            device='cuda',
            batch_size=batch_size,
            normalize_embeddings=False
        )
        
        # Save batch immediately to disk
        batch_file = f"{temp_dir}/batch_{batch_idx}.npy"
        np.save(batch_file, batch_embeddings)
        
        # Clear GPU cache to prevent memory accumulation
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ Error processing batch {batch_idx}: {e}")
        print(f"Input indices: {start_idx} to {end_idx-1}")
        for i, text in enumerate(batch):
            actual_idx = start_idx + i
            print(f"  - Input {actual_idx}: {len(str(text)):,} characters")
        print(f"Saved progress up to batch {batch_idx-1}")
        raise

print("\nAll batches processed. Assembling final results...")

# Load all batch files in order and assemble
all_embeddings = []
for batch_idx in tqdm(range(num_batches), desc="Loading batches"):
    batch_file = f"{temp_dir}/batch_{batch_idx}.npy"
    batch_embeddings = np.load(batch_file)
    all_embeddings.append(batch_embeddings)

# Concatenate all embeddings
embeddings = np.vstack(all_embeddings)

print(f"Embedded {len(embeddings)} task inputs")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Embeddings shape: {embeddings.shape}")

# Merge with original dataframe
print("\nMerging with original data...")

if existing_embeddings_df is not None and 'embeddings' in existing_embeddings_df.columns:
    # We're adding to existing embeddings
    print("  - Merging with existing embeddings...")
    
    # Create a new embeddings series for the rows we just processed
    new_embeddings = pd.Series(list(embeddings), index=processing_indices, name='embeddings')
    
    # Start with existing dataframe
    result_df = existing_embeddings_df.copy()
    
    # Update with new embeddings
    for idx, emb in zip(processing_indices, embeddings):
        if idx in result_df.index:
            result_df.loc[idx, 'embeddings'] = emb
        else:
            # This shouldn't happen, but handle gracefully
            print(f"  ⚠️  Warning: Index {idx} not found in existing dataframe")
    
    # Also merge any new rows from df that aren't in result_df
    missing_indices = df.index.difference(result_df.index)
    if len(missing_indices) > 0:
        print(f"  - Adding {len(missing_indices)} new rows from input data")
        missing_rows = df.loc[missing_indices].copy()
        missing_rows['embeddings'] = None
        result_df = pd.concat([result_df, missing_rows], ignore_index=False)
    
    final_df = result_df
    
else:
    # Fresh start - just add embeddings to df
    print("  - Creating fresh embeddings dataframe...")
    final_df = df.copy()
    final_df['embeddings'] = None
    
    # Add the embeddings we just computed
    for idx, emb in zip(processing_indices, embeddings):
        final_df.loc[idx, 'embeddings'] = emb

# Report statistics
has_embedding = final_df['embeddings'].notna().sum()
print(f"\n📊 Final statistics:")
print(f"  - Total rows: {len(final_df)}")
print(f"  - Rows with embeddings: {has_embedding}")
print(f"  - Rows without embeddings: {len(final_df) - has_embedding}")

# Save to result directory
final_df.to_pickle(output_file)
print(f"✅ Saved final result to: {output_file}")

# Clean up temporary files
print("Cleaning up temporary files...")
import shutil
shutil.rmtree(temp_dir)
print(f"Removed temporary directory: {temp_dir}")

print("Done!")
