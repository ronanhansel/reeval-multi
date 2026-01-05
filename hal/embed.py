import pandas as pd
import numpy as np
import os
import glob
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
task_inputs = df['task_input'].tolist()
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

# Store embeddings back in the dataframe
df['embeddings'] = list(embeddings)

# Save to result directory
output_file = f"{result_dir}/all_benchmarks_with_embeddings.pkl"
df.to_pickle(output_file)
print(f"Saved final result to: {output_file}")

# Clean up temporary files
print("Cleaning up temporary files...")
import shutil
shutil.rmtree(temp_dir)
print(f"Removed temporary directory: {temp_dir}")

print("Done!")
