import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Setup directories
cache_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal/.cache/huggingface'
result_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal/result'

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

print("Loading resmat_binary_success_rate to get unique items...")
with open('data/resmat_binary_success_rate.pkl', 'rb') as f:
    resmat_df = pickle.load(f)

print(f"Resmat shape: {resmat_df.shape}")
print(f"Resmat columns (first 5): {resmat_df.columns[:5].tolist()}")
print(f"Total unique items in resmat: {len(resmat_df.columns)}")

# Extract unique task_ids from resmat columns
# Columns are multiindex tuples like (task_id, text_input, benchmark)
# task_id is the first level
resmat_task_ids = resmat_df.columns.get_level_values(0).unique().tolist()
unique_resmat_task_ids = resmat_task_ids

print(f"Unique task_ids in resmat (level 0): {len(unique_resmat_task_ids)}")
print(f"Sample task_ids: {unique_resmat_task_ids[:10]}")

# Also show total columns for clarity
print(f"Total resmat columns (full tuples): {len(resmat_df.columns)}")

# Load existing embeddings
print("\nLoading existing embeddings...")
embeddings_file = f"{result_dir}/all_benchmarks_with_embeddings.pkl"
if os.path.exists(embeddings_file):
    embeddings_df = pd.read_pickle(embeddings_file)
    print(f"Existing embeddings shape: {embeddings_df.shape}")
    print(f"Columns: {embeddings_df.columns.tolist()}")
    
    # Check what task_ids are already embedded
    if 'task_id' in embeddings_df.columns:
        existing_task_ids = set(embeddings_df['task_id'].unique())
        print(f"Existing task_ids with embeddings: {len(existing_task_ids)}")
    else:
        print("WARNING: No 'task_id' column found in embeddings!")
        existing_task_ids = set()
else:
    print("No existing embeddings file found!")
    embeddings_df = None
    existing_task_ids = set()

# Find missing task_ids
missing_task_ids = set(unique_resmat_task_ids) - existing_task_ids
print(f"\nMissing task_ids that need embedding: {len(missing_task_ids)}")
if len(missing_task_ids) <= 20:
    print(f"Missing task_ids: {sorted(missing_task_ids)}")
else:
    print(f"Sample missing task_ids: {sorted(list(missing_task_ids))[:10]}")

# Load the input data to find the task_input for missing task_ids
print("\nLoading all_benchmarks_inputs to find task_input for missing items...")
inputs_df = pd.read_pickle('data/all_benchmarks_inputs.pkl')
print(f"Inputs shape: {inputs_df.shape}")
print(f"Inputs columns: {inputs_df.columns.tolist()}")

# Check if we need to embed anything
if len(missing_task_ids) == 0:
    print("\n✓ All task_ids already have embeddings!")
else:
    # Filter inputs_df to only missing task_ids
    if 'task_id' in inputs_df.columns:
        missing_inputs_df = inputs_df[inputs_df['task_id'].isin(missing_task_ids)]
    else:
        # Try to match by index or other methods
        print("Warning: task_id column not found in inputs_df")
        print("Inputs columns:", inputs_df.columns.tolist())
        print("Inputs index:", inputs_df.index[:5])
        missing_inputs_df = pd.DataFrame()
    
    print(f"\nFound {len(missing_inputs_df)} inputs to embed")
    
    if len(missing_inputs_df) > 0:
        print("Loading embedding model...")
        model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", cache_folder=cache_dir)
        
        task_inputs = missing_inputs_df['task_input'].tolist()
        task_ids = missing_inputs_df['task_id'].tolist()
        
        print(f"Embedding {len(task_inputs)} missing task inputs...")
        
        # Process in small batches
        batch_size = 4
        all_new_embeddings = []
        
        for i in tqdm(range(0, len(task_inputs), batch_size)):
            batch = task_inputs[i:i+batch_size]
            
            # Truncate extremely long inputs
            MAX_CHARS = 20000
            truncated_batch = [str(text)[:MAX_CHARS] for text in batch]
            
            try:
                batch_embeddings = model.encode(
                    truncated_batch, 
                    show_progress_bar=False, 
                    convert_to_numpy=True,
                    device='cuda',
                    batch_size=batch_size,
                    normalize_embeddings=False
                )
                all_new_embeddings.append(batch_embeddings)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error on batch {i}: {e}")
                raise
        
        # Concatenate all new embeddings
        new_embeddings = np.vstack(all_new_embeddings)
        print(f"Generated {len(new_embeddings)} new embeddings")
        
        # Add to missing_inputs_df
        missing_inputs_df['embeddings'] = list(new_embeddings)
        
        # Combine with existing embeddings
        if embeddings_df is not None:
            combined_df = pd.concat([embeddings_df, missing_inputs_df], ignore_index=True)
        else:
            combined_df = missing_inputs_df
        
        # Save combined embeddings
        combined_file = f"{result_dir}/all_benchmarks_with_embeddings_complete.pkl"
        combined_df.to_pickle(combined_file)
        print(f"\nSaved complete embeddings to: {combined_file}")
    else:
        print("\nWarning: Could not find inputs for missing task_ids")
        print("This might indicate a mismatch in task_id format between resmat and inputs")

# Now merge embeddings into resmat_binary with additional multiindex level
print("\n" + "="*60)
print("Merging embeddings into resmat_binary_success_rate...")
print("="*60)

# Load the complete embeddings
if os.path.exists(f"{result_dir}/all_benchmarks_with_embeddings_complete.pkl"):
    complete_embeddings = pd.read_pickle(f"{result_dir}/all_benchmarks_with_embeddings_complete.pkl")
else:
    complete_embeddings = embeddings_df

# Create a mapping from task_id to embedding
task_id_to_embedding = {}
if complete_embeddings is not None and 'task_id' in complete_embeddings.columns:
    for idx, row in complete_embeddings.iterrows():
        task_id = row['task_id']
        embedding = row['embeddings']
        task_id_to_embedding[task_id] = embedding

print(f"Created mapping for {len(task_id_to_embedding)} task_ids")

# Now create new resmat with embeddings in multiindex
# Current multiindex structure is assumed to be something like (task_id, agent, model, benchmark)
# We want to add 'embedding' as a 4th level

# First, let's understand current column structure
print(f"\nCurrent resmat column structure:")
print(f"Column type: {type(resmat_df.columns)}")
if isinstance(resmat_df.columns, pd.MultiIndex):
    print(f"MultiIndex levels: {resmat_df.columns.nlevels}")
    print(f"MultiIndex names: {resmat_df.columns.names}")
else:
    print(f"Sample columns: {resmat_df.columns[:5].tolist()}")

# Create new columns with embeddings
new_columns = []
for col in resmat_df.columns:
    if isinstance(col, tuple):
        task_id = col[0]
        embedding = task_id_to_embedding.get(task_id, None)
        # Add embedding as a new level in the tuple
        new_col = col + (embedding,)
        new_columns.append(new_col)
    else:
        embedding = task_id_to_embedding.get(col, None)
        new_columns.append((col, embedding))

# Create new DataFrame with extended multiindex
if isinstance(resmat_df.columns, pd.MultiIndex):
    # Extend the existing multiindex names
    new_names = list(resmat_df.columns.names) + ['embedding']
else:
    # Create multiindex from scratch
    new_names = ['task_id', 'embedding']

# Create the new multiindex
new_multiindex = pd.MultiIndex.from_tuples(new_columns, names=new_names)

# Create new DataFrame with the new multiindex
resmat_with_embeddings = pd.DataFrame(
    resmat_df.values,
    index=resmat_df.index,
    columns=new_multiindex
)

print(f"\nNew resmat shape: {resmat_with_embeddings.shape}")
print(f"New column multiindex levels: {resmat_with_embeddings.columns.nlevels}")
print(f"New column names: {resmat_with_embeddings.columns.names}")

# Save the new resmat
output_file = 'data/resmat_binary_success_rate_with_embeddings.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(resmat_with_embeddings, f)

print(f"\nSaved resmat with embeddings to: {output_file}")

# Show summary statistics
non_null_embeddings = sum(1 for col in new_columns if col[-1] is not None)
print(f"\nSummary:")
print(f"Total columns: {len(new_columns)}")
print(f"Columns with embeddings: {non_null_embeddings}")
print(f"Columns without embeddings: {len(new_columns) - non_null_embeddings}")

print("\nDone!")
