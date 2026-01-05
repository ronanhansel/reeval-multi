import pandas as pd
import pickle
import numpy as np

# Load data
with open('data/resmat_binary_success_rate.pkl', 'rb') as f:
    resmat_df = pickle.load(f)

embeddings_df = pd.read_pickle('result/all_benchmarks_with_embeddings.pkl')
inputs_df = pd.read_pickle('data/all_benchmarks_inputs.pkl')

print(f"Resmat columns: {resmat_df.shape[1]}")
print(f"Embeddings rows: {len(embeddings_df)}")

# Create task_id to embedding mapping
task_id_to_embedding = {}
for _, row in embeddings_df.iterrows():
    task_id = row['task_id']
    embedding = row['embeddings']
    task_id_to_embedding[task_id] = embedding

print(f"\nCreated embeddings mapping with {len(task_id_to_embedding)} task_ids")

# Handle the missing task_id='73' - map it to the correct hash
# We found that task_id '73' in resmat corresponds to 
# 'fb9ba3ab6a13d0adc677f993e90d54914a5cdf211305a1bba6bf16ec4ccb9b7c' in inputs
missing_task_id = '73'
correct_hash = 'fb9ba3ab6a13d0adc677f993e90d54914a5cdf211305a1bba6bf16ec4ccb9b7c'

if correct_hash in task_id_to_embedding:
    task_id_to_embedding[missing_task_id] = task_id_to_embedding[correct_hash]
    print(f"✓ Mapped task_id '{missing_task_id}' to embedding from '{correct_hash}'")
else:
    print(f"✗ Warning: Could not find embedding for '{correct_hash}'")
    # Use zero vector as fallback
    embedding_dim = len(list(task_id_to_embedding.values())[0])
    task_id_to_embedding[missing_task_id] = np.zeros(embedding_dim)
    print(f"  Using zero vector with dim={embedding_dim}")

# Create a DataFrame for easier merging - indexed by task_id
embeddings_array = []
task_ids = []

for task_id, embedding in task_id_to_embedding.items():
    task_ids.append(task_id)
    embeddings_array.append(embedding)

embeddings_lookup = pd.DataFrame({
    'task_id': task_ids,
    'embedding': embeddings_array
})
embeddings_lookup = embeddings_lookup.set_index('task_id')

print(f"\nCreated embeddings lookup DataFrame: {embeddings_lookup.shape}")
print(f"  Index (task_ids): {len(embeddings_lookup)}")
print(f"  Embedding dimension: {len(embeddings_lookup.iloc[0]['embedding'])}")

# Verify we can match with resmat
resmat_task_ids = resmat_df.columns.get_level_values(0).unique()
matched = sum(1 for tid in resmat_task_ids if tid in embeddings_lookup.index)
print(f"\nMatching check:")
print(f"  Resmat unique task_ids: {len(resmat_task_ids)}")
print(f"  Matched in embeddings: {matched}/{len(resmat_task_ids)}")

# Save the mapping
with open('data/task_id_to_embedding.pkl', 'wb') as f:
    pickle.dump(task_id_to_embedding, f)
    
embeddings_lookup.to_pickle('data/embeddings_lookup.pkl')

print(f"\n✓ Saved:")
print(f"  - data/task_id_to_embedding.pkl (dict)")
print(f"  - data/embeddings_lookup.pkl (DataFrame)")
