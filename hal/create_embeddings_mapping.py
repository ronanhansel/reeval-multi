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

# Create (task_id, benchmark) to embedding mapping
# This is critical because task_ids are NOT globally unique - they repeat across benchmarks
task_bench_to_embedding = {}
for _, row in embeddings_df.iterrows():
    task_id = str(row['task_id'])
    benchmark = str(row['benchmark_id'])
    embedding = row['embeddings']
    key = (task_id, benchmark)
    
    # Use the first embedding if there are duplicates (shouldn't matter as they're the same task)
    if key not in task_bench_to_embedding:
        task_bench_to_embedding[key] = embedding

print(f"\nCreated embeddings mapping with {len(task_bench_to_embedding)} (task_id, benchmark) pairs")

# Handle the missing (task_id='73', benchmark='scicode') 
# This has empty text in resmat - we'll use a zero vector or find similar item
missing_key = ('73', 'scicode')

if missing_key not in task_bench_to_embedding:
    print(f"\n⚠️  Missing key: {missing_key}")
    
    # Check if there's a task_id='73' in other benchmarks
    task_73_keys = [k for k in task_bench_to_embedding.keys() if k[0] == '73']
    
    if task_73_keys:
        print(f"   Found task_id='73' in other benchmarks: {[k[1] for k in task_73_keys]}")
        # Use the first one found
        source_key = task_73_keys[0]
        task_bench_to_embedding[missing_key] = task_bench_to_embedding[source_key]
        print(f"   ✓ Copied embedding from {source_key} to {missing_key}")
    else:
        # Use zero vector as fallback
        embedding_dim = len(list(task_bench_to_embedding.values())[0])
        task_bench_to_embedding[missing_key] = np.zeros(embedding_dim, dtype=np.float32)
        print(f"   ✓ Using zero vector with dim={embedding_dim}")

# Create a DataFrame for easier lookup - one row per (task_id, benchmark) pair
print("\nCreating lookup structures...")

# Save as dictionary keyed by (task_id, benchmark)
with open('data/task_bench_to_embedding.pkl', 'wb') as f:
    pickle.dump(task_bench_to_embedding, f)

print(f"Created mapping with {len(task_bench_to_embedding)} (task_id, benchmark) pairs")

# Verify we can match with resmat
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

matched_count = 0
unmatched = []

for col in resmat_df.columns:
    task_id = str(col[0])
    benchmark = str(col[2])
    key = (task_id, benchmark)
    
    if key in task_bench_to_embedding:
        matched_count += 1
    else:
        unmatched.append(key)

print(f"Resmat columns: {len(resmat_df.columns)}")
print(f"Matched: {matched_count}/{len(resmat_df.columns)}")
print(f"Unmatched: {len(unmatched)}")

if unmatched:
    print(f"\n⚠️  Unmatched items:")
    for tid, bench in unmatched[:10]:
        print(f"  - (task_id={tid}, benchmark={bench})")

print(f"\n✅ Saved:")
print(f"  - data/task_bench_to_embedding.pkl (dict with {len(task_bench_to_embedding)} keys)")
