import pandas as pd
import pickle

# Load resmat
with open('data/resmat_binary_success_rate.pkl', 'rb') as f:
    resmat_df = pickle.load(f)

# Get unique task_ids from resmat
resmat_task_ids = resmat_df.columns.get_level_values(0).unique().tolist()
print(f"Resmat unique task_ids: {len(resmat_task_ids)}")

# Load embeddings
embeddings_df = pd.read_pickle('result/all_benchmarks_with_embeddings.pkl')
embed_task_ids = embeddings_df['task_id'].unique().tolist()
print(f"Embeddings unique task_ids: {len(embed_task_ids)}")

# Find missing
missing = set(resmat_task_ids) - set(embed_task_ids)
print(f"\nMissing task_ids: {missing}")

# Find which rows in resmat have this task_id
if missing:
    missing_id = list(missing)[0]
    print(f"\n Columns in resmat with task_id '{missing_id}':")
    for col in resmat_df.columns:
        if col[0] == missing_id:
            print(f"  - Full tuple: {col}")
            print(f"    Text preview: {col[1][:200]}...")
            print(f"    Benchmark: {col[2]}")
            
    # Check if this task_id exists in all_benchmarks_inputs
    inputs_df = pd.read_pickle('data/all_benchmarks_inputs.pkl')
    print(f"\nChecking all_benchmarks_inputs.pkl...")
    print(f"  Total rows: {len(inputs_df)}")
    
    # Check task_id column
    if missing_id in inputs_df['task_id'].values:
        print(f"  ✓ Found task_id '{missing_id}' in inputs!")
        matching_rows = inputs_df[inputs_df['task_id'] == missing_id]
        print(f"  Matching rows: {len(matching_rows)}")
        for idx, row in matching_rows.iterrows():
            print(f"    Row {idx}:")
            print(f"      benchmark_id: {row['benchmark_id']}")
            print(f"      model: {row['model']}")
            print(f"      task_input preview: {str(row['task_input'])[:200]}...")
    else:
        print(f"  ✗ Task_id '{missing_id}' NOT found in inputs!")
        print(f"  This explains why it wasn't embedded!")
        
    # Check if maybe the text is in inputs under different task_id
    print(f"\nSearching for matching text in inputs...")
    resmat_text = col[1]
    for idx, row in inputs_df.iterrows():
        if str(row['task_input']) == resmat_text:
            print(f"  ✓ Found matching text at row {idx}!")
            print(f"    task_id in inputs: {row['task_id']}")
            print(f"    benchmark_id: {row['benchmark_id']}")
            break
    else:
        print(f"  ✗ No matching text found in inputs!")
