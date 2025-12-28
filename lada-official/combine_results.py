# Combining LADA joint results data
import os
import pandas as pd

# Define the path to the lada-fitting-joint results
results_path = "../result/lada-fitting-joint"

# List to store individual dataframes
dataframes = []

# Find all k-factor summary files in the directory
if os.path.exists(results_path):
    for filename in os.listdir(results_path):
        if filename.startswith("lada_joint_k") and filename.endswith("_summary.csv"):
            csv_file = os.path.join(results_path, filename)
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            print(f"Loaded {csv_file}: {len(df)} scenarios")
else:
    print(f"Warning: {results_path} not found")

# Combine all dataframes
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
else:
    print("No data files found!")
    exit()

print(f"\nCombined DataFrame shape: {combined_df.shape}")
print(f"Unique scenarios: {combined_df['scenario'].nunique()}")
print(f"Available k-factors: {sorted(combined_df['K'].unique())}")
print(f"Scenarios: {sorted(combined_df['scenario'].unique())}")

# Display first few rows
print("\nFirst 5 rows of combined data:")
print(combined_df.head())

# Group by scenario to see counts
print("\nScenarios by k-factor:")
for k in sorted(combined_df['K'].unique()):
    scenarios = combined_df[combined_df['K'] == k]['scenario'].nunique()
    print(f"K={k}: {scenarios} scenarios")

# Load scenario sizes from resmat
resmat = pd.read_pickle('../data-reeval-multi/resmat.pkl')
counts_dict = resmat.columns.get_level_values('scenario').value_counts().dropna()
scenario_sizes = {fruit: count for fruit, count in counts_dict.items() if count > 0}

# Create a DataFrame with scenario sizes
size_df = pd.DataFrame(list(scenario_sizes.items()), columns=['scenario', 'size'])
print("\nScenario sizes:")
print(size_df.sort_values('size', ascending=False))

# Merge with size information
merged_df = combined_df.merge(size_df, on='scenario', how='left')

print(f"\nMerged DataFrame shape: {merged_df.shape}")
print(f"Columns: {list(merged_df.columns)}")

# Check for any scenarios that didn't match
scenarios_in_combined = set(combined_df['scenario'].unique())
scenarios_in_sizes = set(size_df['scenario'].unique())

print(f"\nScenarios in combined_df: {len(scenarios_in_combined)}")
print(f"Scenarios in size_df: {len(scenarios_in_sizes)}")
print(f"Scenarios in both: {len(scenarios_in_combined.intersection(scenarios_in_sizes))}")

if scenarios_in_combined - scenarios_in_sizes:
    print(f"Scenarios in combined_df but not in size_df: {scenarios_in_combined - scenarios_in_sizes}")

# Prepare data for output - rename K to k_factor for consistency
df = merged_df.copy()
df.rename(columns={'K': 'k_factor'}, inplace=True)

# Add IRT baseline values (test_auc from MIRT results)
irt_data = {
    'scenario': [
        'lsat_qa', 'truthful_qa', 'synthetic_reasoning', 'babi_qa', 'wikifact',
        'bbq', 'thai_exam', 'dyck_language_np=3', 'legal_support', 'civil_comments',
        'legalbench', 'raft', 'air_bench_2024', 'math', 'med_qa', 'gsm',
        'boolq', 'mmlu', 'entity_matching', 'entity_data_imputation',
        'commonsense', 'imdb', 'combined_data'
    ],
    'irt': [
        0.6194833517074585, 0.752168595790863, 0.8685898780822754, 0.825733482837677, 0.8836542367935181,
        0.6776220798492432, 0.8284746408462524, 0.7705100774765015, 0.6697757244110107, 0.774608314037323,
        0.836212158203125, 0.8359396457672119, 0.9038318991661072, 0.8989916443824768, 0.869013786315918,
        0.8985381722450256, 0.833503007888794, 0.8993809223175049, 0.8869590759277344, 0.9344797134399414,
        0.9203937649726868, 0.8829501271247864, 0.8265177607536316
    ]
}

# Create IRT DataFrame and merge
irt_df = pd.DataFrame(irt_data)
df = df.merge(irt_df, on='scenario', how='left')

# Show final data structure
print(f"\nFinal data shape: {df.shape}")
print(f"Available k-factors: {sorted(df['k_factor'].unique())}")
print("\nFirst 5 rows of final data:")
print(df.head())
print("\nScenarios with IRT values:")
print(df.groupby('scenario')['irt'].first())

# Save to pickle
output_path = '../data-reeval-multi/calibration_results_lada_joint.pkl'
df.to_pickle(output_path)
print(f"\nSaved to {output_path}")
