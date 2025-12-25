# Combining LADA joint results data
import os
import pandas as pd

# Define the path to the lada-fitting-joint results
results_path = "../result/lada-fitting-joint"

# List to store individual dataframes
dataframes = []

# Read each k-factor summary file
for k in [2, 3, 4]:
    csv_file = os.path.join(results_path, f"lada_joint_k{k}_summary.csv")
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        dataframes.append(df)
        print(f"Loaded {csv_file}: {len(df)} scenarios")
    else:
        print(f"Warning: {csv_file} not found")

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

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

# Show final data structure
print(f"\nFinal data shape: {df.shape}")
print(f"Available k-factors: {sorted(df['k_factor'].unique())}")
print("\nFirst 5 rows of final data:")
print(df.head())

# Save to pickle
output_path = '../data-reeval-multi/calibration_results_lada_joint.pkl'
df.to_pickle(output_path)
print(f"\nSaved to {output_path}")
