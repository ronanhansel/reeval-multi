# Combining data
import os
import glob
import re
import pandas as pd

history_prefix = "training_history_"

# Define the path to the mirt-fitting results
results_path = "../result/mirt-fitting"

# Get all CSV files starting with the history prefix and containing any k-factor
csv_pattern = os.path.join(results_path, f"{history_prefix}*_k*.csv")
csv_files = glob.glob(csv_pattern)

print(f"Found {len(csv_files)} CSV files with prefix '{history_prefix}' and k-factors")

# List to store individual dataframes
dataframes = []

# Read each CSV file and add scenario column
for csv_file in csv_files:
    # Extract filename without path
    filename = os.path.basename(csv_file)
    
    # Extract k-factor from filename
    k_match = re.search(r'_k(\d+)', filename)
    if not k_match:
        continue  # Skip files without k-factor
    
    k_factor = int(k_match.group(1))
    
    # Extract scenario name: remove prefix and k-factor suffix
    scenario = filename.replace(history_prefix, "").replace(".csv", "")
    scenario = re.sub(r'_k\d+$', '', scenario)  # Remove _k followed by digits
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Add scenario and k_factor columns
    df['scenario'] = scenario
    df['k_factor'] = k_factor
    
    # Add to list
    dataframes.append(df)
    
# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

print(f"\nCombined DataFrame shape: {combined_df.shape}")
print(f"Unique scenarios: {combined_df['scenario'].nunique()}")
print(f"Available k-factors: {sorted(combined_df['k_factor'].unique())}")
print(f"Scenarios: {sorted(combined_df['scenario'].unique())}")

# Display first few rows
print("\nFirst 5 rows of combined data:")
combined_df.head()
combined_df.groupby('scenario').size()
# Check what k-factors we have in the raw data
print("K-factors in raw combined_df:")
print(f"Available k-factors: {sorted(combined_df['k_factor'].unique())}")
print(f"Count by k-factor:")
print(combined_df['k_factor'].value_counts().sort_index())
print(f"\nScenarios by k-factor:")
for k in sorted(combined_df['k_factor'].unique()):
    scenarios = combined_df[combined_df['k_factor'] == k]['scenario'].nunique()
    print(f"k={k}: {scenarios} scenarios")
# Group by both scenario and k_factor, then take max values for each combination
combined_df = combined_df.groupby(['scenario', 'k_factor']).max().reset_index()

print(f"After groupby - Shape: {combined_df.shape}")
print(f"Available k-factors: {sorted(combined_df['k_factor'].unique())}")
print(f"Scenarios per k-factor:")
print(combined_df.groupby('k_factor')['scenario'].nunique())
# Index is already reset in the previous cell
combined_df.head()
# Create DataFrame with the model results
results_data = {
    'scenario': [
        'lsat_qa', 'truthful_qa', 'synthetic_reasoning', 'babi_qa', 'wikifact',
        'bbq', 'thai_exam', 'dyck_language_np=3', 'legal_support', 'civil_comments',
        'legalbench', 'raft', 'air_bench_2024', 'math', 'med_qa', 'gsm',
        'boolq', 'mmlu', 'entity_matching', 'entity_data_imputation',
        'commonsense', 'imdb', 'combined_data'
    ],
    'train_auc': [
        0.688523530960083, 0.7754234075546265, 0.8787771463394165, 0.8385953307151794, 0.8949015140533447,
        0.7257800102233887, 0.857973575592041, 0.7987954020500183, 0.7143356800079346, 0.7923104763031006,
        0.8490860462188721, 0.8433337211608887, 0.9189774990081787, 0.9069100618362427, 0.8781333565711975,
        0.9062897562980652, 0.8516445159912109, 0.9107261896133423, 0.9005221128463745, 0.9436786770820618,
        0.9296308755874634, 0.9275050163269043, 0.8401544094085693
    ],
    'test_auc': [
        0.6194833517074585, 0.752168595790863, 0.8685898780822754, 0.825733482837677, 0.8836542367935181,
        0.6776220798492432, 0.8284746408462524, 0.7705100774765015, 0.6697757244110107, 0.774608314037323,
        0.836212158203125, 0.8359396457672119, 0.9038318991661072, 0.8989916443824768, 0.869013786315918,
        0.8985381722450256, 0.833503007888794, 0.8993809223175049, 0.8869590759277344, 0.9344797134399414,
        0.9203937649726868, 0.8829501271247864, 0.8265177607536316
    ],
    'train_cttcorr': [
        0.9999999954797291, 0.9999999890694462, 0.9999999995909886, 0.9999999976074504, 0.9999999986407333,
        0.9999999872610216, 0.9999999915025175, 0.99999999921584, 0.9999999976040401, 0.9999999982895018,
        0.9999999853873718, 0.9999999898960499, 0.9999999965323432, 0.9999999994130362, 0.9999999955530873,
        0.9999999995129949, 0.9999999996984892, 0.999999994816744, 0.9999999996850908, 0.9999999995785793,
        0.9999999981061295, 0.9999999868962981, 0.9999982965698357
    ],
    'test_cttcorr': [
        0.5906119125422994, 0.9706165826374094, 0.9944507503020064, 0.9801349393837014, 0.9955563464912967,
        0.9853548320237024, 0.9473795683343597, 0.9523431791191582, 0.8647175288487944, 0.9993321449878049,
        0.9853292991185817, 0.9667072922095117, 0.9979504514385135, 0.9904288659823243, 0.9823410596225498,
        0.9927657682987716, 0.993096924815468, 0.9973090123780943, 0.996777295569847, 0.96432828871636,
        0.9816468838401071, 0.9922439784181618, 0.9927450093359975
    ]
}

# Create the results DataFrame
results_df = pd.DataFrame(results_data)
resmat = pd.read_pickle('../data-reeval-multi/resmat.pkl')
counts_dict = resmat.columns.get_level_values('scenario').value_counts().dropna()
scenario_sizes = {fruit: count for fruit, count in counts_dict.items() if count > 0}
scenario_sizes


# Create a DataFrame with scenario sizes
size_df = pd.DataFrame(list(scenario_sizes.items()), columns=['scenario', 'size'])
print("Scenario sizes:")
print(size_df.sort_values('size', ascending=False))
# Merge the results DataFrame with the combined training history DataFrame
merged_df = combined_df.merge(results_df, on='scenario', how='left')

# Also merge with size information
merged_df = merged_df.merge(size_df, on='scenario', how='left')

print(f"Merged DataFrame shape: {merged_df.shape}")
print(f"Columns: {list(merged_df.columns)}")

# Check for any scenarios that didn't match
scenarios_in_combined = set(combined_df['scenario'].unique())
scenarios_in_results = set(results_df['scenario'].unique())
scenarios_in_sizes = set(size_df['scenario'].unique())

print(f"\nScenarios in combined_df: {len(scenarios_in_combined)}")
print(f"Scenarios in results_df: {len(scenarios_in_results)}")
print(f"Scenarios in size_df: {len(scenarios_in_sizes)}")
print(f"Scenarios in all: {len(scenarios_in_combined.intersection(scenarios_in_results).intersection(scenarios_in_sizes))}")

if scenarios_in_combined - scenarios_in_results:
    print(f"Scenarios in combined_df but not in results_df: {scenarios_in_combined - scenarios_in_results}")
if scenarios_in_results - scenarios_in_combined:
    print(f"Scenarios in results_df but not in combined_df: {scenarios_in_results - scenarios_in_combined}")
if scenarios_in_combined - scenarios_in_sizes:
    print(f"Scenarios in combined_df but not in size_df: {scenarios_in_combined - scenarios_in_sizes}")
# Prepare data for plotting - include k_factor information
df = merged_df[['val_auc', 'test_auc', 'scenario', 'k_factor', 'size']].sort_values(by='test_auc', ascending=True).dropna().copy()
df.rename(columns={'val_auc': 'mirt', 'test_auc': 'irt'}, inplace=True)

# Show unique k-factors available
print(f"Available k-factors: {sorted(df['k_factor'].unique())}")
print(f"Data shape: {df.shape}")
df.head()
df.to_pickle('../data-reeval-multi/calibration_results.pkl')