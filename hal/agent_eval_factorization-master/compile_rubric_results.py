import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from util.rename_helper import standardize_task_success_column, clean_model_name, clean_rubric_name

def compile_rubric_results_by_benchmark(directory, benchmark_name, save=True):
    rubric_files = glob.glob(os.path.join(directory, f"{benchmark_name}_*.csv"))
    if not rubric_files:
        print(f"No files found for benchmark {benchmark_name}")
        return pd.DataFrame()  # Return an empty DataFrame if no files found
    merged = compile_file_list(rubric_files)
    if save:
        merged.to_csv(os.path.join(directory, f"{benchmark_name}_merged.csv"), index=False)
    return merged  

def get_rubric_type(file_name):
    return file_name.split('_')[-1].replace('.csv', '')

def compile_file_list(rubric_files):
    join_keys = ['benchmark_id', 'model', 'task_id', 'agent_run_id', "binary_success_rate"]

    dfs = []
    for rubric_file in rubric_files:
        df = pd.read_csv(rubric_file)
        
        # Standardize the success rate column name (currently different across benchmarks)
        df = standardize_task_success_column(df, benchmark_name=df['benchmark_id'].iloc[0])

        # Filter to keep relevant columns
        filtered_columns = join_keys + ['label']
        df = df[filtered_columns] # Note: A significant number of columns are dropped here
        
        # Rename rubric-specific columns to keep data organized
        rubric_type = get_rubric_type(rubric_file)
        rename_dict = {col: f"{rubric_type}.{col}" for col in df.columns if col not in join_keys}
        df = df.rename(columns=rename_dict)        
        dfs.append(df)

    # Outer join all dataframes
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=join_keys, how='outer')
    return merged

def main():
    benchmark_names = ["assistantbench","corebench","scicode", "taubench"]
    directory = "hal-paper-analysis/qualitative/results/rubrics"

    # # compile each benchmark into a single rubric file
    for benchmark_name in benchmark_names:
        benchmark_df = compile_rubric_results_by_benchmark(directory, benchmark_name)

    # Merge all benchmarks into a single file
    all_benchmark_files = glob.glob(os.path.join(directory, "*_merged.csv"))
    df_list = [pd.read_csv(rubric_file) for rubric_file in all_benchmark_files]
    all_merged = pd.concat(df_list, ignore_index=True)
    all_merged['model'] = all_merged['model'].apply(clean_model_name)
    all_merged.to_csv(os.path.join(directory, "all_benchmarks_merged.csv"), index=False)  
    return 


if __name__ == "__main__":
    main()