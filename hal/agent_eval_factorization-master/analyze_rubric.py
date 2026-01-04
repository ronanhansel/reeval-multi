import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from util.rename_helper import clean_rubric_name

def plot_matrix_single_rubric(df: pd.DataFrame, rubric_name: str):
    """
    Plot a heatmap for a single rubric across different models and tasks.
    """
    pivot_df = df.pivot( # Pivot: each model becomes a row, each task becomes a column
            index='model',
            columns='task_column',
            values=rubric_name
        )
    clean_pivot_df = pivot_df.map(lambda x: 1 if x == 'match' else (0 if x == 'no match' else x))
    clean_pivot_df = clean_pivot_df.map(lambda x: 1 if x == 'True' else (0 if x == 'False' else x))
    clean_pivot_df = clean_pivot_df.fillna(-1)  # Replace NaN with -1 for visualization
    
    # # Save the pivot table to a CSV file
    # clean_pivot_df.to_csv(f"{rubric_name}_pivot.csv")
    # print(f"Pivot table saved to {rubric_name}_pivot.csv")

    # Custom colormap: red (-1/NaN), white (0/failure), green (1/success)
    colors = ['red', 'white', 'dodgerblue']
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(20, 8))
    sns.heatmap(clean_pivot_df, annot=False, cmap=cmap, cbar=False,
                linewidths=0.5, 
                linecolor='gray',
                xticklabels=False,
                vmin=-1, vmax=1)  # Set range to include all three values

    clean_rubric_name_str = clean_rubric_name(rubric_name).capitalize()
    plt.title(f'{clean_rubric_name_str} Flag')
    plt.xlabel('Task ID')
    plt.ylabel('Models')
    plt.tight_layout()
    plt.savefig(f"plots/rubric/{rubric_name}_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to plots/rubric/{rubric_name}_matrix.png")
    plt.close()
    return

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot rubric matrices")
    parser.add_argument("dataset", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--plot_matrix_by_rubric", action="store_true",
                        help="Plot all rubric matrices")
    args = parser.parse_args()

    # Read the dataset (i.e. hal-paper-analysis/qualitative/results/rubrics/rubrics_merged/all_benchmarks_merged.csv)
    merged = pd.read_csv(args.dataset)
    merged['task_column'] = merged['benchmark_id'] + '.' + merged['task_id']
    print(summarized := merged.groupby('model').size())

    # plot all rubric matrices if flag is set
    if args.plot_matrix_by_rubric:
        rubric_names = ["selfcorrection.label","tooluse.label","environmentalbarrier.label","verification.label","instructionfollowing.label", "binary_success_rate"]
        for rubric in rubric_names:
            plot_matrix_single_rubric(merged, rubric)
    
    return


if __name__ == "__main__":
    main()