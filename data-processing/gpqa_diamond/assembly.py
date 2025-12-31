import pandas as pd
import numpy as np
from pathlib import Path
import os

def process_csv_file(csv_path):
    """
    Process a single CSV file and return aggregated scores by question ID.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with ID, Input as index and average score as values
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out empty rows (where ID is empty or NaN)
    df = df[df['ID'].notna() & (df['ID'] != '')]
    
    # Map C (correct) to 1, I (incorrect) to 0
    df['score_binary'] = df['Score'].map({'C': 1, 'I': 0})
    
    # Drop rows where Score wasn't 'C' or 'I'
    df = df[df['score_binary'].notna()]
    
    # Group by ID and Input, calculate average score
    grouped = df.groupby(['ID', 'Input'])['score_binary'].mean()
    
    return grouped


def create_response_matrix(traces_folder):
    """
    Create a response matrix from all CSV files in the traces folder.
    
    Args:
        traces_folder: Path to folder containing CSV trace files
    
    Returns:
        DataFrame with models as rows and (ID, Input) as multi-index columns
    """
    traces_path = Path(traces_folder)
    
    # Get all CSV files
    csv_files = sorted(traces_path.glob('*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {traces_folder}")
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Dictionary to store results for each model
    model_results = {}
    
    # First pass: collect all unique question IDs across all files
    all_question_ids = set()
    
    # Process each CSV file
    for csv_file in csv_files:
        model_name = csv_file.stem  # Get filename without extension
        print(f"Processing: {model_name}")
        
        try:
            scores = process_csv_file(csv_file)
            model_results[model_name] = scores
            # Collect all question IDs from this model
            all_question_ids.update(scores.index)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    if not model_results:
        raise ValueError("No valid data extracted from CSV files")
    
    # Create a list of all unique question IDs (sorted)
    all_question_ids = sorted(list(all_question_ids))
    
    # Build response matrix preserving NULLs for unanswered questions
    resmat_dict = {}
    for model_name, scores in model_results.items():
        # Create a series with all question IDs, filling with NaN where not present
        model_row = pd.Series(index=all_question_ids, dtype=float)
        # Fill in the scores that exist for this model
        model_row.update(scores)
        resmat_dict[model_name] = model_row
    
    # Create DataFrame from dictionary (models as rows, questions as columns)
    resmat = pd.DataFrame(resmat_dict).T
    
    # Sort columns by ID
    resmat = resmat.sort_index(axis=1)
    
    # Sort rows (models) alphabetically
    resmat = resmat.sort_index(axis=0)
    
    print(f"\nResponse matrix shape: {resmat.shape}")
    print(f"Models: {resmat.shape[0]}")
    print(f"Questions: {resmat.shape[1]}")
    
    return resmat


def main():
    """Main function to create and save the response matrix."""
    
    # Define paths
    traces_folder = "../../data-reeval-multi/gpqa_diamond/traces"
    output_folder = "../../data-reeval-multi/gpqa_diamond"
    
    # Create response matrix
    print("Creating response matrix...")
    resmat = create_response_matrix(traces_folder)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the response matrix
    output_path = os.path.join(output_folder, "resmat.pkl")
    resmat.to_pickle(output_path)
    print(f"\nResponse matrix saved to: {output_path}")
    
    # Display some statistics
    print("\n=== Statistics ===")
    print(f"Total models: {len(resmat)}")
    print(f"Total questions: {len(resmat.columns)}")
    print(f"\nMissing values: {resmat.isna().sum().sum()}")
    print(f"\nAverage scores across all models:")
    print(f"  Mean: {resmat.mean(axis=1).mean():.3f}")
    print(f"  Std:  {resmat.mean(axis=1).std():.3f}")
    
    # Show first few rows and columns
    print("\n=== Preview (first 5 models, first 3 questions) ===")
    print(resmat.iloc[:5, :3])
    
    return resmat


if __name__ == "__main__":
    resmat = main()
