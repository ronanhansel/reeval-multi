import pandas as pd
from pathlib import Path

def list_unique_models_from_leaderboard():
    """List all unique models from the 3rd column of each CSV file in the leaderboard folder."""
    
    # Path to the leaderboard folder
    leaderboard_dir = Path(__file__).parent / "leaderboard"
    
    # Get all CSV files
    csv_files = sorted(leaderboard_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {leaderboard_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    print("=" * 80)
    
    # Store all models across all files
    all_models = set()
    
    for csv_file in csv_files:
        print(f"\n{csv_file.name}")
        print("-" * 80)
        
        try:
            # Read CSV, using the 3rd column (index 2) for models
            df = pd.read_csv(csv_file)
            
            if len(df.columns) >= 3:
                # Get the 3rd column (index 2)
                model_column = df.iloc[:, 2]
                unique_models = sorted(model_column.dropna().unique())
                
                print(f"Number of unique models: {len(unique_models)}")
                for model in unique_models:
                    print(f"  - {model}")
                    all_models.add(model)
            else:
                print(f"  ⚠️  File has fewer than 3 columns ({len(df.columns)} columns found)")
        
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"\nSUMMARY")
    print("-" * 80)
    print(f"Total unique models across all files: {len(all_models)}")
    print("\nAll unique models:")
    for model in sorted(all_models):
        print(f"  - {model}")

if __name__ == "__main__":
    list_unique_models_from_leaderboard()
