import pandas as pd
from pathlib import Path

def list_unique_models():
    """List all unique models from each CSV file in the rubrics_merged folder."""
    
    # Path to the rubrics_merged folder
    rubrics_dir = Path(__file__).parent.parent / "hal-trait-analysis" / "rubrics_merged"
    
    # Get all CSV files
    csv_files = sorted(rubrics_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {rubrics_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    print("=" * 80)
    
    # Store all models across all files
    all_models = set()
    
    for csv_file in csv_files:
        print(f"\n{csv_file.name}")
        print("-" * 80)
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check if 'model' column exists
            if 'model' in df.columns:
                unique_models = sorted(df['model'].dropna().unique())
                
                print(f"Number of unique models: {len(unique_models)}")
                for model in unique_models:
                    print(f"  - {model}")
                    all_models.add(model)
            else:
                print("  ⚠️  No 'model' column found in this file")
                print(f"  Available columns: {', '.join(df.columns)}")
        
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
    list_unique_models()
