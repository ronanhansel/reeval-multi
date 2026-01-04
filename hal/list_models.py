import pandas as pd
import glob
import os

def extract_models_from_csvs():
    """Extract all unique models from CSV files in the current directory."""
    
    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    
    all_models = set()
    
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file}")
        print(f"{'='*60}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Try different column names that might contain model information
            model_columns = [col for col in df.columns if 'model' in col.lower() or 'primary' in col.lower()]
            
            if model_columns:
                # Use the first matching column
                model_col = model_columns[0]
                models = df[model_col].dropna().unique()
                
                print(f"Found {len(models)} unique models in column '{model_col}':")
                for model in sorted(models):
                    print(f"  - {model}")
                    all_models.add(model)
            else:
                print("No model column found")
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"ALL UNIQUE MODELS ACROSS ALL FILES ({len(all_models)} total)")
    print(f"{'='*60}")
    for model in sorted(all_models):
        print(f"  - {model}")
    
    # Extract base model names (removing variations like High, Low, Medium)
    import re
    base_models = {}
    for model in all_models:
        # Remove common variation suffixes and parentheses content for grouping
        # Pattern: remove " (High|Low|Medium)" before the date
        base_name = re.sub(r'\s+(High|Low|Medium)\s+\(', ' (', model)
        
        if base_name not in base_models:
            base_models[base_name] = []
        base_models[base_name].append(model)
    
    print(f"\n{'='*60}")
    print(f"UNIQUE BASE MODELS ({len(base_models)} total)")
    print(f"{'='*60}")
    for base_name in sorted(base_models.keys()):
        variations = sorted(base_models[base_name])
        print(f"\n{base_name}:")
        for variation in variations:
            print(f"  - {variation}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique model variations: {len(all_models)}")
    print(f"Total unique base models: {len(base_models)}")
    
    return sorted(all_models)

if __name__ == "__main__":
    models = extract_models_from_csvs()
