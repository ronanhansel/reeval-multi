#!/usr/bin/env python3
"""
Script to rename JSON files to match their corresponding CSV files by time order.
"""

from pathlib import Path

EPOCH_DIR = Path("/Users/ronan/Downloads/epoch_ai")

def get_files_by_time(pattern):
    """Get files sorted by modification time (oldest first)."""
    files = []
    for f in EPOCH_DIR.glob(pattern):
        stat = f.stat()
        files.append((f, stat.st_mtime))
    
    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[1])
    return [f[0] for f in files]

def main():
    # Get all CSV and JSON files sorted by time
    csv_files = get_files_by_time("*.csv")
    json_files = get_files_by_time("info_*.json")
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Found {len(json_files)} JSON files with 'info_' prefix")
    
    if not json_files:
        print("\n✓ No JSON files to rename!")
        return
    
    if len(csv_files) != len(json_files):
        print(f"\n⚠ WARNING: File count mismatch!")
        print(f"  CSV files: {len(csv_files)}")
        print(f"  JSON files: {len(json_files)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n" + "="*80)
    print("JSON FILE RENAMING PLAN (Matched by time order)")
    print("="*80)
    
    rename_plan = []
    for i, (csv_file, json_file) in enumerate(zip(csv_files, json_files), 1):
        old_json_name = json_file.name
        new_json_name = f"{csv_file.stem}.json"
        
        if old_json_name != new_json_name:
            rename_plan.append((json_file, new_json_name))
            print(f"{i:3d}. {old_json_name:70s} -> {new_json_name}")
    
    if not rename_plan:
        print("\n✓ All JSON files are already correctly named!")
        return
    
    print(f"\n{len(rename_plan)} JSON files need to be renamed.")
    response = input("\nProceed with renaming? (y/n): ")
    
    if response.lower() == 'y':
        print("\nRenaming JSON files...")
        for json_path, new_json_name in rename_plan:
            new_json_path = json_path.parent / new_json_name
            
            # Handle name conflicts
            if new_json_path.exists() and new_json_path != json_path:
                print(f"  ⚠ Target exists: {new_json_name}, skipping")
                continue
            
            json_path.rename(new_json_path)
            print(f"  ✓ {json_path.name} -> {new_json_name}")
        
        print("\n✓ All JSON files renamed successfully!")
    else:
        print("\nRenaming cancelled.")

if __name__ == "__main__":
    main()
