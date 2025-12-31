#!/usr/bin/env python3
"""
Script to rename epoch_ai CSV and JSON files according to models_map.csv order.
Files are downloaded sequentially, so we match by creation time order.
"""

import os
import csv
from pathlib import Path
from datetime import datetime

# Paths
EPOCH_DIR = Path("/Users/ronan/Downloads/epoch_ai")
MODELS_MAP = Path("/Users/ronan/Developer/reeval-multi/inspect-ai/models_map.csv")

def get_csv_and_json_files_by_time(directory):
    """Get CSV and JSON file pairs sorted by modification time (oldest first)."""
    csv_files = []
    for f in directory.glob("*.csv"):
        stat = f.stat()
        csv_files.append((f, stat.st_mtime))
    
    # Sort by modification time (oldest first)
    csv_files.sort(key=lambda x: x[1])
    
    # For each CSV, find corresponding JSON
    file_pairs = []
    for csv_path, _ in csv_files:
        csv_name = csv_path.stem
        # Look for corresponding info_ json file
        json_candidates = [
            directory / f"info_{csv_name}.json",
            directory / f"{csv_name}.json",
        ]
        json_path = None
        for candidate in json_candidates:
            if candidate.exists():
                json_path = candidate
                break
        
        file_pairs.append((csv_path, json_path))
    
    return file_pairs

def read_models_map(csv_path):
    """Read the models from models_map.csv."""
    models = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Skip empty rows
                models.append(row[0])
    return models

def sanitize_filename(name):
    """Sanitize model name for use as filename."""
    # Replace slashes with underscores
    return name.replace('/', '_')

def main():
    print("Reading models map...")
    models = read_models_map(MODELS_MAP)
    print(f"Found {len(models)} models in models_map.csv")
    
    print("\nGetting CSV and JSON files sorted by time...")
    file_pairs = get_csv_and_json_files_by_time(EPOCH_DIR)
    print(f"Found {len(file_pairs)} file pairs in epoch_ai directory")
    
    if len(file_pairs) != len(models):
        print(f"\nWARNING: Mismatch in counts!")
        print(f"  Models in map: {len(models)}")
        print(f"  File pairs: {len(file_pairs)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create mapping
    print("\n" + "="*80)
    print("FILE RENAMING PLAN (Earliest to Latest)")
    print("="*80)
    
    rename_plan = []
    for i, ((csv_file, json_file), model_name) in enumerate(zip(file_pairs, models), 1):
        old_csv_name = csv_file.name
        sanitized_model = sanitize_filename(model_name)
        new_csv_name = f"{sanitized_model}.csv"
        new_json_name = f"{sanitized_model}.json"
        
        # Check if renaming is needed
        csv_needs_rename = old_csv_name != new_csv_name
        json_needs_rename = json_file and json_file.name != new_json_name
        
        if csv_needs_rename or json_needs_rename:
            rename_plan.append((csv_file, new_csv_name, json_file, new_json_name, model_name))
            csv_status = f"{old_csv_name:60s} -> {new_csv_name}" if csv_needs_rename else f"{old_csv_name:60s} [OK]"
            print(f"{i:3d}. CSV: {csv_status}")
            if json_file:
                json_status = f"{json_file.name:60s} -> {new_json_name}" if json_needs_rename else f"{json_file.name:60s} [OK]"
                print(f"     JSON: {json_status}")
        else:
            print(f"{i:3d}. {old_csv_name:60s} [ALREADY CORRECT]")
    
    if not rename_plan:
        print("\n✓ All files are already correctly named!")
        return
    
    print(f"\n{len(rename_plan)} file pairs need to be renamed.")
    response = input("\nProceed with renaming? (y/n): ")
    
    if response.lower() == 'y':
        print("\nRenaming files...")
        for csv_path, new_csv_name, json_path, new_json_name, model_name in rename_plan:
            # Rename CSV
            new_csv_path = csv_path.parent / new_csv_name
            if csv_path.name != new_csv_name:
                # Handle name conflicts
                if new_csv_path.exists() and new_csv_path != csv_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{new_csv_path.stem}_backup_{timestamp}{new_csv_path.suffix}"
                    backup_path = new_csv_path.parent / backup_name
                    print(f"  ⚠ CSV target exists, backing up: {new_csv_path.name} -> {backup_name}")
                    new_csv_path.rename(backup_path)
                
                csv_path.rename(new_csv_path)
                print(f"  ✓ CSV: {csv_path.name} -> {new_csv_name}")
            
            # Rename JSON if exists
            if json_path and json_path.name != new_json_name:
                new_json_path = json_path.parent / new_json_name
                
                # Handle name conflicts
                if new_json_path.exists() and new_json_path != json_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{new_json_path.stem}_backup_{timestamp}{new_json_path.suffix}"
                    backup_path = new_json_path.parent / backup_name
                    print(f"  ⚠ JSON target exists, backing up: {new_json_path.name} -> {backup_name}")
                    new_json_path.rename(backup_path)
                
                json_path.rename(new_json_path)
                print(f"  ✓ JSON: {json_path.name} -> {new_json_name}")
        
        print("\n✓ All files renamed successfully!")
    else:
        print("\nRenaming cancelled.")

if __name__ == "__main__":
    main()
