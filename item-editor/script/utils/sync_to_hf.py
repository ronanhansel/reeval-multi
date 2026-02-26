#!/usr/bin/env python3
"""
Syncs the 'eval_traces' and 'eval_response_matrix' directories to the Hugging Face Hub.
Only uploads new or modified files.
"""

import os
import sys
from huggingface_hub import HfApi

# Configuration
SYNC_CONFIGS = [
    {
        "local_dir": "eval_response_matrix",
        "repo_id": "aims-foundation/eval_response_matrix",
        "repo_type": "dataset",
    },
    {
        "local_dir": "eval_traces",
        "repo_id": "aims-foundation/eval_traces",
        "repo_type": "dataset",
    }
]

def sync_all():
    api = HfApi()
    
    for config in SYNC_CONFIGS:
        local_dir = config["local_dir"]
        repo_id = config["repo_id"]
        repo_type = config["repo_type"]

        # Ensure the local directory exists
        if not os.path.isdir(local_dir):
            print(f"Warning: Local directory '{local_dir}' does not exist. Skipping...")
            continue
        
        print(f"\nStarting sync of '{local_dir}' to '{repo_id}' (type={repo_type})...")
        print("This will upload only files that are new or modified compared to the repository.")

        try:
            # upload_folder handles hashing and only uploads changes
            url = api.upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print(f"Successfully synced '{local_dir}'.")
            print(f"View dataset at: {url}")
            
        except Exception as e:
            print(f"Error during sync of '{local_dir}': {e}")
            if "401" in str(e) or "403" in str(e):
                print("\nAuthentication failed/required.")
                print("Please set the 'HF_TOKEN' environment variable or run 'huggingface-cli login'.")
            # We continue to the next config even if one fails
            # sys.exit(1) # Removed so it tries the next one

if __name__ == "__main__":
    sync_all()
