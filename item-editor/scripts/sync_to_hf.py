#!/usr/bin/env python3
"""
Syncs the 'eval_traces' directory to the Hugging Face Hub dataset 'aims-foundation/eval_traces'.
Only uploads new or modified files.
"""

import os
import sys
from huggingface_hub import HfApi

# Configuration
LOCAL_DIR = "eval_traces"
REPO_ID = "aims-foundation/eval_traces"
REPO_TYPE = "dataset"

def sync_traces():
    # Ensure the local directory exists
    if not os.path.isdir(LOCAL_DIR):
        print(f"Error: Local directory '{LOCAL_DIR}' does not exist.")
        sys.exit(1)

    api = HfApi()
    
    print(f"Starting sync of '{LOCAL_DIR}' to '{REPO_ID}' (type={REPO_TYPE})...")
    print("This will upload only files that are new or modified compared to the repository.")

    try:
        # upload_folder handles hashing and only uploads changes
        url = api.upload_folder(
            folder_path=LOCAL_DIR,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            # You can add delete_patterns="*" if you wanted to delete remote files not in local, 
            # but the user asked to sync files *to* the repo, usually implying additive/update.
            # Default behavior is additive/update.
        )
        print(f"Successfully synced.")
        print(f"View dataset at: {url}")
        
    except Exception as e:
        print(f"Error during sync: {e}")
        if "401" in str(e) or "403" in str(e):
             print("\nAuthentication failed/required.")
             print("Please set the 'HF_TOKEN' environment variable or run 'huggingface-cli login'.")
        sys.exit(1)

if __name__ == "__main__":
    sync_traces()
