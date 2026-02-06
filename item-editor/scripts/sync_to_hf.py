#!/usr/bin/env python3
"""
Syncs a directory to the Hugging Face Hub dataset.
Only uploads new or modified files. Handles large folders efficiently.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

# Default Configuration
DEFAULT_LOCAL_DIR = "eval_traces"
DEFAULT_REPO_ID = "aims-foundation/eval_traces"
DEFAULT_REPO_TYPE = "dataset"

def sync_traces(local_dir: str, repo_id: str, repo_type: str, large: bool = False):
    # Ensure the local directory exists
    if not os.path.isdir(local_dir):
        print(f"Error: Local directory '{local_dir}' does not exist.")
        sys.exit(1)

    api = HfApi()
    
    print(f"Starting sync of '{local_dir}' to '{repo_id}' (type={repo_type})...")
    if large:
        print("Using 'upload_large_folder' for large datasets (this may take some time and uses multiple commits).")
    else:
        print("Using standard 'upload_folder'. If this fails or warns about folder size, try with --large.")
    
    print("Only new or modified files will be uploaded.")

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print("HF Transfer (hf_transfer) is enabled for faster uploads.")

    try:
        if large:
            # upload_large_folder is recommended for thousands of files or hundreds of GB
            # It handles large uploads by splitting them into multiple commits.
            api.upload_large_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            # upload_large_folder doesn't return a commit info, so we construct the URL
            if repo_type == "dataset":
                url = f"https://huggingface.co/datasets/{repo_id}"
            elif repo_type == "space":
                url = f"https://huggingface.co/spaces/{repo_id}"
            else:
                url = f"https://huggingface.co/{repo_id}"
        else:
            # upload_folder handles hashing and only uploads changes.
            # Best for smaller or incrementally updated folders.
            commit_info = api.upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            # Construct URL from repo info if commit_info is available
            if hasattr(commit_info, 'commit_url'):
                url = commit_info.commit_url
            else:
                if repo_type == "dataset":
                    url = f"https://huggingface.co/datasets/{repo_id}"
                else:
                    url = f"https://huggingface.co/{repo_id}"

        print(f"Successfully synced.")
        print(f"View repository at: {url}")
        
    except Exception as e:
        print(f"Error during sync: {e}")
        if "401" in str(e) or "403" in str(e):
             print("\nAuthentication failed/required.")
             print("Please set the 'HF_TOKEN' environment variable or run 'huggingface-cli login'.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync local folder to Hugging Face Hub.")
    parser.add_argument("--folder", default=DEFAULT_LOCAL_DIR, help=f"Local folder to sync (default: {DEFAULT_LOCAL_DIR})")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help=f"Repo ID on HF (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=["model", "dataset", "space"], help=f"Repo type (default: {DEFAULT_REPO_TYPE})")
    parser.add_argument("--large", action="store_true", help="Use upload_large_folder for large datasets (>50GB or >100k files)")
    
    args = parser.parse_args()
    
    sync_traces(args.folder, args.repo, args.repo_type, args.large)