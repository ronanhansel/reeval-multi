import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from huggingface_hub import snapshot_download
import pickle
import subprocess
from pathlib import Path
from huggingface_hub import login

# Create the traces directory
traces_dir = Path("item-editor")
traces_dir.mkdir(exist_ok=True)
# Login with Hugging Face token
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    login()  # Will prompt for token interactively


local_path = snapshot_download(
    repo_id="aims-foundation/eval_response_matrix", 
    repo_type="dataset",
    local_dir=traces_dir / "eval_response_matrix",  # Specify the local directory
    max_workers=8
)

local_path = snapshot_download(
    repo_id="aims-foundation/eval_traces", 
    repo_type="dataset",
    local_dir=traces_dir / "eval_traces",  # Specify the local directory
    max_workers=8
)