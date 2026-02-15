import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Root of the repository
REPO_ROOT = Path(__file__).resolve().parents[2]

# Colors for CLI output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def log(msg: str, color: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{Colors.NC}", flush=True)

def detect_data_root() -> Path:
    """
    Detect the root directory for data storage.
    Priority:
    1. DATA_PATH env var
    2. HAL_DATA_ROOT env var
    3. REPO_ROOT/.hal_data (default)
    """
    for env_var in ["DATA_PATH", "HAL_DATA_ROOT"]:
        val = os.environ.get(env_var)
        if val:
            p = Path(val)
            if p.exists() and os.access(p, os.W_OK):
                return p
    
    # Default to .hal_data in repo
    local_data = REPO_ROOT / "result" / ".hal_data"
    local_data.mkdir(exist_ok=True, parents=True)
    return local_data

def get_run_root() -> Path:
    """
    Detect the specific run root directory.
    If using repo-local storage, it's just the data root.
    Otherwise, it includes namespace and repo name.
    """
    data_root = detect_data_root()
    
    if data_root == REPO_ROOT / "result" / ".hal_data":
        return data_root
    
    namespace = os.environ.get("HAL_DATA_NAMESPACE", os.getlogin())
    repo_name = REPO_ROOT.name
    return data_root / "hal_runs" / namespace / repo_name

def setup_data_dirs():
    """
    Set up standard directories and environment variables.
    """
    run_root = get_run_root()
    
    dirs = {
        "TMPDIR": run_root / "tmp",
        "HAL_RESULTS_DIR": run_root / "results",
        "HAL_TRACES_DIR": run_root / "traces",
        "HAL_TMP_DIR": run_root / "tmp",
        "HAL_LOGS_DIR": run_root / "logs"
    }
    
    for key, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        # Standard temp env vars
        if key == "TMPDIR":
            os.environ["TEMP"] = str(path)
            os.environ["TMP"] = str(path)
            os.environ["PYTHON_TEMPDIR"] = str(path)
    
    # Link directories for convenience if they don't exist as symlinks
    if os.environ.get("HAL_LINK_DATA_DIRS", "1") != "0":
        for name, target in dirs.items():
            if name == "TMPDIR": name = ".tmp"
            else: name = name.replace("HAL_", "").replace("_DIR", "").lower()
            
            # Special cases for names
            if name == "logs": names = ["logs", "log"]
            else: names = [name]
            
            for n in names:
                src = REPO_ROOT / n
                if src.is_symlink():
                    if not src.exists():
                        src.unlink()
                        src.symlink_to(target)
                elif not src.exists():
                    src.symlink_to(target)
                # If it's a real directory, we don't move it automatically 
                # to avoid data loss, but we warn.
                elif not src.is_symlink():
                    log(f"Warning: {src} exists and is not a symlink. Centralization might be partial.", Colors.YELLOW)

    return dirs

def get_hal_harness_path() -> Path:
    return REPO_ROOT / "hal-harness"

def get_dotenv_path() -> Path:
    env_path = os.environ.get("HAL_DOTENV_PATH")
    if env_path:
        return Path(env_path)
    
    harness_env = get_hal_harness_path() / ".env"
    if harness_env.exists():
        return harness_env
    
    return REPO_ROOT / ".env"
