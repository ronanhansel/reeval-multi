#!/usr/bin/env python3
"""
Unified Benchmark Fix Runner

This script runs HAL evaluations for benchmarks with optional fix application.

Modes:
    1. Run only tasks with fixes (default): Only runs tasks that have fixes defined
    2. Run all tasks with --all-tasks: Runs ENTIRE benchmark, applying fixes where available

Parallelism:
    --parallel-models N   Run N model configurations concurrently
    --parallel-tasks N    Run N tasks concurrently within each HAL evaluation
                          (uses HAL's --max_concurrent flag)

Usage:
    # List available configurations for a benchmark
    python scripts/run_benchmark_fixes.py --benchmark scicode --list-configs

    # List all benchmarks with fixes
    python scripts/run_benchmark_fixes.py --list-benchmarks

    # Run ALL tasks in benchmark, applying fixes where available
    python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs \
        --all-tasks --prefix iter1_ --docker --parallel-tasks 5

    # Run ALL benchmarks, ALL tasks, with fixes applied where available
    python scripts/run_benchmark_fixes.py --all-benchmarks --all-configs \
        --all-tasks --prefix iter1_ --docker --parallel-models 2 --parallel-tasks 5

    # Run only tasks with fixes (original behavior)
    python scripts/run_benchmark_fixes.py --benchmark scicode \
        --config gpt-5_scicode_tool_calling \
        --prefix test_ \
        --docker

    # Dry run
    python scripts/run_benchmark_fixes.py --all-benchmarks --all-configs --all-tasks --dry-run
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import threading

# Repo paths (needed before dotenv loading)
REPO_ROOT = Path(__file__).resolve().parents[1]
HAL_HARNESS = REPO_ROOT / "hal-harness"

# Load Azure/TRAPI .env first when available, then local overrides.
from dotenv import load_dotenv
if not os.environ.get("HAL_DOTENV_PATH"):
    default_dotenv = HAL_HARNESS / ".env"
    if default_dotenv.exists():
        os.environ["HAL_DOTENV_PATH"] = str(default_dotenv)
        load_dotenv(default_dotenv, override=False)
load_dotenv(override=False)

# If using direct Azure, remove proxy URLs and validate tokens
if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
    for key in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_BASE_URL', 'LITELLM_BASE_URL'):
        os.environ.pop(key, None)
    print("[INFO] Direct Azure mode: removed proxy URLs from environment")

    # Validate Azure token at startup
    def _check_azure_token():
        """Check if Azure MSAL tokens are available and valid."""
        try:
            import msal
            cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
            if not os.path.exists(cache_path):
                print("[WARN] Azure MSAL cache not found at ~/.azure/msal_token_cache.json")
                print("[WARN] Run 'az login' to authenticate")
                return False

            cache = msal.SerializableTokenCache()
            with open(cache_path, 'r') as f:
                cache.deserialize(f.read())

            app = msal.PublicClientApplication(
                '04b07795-8ddb-461a-bbee-02f9e1bf7b46',  # Azure CLI client ID
                authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',
                token_cache=cache
            )

            accounts = app.get_accounts()
            if not accounts:
                print("[WARN] No Azure accounts found in MSAL cache")
                print("[WARN] Run 'az login' to authenticate")
                return False

            # Try to acquire token silently from ALL accounts (will refresh if needed)
            scope = os.environ.get('TRAPI_SCOPE', 'api://trapi/.default')
            last_error = None
            for idx, account in enumerate(accounts):
                username = account.get('username', 'unknown')
                result = app.acquire_token_silent([scope], account=account, force_refresh=True)

                if result and 'access_token' in result:
                    if cache.has_state_changed:
                        try:
                            with open(cache_path, 'w') as f:
                                f.write(cache.serialize())
                        except Exception as e:
                            print(f"[WARN] Failed to persist MSAL cache: {e}")
                    print(f"[OK] Azure token valid (account {idx}: {username})")
                    return True

                if result:
                    last_error = result.get('error_description', 'unknown')
                else:
                    last_error = f"No token for account {username}"

            print(f"[WARN] Azure token refresh failed: {last_error or 'no result'}")
            print("[WARN] Run 'az login' to re-authenticate")
            return False

        except ImportError:
            print("[ERROR] msal package not installed - cannot validate Azure tokens")
            return False
        except Exception as e:
            print(f"[WARN] Azure token check failed: {e}")
            return False

    def _require_azure_ready(max_attempts: int = 3, delay_seconds: float = 2.0) -> None:
        """Retry Azure token acquisition a few times, then fail fast."""
        import time
        for attempt in range(1, max_attempts + 1):
            ok = _check_azure_token()
            if ok:
                return
            if attempt < max_attempts:
                print(f"[WARN] Azure preflight failed (attempt {attempt}/{max_attempts}). Retrying...")
                time.sleep(delay_seconds)

        print("[ERROR] Azure preflight failed after retries. Aborting to avoid long-running failure.")
        raise SystemExit(2)

    _require_azure_ready()

# =============================================================================
# Path Configuration
# =============================================================================
FIXES_DIR = REPO_ROOT / "fixes"

def _resolve_data_dir(env_key: str, default_path: Path) -> Path:
    raw = os.environ.get(env_key)
    if not raw:
        return default_path
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path

TRACES_DIR = _resolve_data_dir("HAL_TRACES_DIR", REPO_ROOT / "traces")
RESULTS_DIR = _resolve_data_dir("HAL_RESULTS_DIR", REPO_ROOT / "results")
TMP_DIR = _resolve_data_dir("HAL_TMP_DIR", REPO_ROOT / ".tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Add shared module to path
sys.path.insert(0, str(HAL_HARNESS / "agents"))

# Try to import model utilities for quirks handling
try:
    from shared.model_utils import (
        supports_temperature,
        supports_reasoning_effort,
        uses_max_completion_tokens,
        supports_stop,
        is_reasoning_model,
    )
    MODEL_UTILS_AVAILABLE = True
except ImportError:
    MODEL_UTILS_AVAILABLE = False
    print("[WARNING] shared.model_utils not available - model quirks may not be applied")

# =============================================================================
# CoreBench Preparation (decrypt + download capsules)
# =============================================================================
COREBENCH_DIR = HAL_HARNESS / "hal" / "benchmarks" / "corebench"
COREBENCH_CAPSULES_DIR = COREBENCH_DIR / "capsules"
COREBENCH_JSON = COREBENCH_DIR / "core_test.json"
COREBENCH_JSON_GPG = COREBENCH_DIR / "core_test.json.gpg"
COREBENCH_CAPSULE_URL = "https://corebench.cs.princeton.edu/capsules"


def decrypt_corebench_dataset() -> bool:
    """
    Decrypt core_test.json.gpg if core_test.json doesn't exist.
    Prompts user for passphrase with hint.

    Returns:
        True if file exists or was decrypted successfully, False otherwise.
    """
    if COREBENCH_JSON.exists():
        return True

    if not COREBENCH_JSON_GPG.exists():
        print(f"[corebench] ERROR: Neither {COREBENCH_JSON} nor {COREBENCH_JSON_GPG} found")
        return False

    print(f"[corebench] core_test.json not found, need to decrypt from .gpg file")
    print(f"[corebench] Hint: The passphrase is 'reproducibility'")

    passphrase = getpass.getpass("[corebench] Enter passphrase: ")

    try:
        result = subprocess.run(
            [
                "gpg", "--batch", "--yes", "--passphrase-fd", "0",
                "--output", str(COREBENCH_JSON),
                "--decrypt", str(COREBENCH_JSON_GPG)
            ],
            input=passphrase.encode(),
            capture_output=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[corebench] ERROR: Decryption failed: {result.stderr.decode()}")
            return False

        print(f"[corebench] Successfully decrypted core_test.json")
        return True

    except subprocess.TimeoutExpired:
        print(f"[corebench] ERROR: Decryption timed out")
        return False
    except FileNotFoundError:
        print(f"[corebench] ERROR: gpg command not found. Install gnupg.")
        return False
    except Exception as e:
        print(f"[corebench] ERROR: Decryption failed: {e}")
        return False


def download_corebench_capsule(capsule_id: str, max_retries: int = 3) -> str:
    """Download and extract a single CoreBench capsule."""
    capsule_dir = COREBENCH_CAPSULES_DIR / capsule_id

    if capsule_dir.exists():
        return f"{capsule_id}: already exists"

    tar_path = COREBENCH_CAPSULES_DIR / f"{capsule_id}.tar.gz"
    url = f"{COREBENCH_CAPSULE_URL}/{capsule_id}.tar.gz"

    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, str(tar_path))

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=str(COREBENCH_CAPSULES_DIR))

            tar_path.unlink()
            return f"{capsule_id}: downloaded"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                if tar_path.exists():
                    tar_path.unlink()
                return f"{capsule_id}: FAILED - {e}"

    return f"{capsule_id}: FAILED after {max_retries} attempts"


def download_all_corebench_capsules() -> bool:
    """
    Download all CoreBench capsules in parallel.

    Returns:
        True if all capsules downloaded successfully, False otherwise.
    """
    if not COREBENCH_JSON.exists():
        print(f"[corebench] ERROR: core_test.json not found, cannot determine capsules")
        return False

    with open(COREBENCH_JSON, 'r') as f:
        dataset = json.load(f)

    capsule_ids = list(set(task["capsule_id"] for task in dataset))

    # Check how many already exist
    existing = sum(1 for cid in capsule_ids if (COREBENCH_CAPSULES_DIR / cid).exists())

    if existing == len(capsule_ids):
        print(f"[corebench] All {len(capsule_ids)} capsules already downloaded")
        return True

    print(f"[corebench] Downloading capsules: {existing}/{len(capsule_ids)} already exist")

    COREBENCH_CAPSULES_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_corebench_capsule, cid): cid for cid in capsule_ids}

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            capsule_id = futures[future]

            if "FAILED" in result:
                failed.append(capsule_id)
                print(f"[corebench] [{completed}/{len(capsule_ids)}] {result}")
            elif "downloaded" in result:
                print(f"[corebench] [{completed}/{len(capsule_ids)}] {result}")

    if failed:
        print(f"[corebench] WARNING: {len(failed)} capsules failed to download: {failed}")
        return False

    print(f"[corebench] All {len(capsule_ids)} capsules ready")
    return True


def prepare_corebench() -> bool:
    """
    Prepare CoreBench for running: decrypt dataset and download all capsules.

    Returns:
        True if preparation successful, False otherwise.
    """
    print(f"[corebench] Preparing CoreBench environment...")

    # Step 1: Decrypt dataset if needed
    if not decrypt_corebench_dataset():
        return False

    # Step 2: Download all capsules
    if not download_all_corebench_capsules():
        return False

    print(f"[corebench] CoreBench preparation complete")
    return True


# =============================================================================
# SciCode Preparation
# =============================================================================

def download_scicode_test_data() -> bool:
    """Download SciCode test_data.h5 if missing."""
    h5_path = HAL_HARNESS / "hal" / "benchmarks" / "SciCode" / "eval" / "data" / "test_data.h5"
    if h5_path.exists():
        return True
    
    print(f"[scicode] Downloading test_data.h5 from Google Drive...")
    try:
        # Check if gdown is installed
        subprocess.run(["gdown", "--version"], check=True, capture_output=True)
        
        # Download
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        # File ID for test_data.h5
        cmd = ["gdown", "--id", "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW", "-O", str(h5_path)]
        subprocess.run(cmd, check=True)
        
        if h5_path.exists():
            print(f"[scicode] Download successful")
            return True
        else:
            print(f"[scicode] Download failed (file not found)")
            return False
            
    except FileNotFoundError:
        print(f"[scicode] gdown not found. Please run: pip install gdown")
        return False
    except Exception as e:
        print(f"[scicode] Download failed: {e}")
        return False

def prepare_scicode() -> bool:
    """Prepare SciCode environment."""
    return download_scicode_test_data()


# =============================================================================
# Benchmark to HAL benchmark name mapping
# =============================================================================
# Maps fixes directory names to HAL benchmark names
BENCHMARK_HAL_NAME_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "corebench_hard": "corebench_hard",
    "usaco": "usaco",
    # ColBench: ONLY backend is supported, frontend uses CLIP which is different
    "colbench": "colbench_backend_programming",
    "colbench_backend_programming": "colbench_backend_programming",
}

# Environment variables used by HAL benchmarks for custom datasets
BENCHMARK_DATASET_ENV_VAR = {
    "scicode": "SCICODE_DATASET_PATH",
    "scienceagentbench": "SCIENCEAGENTBENCH_DATASET_PATH",
    "corebench": "HAL_COREBENCH_DATASET_PATH",
    "corebench_hard": "HAL_COREBENCH_DATASET_PATH",
    "colbench": "COLBENCH_BACKEND_DATASET_PATH",
    "colbench_backend_programming": "COLBENCH_BACKEND_DATASET_PATH",
    # USACO uses local files, handled differently
}

# Task ID field names for different benchmarks
BENCHMARK_TASK_ID_FIELD = {
    "scicode": "problem_id",
    "scienceagentbench": "instance_id",
    "corebench": "capsule_id",
    "corebench_hard": "capsule_id",
    "colbench": "id",
    "colbench_backend_programming": "id",
    "usaco": "problem_id",
}

# ColBench warning - frontend is NOT supported
COLBENCH_WARNING = """
================================================================================
WARNING: ColBench only supports colbench_backend_programming tasks!

The frontend tasks (colbench_frontend_design) use CLIP similarity evaluation
which requires different handling and is NOT supported by this script.

Running: colbench_backend_programming
================================================================================
"""


def log(msg: str, prefix: str = "main") -> None:
    """Log with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


# =============================================================================
# Model Quirks Handling
# =============================================================================

def get_model_quirks(model_id: str) -> Dict[str, bool]:
    """
    Get model quirks/capabilities.

    Returns dict with:
        - supports_temperature: bool
        - supports_stop: bool
        - supports_reasoning_effort: bool
        - uses_max_completion_tokens: bool
        - is_reasoning_model: bool
    """
    if MODEL_UTILS_AVAILABLE:
        return {
            'supports_temperature': supports_temperature(model_id),
            'supports_stop': supports_stop(model_id),
            'supports_reasoning_effort': supports_reasoning_effort(model_id),
            'uses_max_completion_tokens': uses_max_completion_tokens(model_id),
            'is_reasoning_model': is_reasoning_model(model_id),
        }

    # Fallback implementation
    model_lower = model_id.lower()
    is_o_series = any(model_lower.startswith(p) for p in ['o1', 'o3', 'o4'])
    is_gpt5 = 'gpt-5' in model_lower or model_lower.startswith('gpt-5')
    is_deepseek_r1 = 'deepseek-r1' in model_lower or 'deepseek_r1' in model_lower

    return {
        'supports_temperature': not (is_o_series or is_gpt5 or is_deepseek_r1),
        'supports_stop': not (is_o_series or is_gpt5),
        'supports_reasoning_effort': is_o_series or is_gpt5,
        'uses_max_completion_tokens': is_o_series or is_gpt5,
        'is_reasoning_model': is_o_series or is_gpt5 or is_deepseek_r1,
    }


def validate_model_config(config_key: str, entry: Dict[str, Any]) -> List[str]:
    """
    Validate model configuration against quirks.
    Returns list of warnings/issues.
    """
    warnings = []
    model_id = entry.get('model_id', '')
    quirks = get_model_quirks(model_id)

    # Check temperature
    if 'temperature' in entry and not quirks['supports_temperature']:
        warnings.append(f"  [WARN] {config_key}: temperature={entry['temperature']} will be ignored (model doesn't support it)")

    # Check reasoning_effort
    if 'reasoning_effort' in entry and not quirks['supports_reasoning_effort']:
        warnings.append(f"  [WARN] {config_key}: reasoning_effort={entry['reasoning_effort']} will be ignored (model doesn't support it)")

    # Suggest reasoning_effort for reasoning models that don't have it set
    if quirks['is_reasoning_model'] and 'reasoning_effort' not in entry:
        warnings.append(f"  [INFO] {config_key}: reasoning model without reasoning_effort - will use default")

    return warnings


# =============================================================================
# Dataset Loading
# =============================================================================

def load_benchmark_dataset(benchmark: str) -> List[Dict[str, Any]]:
    """
    Load the full dataset for a benchmark.

    Args:
        benchmark: Benchmark config name (scicode, scienceagentbench, corebench, colbench)

    Returns:
        List of task dictionaries
    """
    try:
        if benchmark == "scicode":
            try:
                from datasets import load_dataset
                dataset = list(load_dataset("SciCode1/SciCode", split="test"))
                log(f"Loaded {len(dataset)} tasks from SciCode (HuggingFace)", "data")
                return dataset
            except ImportError as e:
                log(f"ERROR: 'datasets' package required for SciCode: {e}", "data")
                return []

        elif benchmark == "scienceagentbench":
            try:
                from datasets import load_dataset
                dataset = list(load_dataset("osunlp/ScienceAgentBench", split="validation"))
                log(f"Loaded {len(dataset)} tasks from ScienceAgentBench (HuggingFace)", "data")
                return dataset
            except ImportError:
                log("ERROR: 'datasets' package required for ScienceAgentBench", "data")
                return []

        elif benchmark in ("corebench", "corebench_hard"):
            # CoreBench loads from local core_test.json file
            core_test_path = HAL_HARNESS / "hal" / "benchmarks" / "corebench" / "core_test.json"
            if not core_test_path.exists():
                encrypted_file = HAL_HARNESS / "hal" / "benchmarks" / "corebench" / "core_test.json.gpg"
                if encrypted_file.exists():
                    log(f"ERROR: core_test.json not found. Decrypt with:", "data")
                    log(f"  gpg --output {core_test_path} --decrypt {encrypted_file}", "data")
                    log(f"  Password: reproducibility", "data")
                else:
                    log(f"ERROR: core_test.json not found at {core_test_path}", "data")
                return []

            dataset = json.loads(core_test_path.read_text())
            log(f"Loaded {len(dataset)} tasks from CoreBench (local JSON)", "data")
            return dataset

        elif benchmark in ("colbench", "colbench_backend_programming"):
            # ColBench loads from local JSONL file
            colbench_path = HAL_HARNESS / "hal" / "benchmarks" / "colbench" / "data" / "backend_test.jsonl"
            if not colbench_path.exists():
                log(f"ERROR: ColBench data not found at {colbench_path}", "data")
                log("Download ColBench data to hal/benchmarks/colbench/data/", "data")
                return []

            tasks = []
            with open(colbench_path, "r") as f:
                for i, line in enumerate(f):
                    task = json.loads(line)
                    task["id"] = str(i)  # Add task ID based on line number
                    tasks.append(task)

            log(f"Loaded {len(tasks)} tasks from ColBench backend (local JSONL)", "data")
            return tasks

        else:
            log(f"Unknown benchmark: {benchmark} - cannot load dataset", "data")
            return []

    except Exception as e:
        log(f"Failed to load dataset for {benchmark}: {e}", "data")
        import traceback
        traceback.print_exc()
        return []


def load_fix_package(task_id: str, benchmark: str) -> Optional[Dict[str, Any]]:
    """
    Load fix package for a specific task.

    Args:
        task_id: Task identifier
        benchmark: Benchmark config name

    Returns:
        Fix dictionary or None if no fix exists
    """
    # Determine fixes directory
    if benchmark == "colbench":
        fix_dirs = [FIXES_DIR / "colbench", FIXES_DIR / "colbench_backend_programming"]
    elif benchmark == "corebench":
        fix_dirs = [FIXES_DIR / "corebench_hard"]
    else:
        fix_dirs = [FIXES_DIR / benchmark]

    for fix_dir in fix_dirs:
        task_fix_dir = fix_dir / str(task_id)
        if task_fix_dir.exists():
            fix = {"task_id": task_id, "fix_dir": task_fix_dir}

            # Load all override types
            for override_name in [
                "instruction_override", "dependency_override",
                "evaluation_override", "env_override", "input_override"
            ]:
                override_path = task_fix_dir / f"{override_name}.json"
                if override_path.exists():
                    try:
                        fix[override_name] = json.loads(override_path.read_text())
                    except json.JSONDecodeError as e:
                        log(f"Warning: Failed to parse {override_path}: {e}", "fix")

            # Load problem statement override if exists
            problem_stmt_path = task_fix_dir / "problem_statement.txt"
            if problem_stmt_path.exists():
                fix["problem_statement"] = problem_stmt_path.read_text()

            return fix

    return None


def load_all_fixes(benchmark: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all fixes for a benchmark.

    Args:
        benchmark: Benchmark config name

    Returns:
        Dictionary mapping task_id -> fix dict
    """
    fixes = {}
    task_ids = get_task_ids_with_fixes(benchmark)

    for task_id in task_ids:
        fix = load_fix_package(task_id, benchmark)
        if fix:
            fixes[task_id] = fix

    log(f"Loaded {len(fixes)} fixes for {benchmark}", "fix")
    return fixes


def apply_fix_to_task(task: Dict[str, Any], fix: Dict[str, Any], benchmark: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply a fix to a task.

    This is a simplified version that handles the common fix types.
    Benchmark-specific complex fixes may need the full runners.

    Args:
        task: Original task dictionary
        fix: Fix dictionary with override fields
        benchmark: Benchmark name

    Returns:
        Tuple of (modified_task, list of changes made)
    """
    import copy
    modified = copy.deepcopy(task)
    changes = []

    # Get task ID field for this benchmark
    id_field = BENCHMARK_TASK_ID_FIELD.get(benchmark, "id")
    task_id = str(task.get(id_field, ""))

    # Apply instruction_override
    if fix.get("instruction_override"):
        instr = fix["instruction_override"]

        # Add clarifications to task description/instruction
        if instr.get("clarifications"):
            clarifications = instr["clarifications"]
            clarification_text = "\n\nCLARIFICATIONS:\n" + "\n".join(f"- {c}" for c in clarifications)

            # Different benchmarks use different fields for instructions
            for field in ["instruction", "task_inst", "problem_statement", "description", "task"]:
                if field in modified:
                    modified[field] = str(modified[field]) + clarification_text
                    changes.append(f"Added {len(clarifications)} clarifications to {field}")
                    break

        # Handle step-specific overrides (for SciCode)
        if instr.get("overrides"):
            modified["_fix_overrides"] = instr["overrides"]
            changes.append(f"Added step overrides: {list(instr['overrides'].keys())}")

        if instr.get("step_overrides"):
            modified["_fix_step_overrides"] = instr["step_overrides"]
            changes.append(f"Added step overrides: {list(instr['step_overrides'].keys())}")

    # Apply problem_statement override
    if fix.get("problem_statement"):
        for field in ["problem_statement", "instruction", "task_inst", "description"]:
            if field in modified:
                modified[field] = fix["problem_statement"]
                changes.append(f"Replaced {field} with fix problem_statement")
                break

    # Apply dependency_override (store for agent to use)
    if fix.get("dependency_override"):
        modified["_fix_dependencies"] = fix["dependency_override"]
        changes.append(f"Added dependency override: {list(fix['dependency_override'].keys())}")

    # Apply evaluation_override (store for evaluation)
    if fix.get("evaluation_override"):
        modified["_fix_evaluation"] = fix["evaluation_override"]
        changes.append(f"Added evaluation override: {list(fix['evaluation_override'].keys())}")

    # Apply env_override (store for environment setup)
    if fix.get("env_override"):
        modified["_fix_env"] = fix["env_override"]
        changes.append(f"Added env override: {list(fix['env_override'].keys())}")

    # Apply input_override (for SciCode)
    if fix.get("input_override"):
        modified["_fix_input"] = fix["input_override"]
        changes.append(f"Added input override")

    return modified, changes


def create_modified_dataset(
    dataset: List[Dict[str, Any]],
    fixes: Dict[str, Dict[str, Any]],
    benchmark: str,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], int, int]:
    """
    Create a modified dataset with fixes applied.

    Args:
        dataset: Original dataset
        fixes: Dictionary of task_id -> fix
        benchmark: Benchmark name
        verbose: Whether to log changes

    Returns:
        Tuple of (modified_dataset, changes_dict, tasks_with_fixes, tasks_without_fixes)
    """
    id_field = BENCHMARK_TASK_ID_FIELD.get(benchmark, "id")
    modified_dataset = []
    all_changes = {}
    tasks_with_fixes = 0
    tasks_without_fixes = 0

    for task in dataset:
        task_id = str(task.get(id_field, ""))

        if task_id in fixes:
            modified, changes = apply_fix_to_task(task, fixes[task_id], benchmark)
            modified_dataset.append(modified)
            all_changes[task_id] = changes
            tasks_with_fixes += 1

            if verbose and changes:
                log(f"Applied {len(changes)} fix(es) to task {task_id}", "fix")
        else:
            # No fix - add task as-is
            modified_dataset.append(task)
            tasks_without_fixes += 1

    return modified_dataset, all_changes, tasks_with_fixes, tasks_without_fixes


def sample_dataset_tasks(
    dataset: List[Dict[str, Any]],
    benchmark: str,
    sample_size: int,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    total = len(dataset)
    if total <= sample_size:
        log(f"Sample size {sample_size} >= total {total}; using full dataset", "main")
        return dataset

    rng = random.Random(seed)
    sampled = rng.sample(dataset, sample_size)
    id_field = BENCHMARK_TASK_ID_FIELD.get(benchmark, "id")
    sample_ids = [str(task.get(id_field, "")) for task in sampled[:10]]
    seed_label = seed if seed is not None else "random"
    log(f"Sampled {len(sampled)}/{total} tasks for {benchmark} (seed={seed_label})", "main")
    if sample_ids:
        log(f"Sampled task IDs (first {len(sample_ids)}): {', '.join(sample_ids)}", "main")
    return sampled


# =============================================================================
# Benchmark and Fix Detection
# =============================================================================

def get_benchmarks_with_fixes() -> List[str]:
    """
    Get list of benchmarks that have fixes available.

    Returns benchmark names (keys for model_to_baseline_*.json files).
    """
    benchmarks = set()

    if not FIXES_DIR.exists():
        return []

    for subdir in FIXES_DIR.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            # Check if there are actual task fixes inside
            task_dirs = [d for d in subdir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if task_dirs:
                # Map directory name to config file name
                dir_name = subdir.name

                # Map to config file names
                if dir_name in ("colbench", "colbench_backend_programming"):
                    benchmarks.add("colbench")
                elif dir_name == "corebench_hard":
                    benchmarks.add("corebench")
                else:
                    benchmarks.add(dir_name)

    return sorted(benchmarks)


def get_task_ids_with_fixes(benchmark: str) -> List[str]:
    """
    Get list of task IDs that have fixes for a benchmark.

    Args:
        benchmark: Benchmark name (config file key)

    Returns:
        List of task IDs with fixes
    """
    task_ids = []

    # Map benchmark config name to fixes directory name(s)
    if benchmark == "colbench":
        fix_dirs = [FIXES_DIR / "colbench", FIXES_DIR / "colbench_backend_programming"]
    elif benchmark == "corebench":
        fix_dirs = [FIXES_DIR / "corebench_hard"]
    else:
        fix_dirs = [FIXES_DIR / benchmark]

    for fix_dir in fix_dirs:
        if fix_dir.exists():
            for task_dir in fix_dir.iterdir():
                if task_dir.is_dir() and not task_dir.name.startswith("."):
                    # Check if there's at least one fix file inside
                    fix_files = [f for f in task_dir.iterdir() if f.suffix == ".json" and f.name != "status.json"]
                    if fix_files or (task_dir / "README.md").exists():
                        task_ids.append(task_dir.name)

    return sorted(set(task_ids), key=lambda x: (int(x) if x.isdigit() else float('inf'), x))


# =============================================================================
# Configuration Loading
# =============================================================================

def load_benchmark_config(benchmark: str) -> Dict[str, Any]:
    """Load model_to_baseline_<benchmark>.json configuration."""
    config_path = REPO_ROOT / "model_configs" / f"model_to_baseline_{benchmark}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def get_model_entries(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get all model entries from config (excluding _meta)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def get_agent_info(entry: Dict[str, Any]) -> Tuple[Path, str]:
    """Extract agent directory and function from entry."""
    agent_dir = entry.get("agent_dir", "")
    agent_function = entry.get("agent_function", "main.run")

    if not agent_dir:
        raise ValueError("agent_dir not found in entry")

    # Resolve to absolute path
    agent_path = Path(agent_dir)
    if not agent_path.is_absolute():
        agent_path = REPO_ROOT / agent_path

    return agent_path, agent_function


def build_agent_args(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Build agent_args dict from model entry, respecting model quirks."""
    model_id = entry.get("model_id", "")
    quirks = get_model_quirks(model_id)

    args: Dict[str, Any] = {}

    if model_id:
        args["model_name"] = model_id

    # Only add temperature if model supports it
    if "temperature" in entry:
        if quirks['supports_temperature']:
            args["temperature"] = entry["temperature"]
        else:
            log(f"Skipping temperature for {model_id} (not supported)", "quirks")

    # Only add reasoning_effort if model supports it
    if "reasoning_effort" in entry:
        if quirks['supports_reasoning_effort']:
            args["reasoning_effort"] = entry["reasoning_effort"]
        else:
            log(f"Skipping reasoning_effort for {model_id} (not supported)", "quirks")

    # Always pass these through (agent-specific parameters)
    passthrough_keys = [
        "max_steps", "budget", "max_tokens",
        "use_self_debug", "use_knowledge",  # sab_example_agent params
        "context_cutoff",  # Optional agent params
    ]
    for key in passthrough_keys:
        if key in entry:
            args[key] = entry[key]

    return args


# =============================================================================
# Retry Logic
# =============================================================================

def is_retryable_error(error_msg: str) -> bool:
    """Check if an error is retryable (TRAPI timeout, auth, rate limit)."""
    retryable_patterns = [
        "timeout", "timed out", "connection reset", "connection refused",
        "503", "504", "502", "500", "429", "rate limit",
        "401", "403", "unauthorized", "authentication", "invalid token",
        "expired token", "token expired", "authenticationerror",
        "overloaded", "service unavailable", "bad gateway",
    ]
    error_lower = str(error_msg).lower()
    return any(pattern in error_lower for pattern in retryable_patterns)


def run_with_retry(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    max_retries: int = 3,
    base_timeout: Optional[int] = 3600,
) -> Tuple[bool, str, Optional[subprocess.CompletedProcess]]:
    """Run subprocess with retry logic."""
    for attempt in range(max_retries):
        if base_timeout is None:
            timeout = None
        else:
            timeout = base_timeout * (attempt + 1)

        try:
            if attempt > 0:
                timeout_str = f"{timeout}s" if timeout else "infinite"
                log(f"Attempt {attempt + 1}/{max_retries} (timeout={timeout_str})", "retry")

            # Stream output live
            process = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                encoding='utf-8',
                errors='replace'
            )

            stdout_lines = []
            stderr_lines = []

            def stream_reader(pipe, lines_acc, dest):
                try:
                    for line in pipe:
                        dest.write(line)
                        dest.flush()
                        lines_acc.append(line)
                except Exception:
                    pass

            t_out = threading.Thread(target=stream_reader, args=(process.stdout, stdout_lines, sys.stdout))
            t_err = threading.Thread(target=stream_reader, args=(process.stderr, stderr_lines, sys.stderr))

            t_out.start()
            t_err.start()

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                t_out.join()
                t_err.join()
                raise subprocess.TimeoutExpired(cmd, timeout)

            t_out.join()
            t_err.join()

            result = subprocess.CompletedProcess(
                cmd, process.returncode, "".join(stdout_lines), "".join(stderr_lines)
            )

            if result.returncode == 0:
                return True, "Success", result

            combined_output = (result.stderr or "") + (result.stdout or "")

            if is_retryable_error(combined_output):
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 5))
                log(f"Retryable error. Waiting {wait_time:.1f}s...", "retry")
                time.sleep(wait_time)
                continue
            else:
                return False, f"Exit code {result.returncode}", result

        except subprocess.TimeoutExpired:
            log(f"Timeout on attempt {attempt + 1}/{max_retries}", "retry")
            if attempt < max_retries - 1:
                time.sleep(min(30, (2 ** attempt)))
            else:
                return False, "Timeout after retries", None

        except Exception as e:
            if is_retryable_error(str(e)) and attempt < max_retries - 1:
                time.sleep(min(60, (2 ** attempt)))
            else:
                return False, str(e), None

    return False, "Max retries exceeded", None


# =============================================================================
# Completion checks
# =============================================================================

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)

def output_has_incomplete_marker(stdout: Optional[str], stderr: Optional[str]) -> bool:
    combined = f"{stdout or ''}\n{stderr or ''}"
    combined = strip_ansi(combined).lower()
    markers = (
        "tasks are incomplete",
        "incomplete tasks",
        "continue-run flag to retry",
    )
    return any(marker in combined for marker in markers)

def count_dataset_tasks(dataset_path: Optional[Path]) -> Optional[int]:
    if not dataset_path or not dataset_path.exists():
        return None
    try:
        if dataset_path.suffix.lower() == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None

def count_completed_submissions(submissions_path: Optional[Path]) -> Optional[int]:
    if not submissions_path or not submissions_path.exists():
        return None
    completed: Dict[str, object] = {}
    try:
        with open(submissions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    submission = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(submission, dict) or not submission:
                    continue
                task_id = next(iter(submission.keys()))
                completed[task_id] = submission[task_id]
    except Exception:
        return None
    return sum(
        1 for value in completed.values()
        if not (isinstance(value, str) and value.startswith("ERROR"))
    )


# =============================================================================
# HAL Evaluation Runner
# =============================================================================

def find_latest_run_id(benchmark: str, prefix: str, config_key: str) -> Optional[str]:
    results_root = RESULTS_DIR / benchmark
    if not results_root.exists():
        return None
    name_prefix = f"{prefix}{config_key}_"
    latest_name = None
    latest_mtime = -1.0
    for path in results_root.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(name_prefix):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_name = path.name
    return latest_name

def run_hal_eval(
    benchmark: str,
    config_key: str,
    entry: Dict[str, Any],
    prefix: str,
    resume: bool = False,
    docker: bool = False,
    max_tasks: Optional[int] = None,
    parallel_tasks: int = 1,
    dataset_path: Optional[Path] = None,
    all_tasks_mode: bool = False,
    trace_mode: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Run HAL evaluation for a specific configuration.

    Args:
        benchmark: HAL benchmark name (e.g., "scicode", "colbench_backend_programming")
        config_key: Configuration key (e.g., "gpt-5_scicode_tool_calling")
        entry: Model entry dict from config
        prefix: Output prefix for traces
        docker: Whether to use Docker isolation
        max_tasks: Maximum number of tasks to run
        parallel_tasks: Number of tasks to run concurrently (HAL's --max_concurrent)
        dataset_path: Path to modified dataset file with fixes applied
        all_tasks_mode: Whether running in --all-tasks mode (full benchmark vs fixes-only)
        trace_mode: HAL_TRACE_MODE value to set for the run

    Returns:
        (success, message, trace_path)
    """
    agent_path, agent_function = get_agent_info(entry)
    agent_args = build_agent_args(entry)

    # Build or reuse run_id from config_key (e.g., "o4-mini_core_agent")
    continue_run = False
    run_id: Optional[str] = None
    if resume:
        existing = find_latest_run_id(benchmark, prefix, config_key)
        if existing:
            run_id = existing
            continue_run = True
            log(f"Resuming run_id: {run_id}", "hal")
        else:
            log(f"Resume requested but no prior run found for {config_key}", "hal")
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}{config_key}_{timestamp}"

    # Build command
    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", f"{prefix}{config_key}",
        "--agent_function", agent_function,
        "--agent_dir", str(agent_path),
        "--run_id", run_id,
        "--max_concurrent", str(parallel_tasks),  # HAL's concurrent task execution
    ]

    if continue_run:
        cmd.append("--continue_run")
    else:
        # Even if not resuming an old run, we enable continue_run
        # so that internal retries (e.g. timeouts) don't restart from scratch
        # and create duplicate entries in the JSONL file.
        cmd.append("--continue_run")

    if docker:
        cmd.append("--docker")

    # Always continue on errors so we don't lose progress
    cmd.append("--ignore_errors")

    if max_tasks:
        cmd.extend(["--max_tasks", str(max_tasks)])

    # Pass timeout if provided (overrides default 7500s)
    if timeout:
        cmd.extend(["-A", f"timeout={timeout}"])

    # NOTE: HAL doesn't support --task_ids flag.
    # Task filtering is handled via the modified dataset (dataset_path).
    # - all_tasks_mode=True: dataset_path contains ALL tasks with fixes applied
    # - all_tasks_mode=False: dataset_path contains ONLY tasks with fixes
    # If dataset_path is None, HAL runs all tasks from the default dataset.

    # Add agent args
    for key, value in agent_args.items():
        if isinstance(value, (dict, list)):
            cmd.extend(["-A", f"{key}={json.dumps(value)}"])
        else:
            cmd.extend(["-A", f"{key}={value}"])

    # Add benchmark name as agent arg (some agents need it)
    cmd.extend(["-A", f"benchmark_name={benchmark}"])

    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HAL_HARNESS}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault("WANDB_SILENT", "true")
    env["HAL_PRICING_MODEL_NAME"] = config_key
    env["HAL_WEAVE_PROJECT"] = f"{prefix.rstrip('_')}_{benchmark}"
    if trace_mode:
        env["HAL_TRACE_MODE"] = trace_mode

    # If dataset_path provided, set the appropriate environment variable
    if dataset_path:
        # Look up benchmark name without _hard suffix for env var lookup
        benchmark_base = benchmark.replace("_hard", "").replace("_backend_programming", "")
        env_var = BENCHMARK_DATASET_ENV_VAR.get(benchmark, BENCHMARK_DATASET_ENV_VAR.get(benchmark_base))
        if env_var:
            env[env_var] = str(dataset_path)
            # Verify the file exists and count tasks
            if dataset_path.exists():
                try:
                    suffix = dataset_path.suffix.lower()
                    if suffix == ".jsonl":
                        with open(dataset_path, "r", encoding="utf-8") as _f:
                            count = sum(1 for line in _f if line.strip())
                        log(f"Custom dataset: {env_var}={dataset_path} ({count} tasks)", "hal")
                    else:
                        import json as _json
                        with open(dataset_path, "r", encoding="utf-8") as _f:
                            _data = _json.load(_f)
                        log(f"Custom dataset: {env_var}={dataset_path} ({len(_data)} tasks)", "hal")
                except Exception as _e:
                    log(f"Custom dataset: {env_var}={dataset_path} (could not count: {_e})", "hal")
            else:
                log(f"WARNING: Custom dataset path does not exist: {dataset_path}", "hal")
        else:
            log(f"WARNING: No dataset env var known for {benchmark}", "hal")
    else:
        log(f"WARNING: No dataset_path provided - HAL will use default dataset", "hal")

    # Ensure Azure/Weave integration
    if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
        env["USE_DIRECT_AZURE"] = "true"

    log(f"Running: {config_key}", "hal")
    log(f"Agent: {agent_path.name}", "hal")
    log(f"Model: {agent_args.get('model_name')}", "hal")
    log(f"Benchmark: {benchmark}", "hal")
    log(f"Parallel tasks: {parallel_tasks}", "hal")
    if all_tasks_mode:
        log(f"Mode: ALL TASKS (fixes applied where available)", "hal")
    else:
        log(f"Mode: FIXES ONLY (filtered dataset)", "hal")
    log(f"Run ID: {run_id}", "hal")

    # Run with retry
    success, error_msg, result = run_with_retry(
        cmd=cmd, env=env, cwd=REPO_ROOT,
        max_retries=3, base_timeout=None,  # No total timeout (infinite)
    )

    # Extract and show dataset loading info from HAL output
    if result and result.stdout:
        for line in result.stdout.split('\n'):
            if '[SciCode]' in line or 'Loading SciCode' in line or 'SCICODE_DATASET_PATH' in line:
                log(f"HAL: {line.strip()}", "debug")

    if not success:
        log(f"Failed: {error_msg}", "hal")
        if result:
            # Print last 1000 chars of both stderr and stdout for debugging
            if result.stderr:
                log(f"Stderr (last 1000 chars):\n{result.stderr[-1000:]}", "hal")
            if result.stdout:
                log(f"Stdout (last 1000 chars):\n{result.stdout[-1000:]}", "hal")
        return False, error_msg, None

    incomplete_msg: Optional[str] = None
    if result and output_has_incomplete_marker(result.stdout, result.stderr):
        incomplete_msg = "HAL reported incomplete tasks; rerun with --resume."

    # Find output trace
    results_dir = RESULTS_DIR / benchmark / run_id
    trace_path = results_dir / f"{run_id}_UPLOAD.json"

    # Also check HAL_HARNESS results as fallback
    if not trace_path.exists():
        alt_results = HAL_HARNESS / "results" / benchmark / run_id
        alt_trace = alt_results / f"{run_id}_UPLOAD.json"
        if alt_trace.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(alt_trace, trace_path)
    if incomplete_msg is None:
        submissions_path = results_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
        if not submissions_path.exists():
            alt_results = HAL_HARNESS / "results" / benchmark / run_id
            alt_submissions = alt_results / f"{run_id}_RAW_SUBMISSIONS.jsonl"
            if alt_submissions.exists():
                submissions_path = alt_submissions
        total_tasks = count_dataset_tasks(dataset_path)
        completed_tasks = count_completed_submissions(submissions_path)
        if total_tasks is not None and completed_tasks is not None and completed_tasks < total_tasks:
            incomplete_msg = f"Incomplete tasks: completed {completed_tasks}/{total_tasks}"

    if trace_path.exists():
        # Copy to traces directory
        dest = TRACES_DIR / f"{prefix}{trace_path.name}"
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(trace_path, dest)
        log(f"Trace saved: {dest.name}", "hal")
        if incomplete_msg:
            return False, incomplete_msg, dest
        return True, "Success", dest

    if incomplete_msg:
        return False, incomplete_msg, None
    return True, "Success (no trace)", None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark fix runner - runs ALL tasks with fixes."
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark", "-b",
        help="Benchmark name (scicode, scienceagentbench, corebench, colbench, usaco)."
    )
    parser.add_argument(
        "--all-benchmarks", action="store_true",
        help="Run ALL benchmarks that have fixes available."
    )

    # Configuration selection
    parser.add_argument(
        "--config", "-c", dest="configs", action="append",
        help="Config key(s) to run. Can be repeated."
    )
    parser.add_argument(
        "--all-configs", action="store_true",
        help="Run all configurations in the config file."
    )
    parser.add_argument(
        "--agent", "-a",
        help="Filter configs by agent name (e.g., 'scicode_tool_calling_agent')."
    )
    parser.add_argument(
        "--model-filter", "-m",
        help="Filter configs by model pattern (e.g., 'gpt-5')."
    )

    # Output
    parser.add_argument(
        "--prefix", "-p", default="run_",
        help="Prefix for run IDs and output files (default: run_)."
    )

    # Execution options
    parser.add_argument(
        "--docker", action="store_true",
        help="Run with Docker isolation."
    )
    parser.add_argument(
        "--parallel-models", type=int, default=1,
        help="Number of model configurations to run concurrently (default: 1)."
    )
    parser.add_argument(
        "--parallel-tasks", type=int, default=1,
        help="Number of tasks to run concurrently within each HAL evaluation (default: 1). "
             "Uses HAL's --max_concurrent flag."
    )
    parser.add_argument(
        "--trace-mode",
        help="Set HAL_TRACE_MODE for runs (e.g., local)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume prior runs when available (uses HAL --continue_run with last run_id)."
    )
    parser.add_argument(
        "--max-tasks", type=int,
        help="Maximum tasks per config (for testing)."
    )
    parser.add_argument(
        "--sample-tasks", type=int,
        help="Randomly sample N tasks from the dataset before running."
    )
    parser.add_argument(
        "--sample-seed", type=int,
        help="Seed for --sample-tasks to make selection reproducible."
    )
    parser.add_argument(
        "--all-tasks", action="store_true",
        help="Run ALL tasks in benchmark, not just those with fixes. "
             "Fixes are automatically applied to tasks that have them. "
             "Tasks without fixes run normally."
    )
    parser.add_argument(
        "--no-fix", action="store_true",
        help="Run ALL tasks in benchmark WITHOUT applying any fixes. "
             "Ignores --all-tasks and --fix-only behavior."
    )

    parser.add_argument(
        "--timeout", type=int,
        help="Per-task timeout in seconds (overrides default 7500)."
    )
    
    # Modes
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List available configurations and exit."
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true",
        help="List benchmarks with fixes and exit."
    )
    parser.add_argument(
        "--list-fixes", action="store_true",
        help="List task IDs with fixes for each benchmark."
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate configurations against model quirks and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing."
    )

    args = parser.parse_args()
    if args.sample_tasks is not None and args.sample_tasks <= 0:
        log("ERROR: --sample-tasks must be > 0", "main")
        sys.exit(1)
    if args.sample_seed is not None and args.sample_tasks is None:
        log("WARNING: --sample-seed has no effect without --sample-tasks", "main")

    # List benchmarks mode
    if args.list_benchmarks:
        benchmarks = get_benchmarks_with_fixes()
        print(f"\n{'='*70}")
        print("Benchmarks with fixes available:")
        print(f"{'='*70}\n")
        for b in benchmarks:
            task_count = len(get_task_ids_with_fixes(b))
            hal_name = BENCHMARK_HAL_NAME_MAP.get(b, b)
            if b == "colbench":
                print(f"  {b}: {task_count} tasks (runs as {hal_name} - BACKEND ONLY)")
            else:
                print(f"  {b}: {task_count} tasks")
        print(f"\n{'='*70}\n")
        return

    # Determine which benchmarks to run
    if args.all_benchmarks:
        benchmarks_to_run = get_benchmarks_with_fixes()
        if not benchmarks_to_run:
            log("ERROR: No benchmarks with fixes found", "main")
            sys.exit(1)
        log(f"Running ALL benchmarks with fixes: {', '.join(benchmarks_to_run)}", "main")
    elif args.benchmark:
        benchmarks_to_run = [args.benchmark]
    else:
        if not args.list_configs and not args.list_fixes and not args.validate:
            log("ERROR: Specify --benchmark or --all-benchmarks", "main")
            sys.exit(1)
        benchmarks_to_run = []

    # List fixes mode
    if args.list_fixes:
        if not benchmarks_to_run:
            benchmarks_to_run = get_benchmarks_with_fixes()

        print(f"\n{'='*70}")
        print("Tasks with fixes:")
        print(f"{'='*70}")

        for benchmark in benchmarks_to_run:
            task_ids = get_task_ids_with_fixes(benchmark)
            hal_name = BENCHMARK_HAL_NAME_MAP.get(benchmark, benchmark)
            print(f"\n{benchmark} ({len(task_ids)} tasks) -> HAL: {hal_name}")
            if benchmark == "colbench":
                print("  WARNING: Only colbench_backend_programming is supported!")
            for tid in task_ids:
                print(f"    - {tid}")

        print(f"\n{'='*70}\n")
        return

    # Validate mode
    if args.validate:
        if not benchmarks_to_run:
            benchmarks_to_run = get_benchmarks_with_fixes()

        print(f"\n{'='*70}")
        print("Configuration Validation (Model Quirks)")
        print(f"{'='*70}")

        all_warnings = []
        for benchmark in benchmarks_to_run:
            try:
                config = load_benchmark_config(benchmark)
            except FileNotFoundError:
                print(f"\n[ERROR] {benchmark}: config file not found")
                continue

            entries = get_model_entries(config)
            print(f"\n{benchmark}:")

            for key, entry in entries.items():
                warnings = validate_model_config(key, entry)
                model_id = entry.get('model_id', '?')
                quirks = get_model_quirks(model_id)

                status = "OK" if not warnings else "WARN"
                print(f"  [{status}] {key}")
                print(f"       model: {model_id}")
                print(f"       quirks: temp={quirks['supports_temperature']}, stop={quirks['supports_stop']}, reasoning={quirks['supports_reasoning_effort']}")

                for w in warnings:
                    print(w)
                    all_warnings.append(w)

        print(f"\n{'='*70}")
        print(f"Total warnings: {len(all_warnings)}")
        print(f"{'='*70}\n")
        return

    # Process each benchmark
    all_results: List[Tuple[str, str, bool, str, Optional[Path]]] = []

    for benchmark in benchmarks_to_run:
        # Show ColBench warning
        if benchmark == "colbench":
            print(COLBENCH_WARNING)

        # Prepare CoreBench: decrypt dataset + download capsules
        if benchmark in ("corebench", "corebench_hard") and (args.all_tasks or args.no_fix):
            if not prepare_corebench():
                log(f"ERROR: CoreBench preparation failed, skipping", "main")
                continue

        # Prepare SciCode: download test data
        if benchmark == "scicode":
            if not prepare_scicode():
                log(f"ERROR: SciCode preparation failed, skipping", "main")
                continue

        # Get HAL benchmark name
        hal_benchmark = BENCHMARK_HAL_NAME_MAP.get(benchmark, benchmark)

        # Get task IDs with fixes
        task_ids_with_fixes = get_task_ids_with_fixes(benchmark)

        # For --all-tasks mode, we need at least some fixes OR explicit benchmark request
        # For --no-fix mode, we run regardless of fixes
        if not args.all_tasks and not args.no_fix and not task_ids_with_fixes:
            log(f"WARNING: No fixes found for {benchmark}, skipping", "main")
            continue

        log(f"Benchmark: {benchmark} (HAL: {hal_benchmark})", "main")
        log(f"Tasks with fixes: {len(task_ids_with_fixes)}", "main")

        # Handle dataset creation for both modes
        # NOTE: HAL doesn't support --task_ids, so we create filtered datasets instead
        dataset = None
        fixes = {}
        modified_dataset_path = None

        if args.no_fix:
            log(f"MODE: --no-fix (run ENTIRE benchmark WITHOUT fixes)", "main")
            
            # Load full dataset
            dataset = load_benchmark_dataset(benchmark)
            if not dataset:
                log(f"ERROR: Could not load dataset for {benchmark}", "main")
                all_results.append((benchmark, "setup", False, "Could not load dataset", None))
                continue
            
            log(f"Loaded {len(dataset)} total tasks from dataset", "main")
            
            # No fixes applied
            modified_dataset = dataset
            
        elif args.all_tasks:
            log(f"MODE: --all-tasks (run ENTIRE benchmark with fixes applied)", "main")

            # Load full dataset
            dataset = load_benchmark_dataset(benchmark)
            if not dataset:
                log(f"ERROR: Could not load dataset for {benchmark}", "main")
                all_results.append((benchmark, "setup", False, "Could not load dataset", None))
                continue

            log(f"Loaded {len(dataset)} total tasks from dataset", "main")

            # Load all fixes
            fixes = load_all_fixes(benchmark)

            # Create modified dataset with fixes applied (ALL tasks)
            modified_dataset, changes, with_fixes, without_fixes = create_modified_dataset(
                dataset, fixes, benchmark, verbose=False
            )

            log(f"Modified dataset: {with_fixes} tasks with fixes, {without_fixes} tasks without fixes", "main")

        else:
            # Non-all-tasks mode: Create dataset with ONLY tasks that have fixes
            log(f"MODE: fixes-only (run only tasks with fixes)", "main")

            # Load full dataset
            dataset = load_benchmark_dataset(benchmark)
            if not dataset:
                log(f"ERROR: Could not load dataset for {benchmark}", "main")
                all_results.append((benchmark, "setup", False, "Could not load dataset", None))
                continue

            # Get task ID field for filtering
            id_field = BENCHMARK_TASK_ID_FIELD.get(benchmark, "id")

            # Load all fixes
            fixes = load_all_fixes(benchmark)

            # Create dataset with ONLY tasks that have fixes
            filtered_dataset = []
            for task in dataset:
                task_id = str(task.get(id_field, ""))
                if task_id in fixes:
                    # Apply fix and add to filtered dataset
                    modified_task, changes = apply_fix_to_task(task, fixes[task_id], benchmark)
                    filtered_dataset.append(modified_task)
                    if changes:
                        log(f"Applied {len(changes)} fix(es) to task {task_id}", "fix")

            modified_dataset = filtered_dataset
            log(f"Filtered dataset: {len(modified_dataset)} tasks with fixes (from {len(dataset)} total)", "main")

        if args.sample_tasks:
            try:
                modified_dataset = sample_dataset_tasks(
                    modified_dataset,
                    benchmark,
                    args.sample_tasks,
                    args.sample_seed,
                )
            except ValueError as e:
                log(f"ERROR: {e}", "main")
                continue

        # Save modified dataset to temp file
        # Different benchmarks use different formats:
        # - ColBench: JSONL (one JSON object per line)
        # - Others: JSON array
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if benchmark in ("colbench", "colbench_backend_programming"):
            # ColBench uses JSONL format
            modified_dataset_path = TMP_DIR / f"{benchmark}_modified_{timestamp}.jsonl"
            with open(modified_dataset_path, "w") as f:
                for task in modified_dataset:
                    f.write(json.dumps(task, default=str) + "\n")
            log(f"Saved modified dataset (JSONL format) to: {modified_dataset_path}", "main")
        else:
            # Other benchmarks use JSON array format
            modified_dataset_path = TMP_DIR / f"{benchmark}_modified_{timestamp}.json"
            modified_dataset_path.write_text(json.dumps(modified_dataset, indent=2, default=str))
            log(f"Saved modified dataset (JSON format) to: {modified_dataset_path}", "main")
        
        # Explicitly log task count in a format the watcher can parse: (N tasks)
        log(f"Total dataset size: ({len(modified_dataset)} tasks)", "main")

        # Load configuration
        try:
            config = load_benchmark_config(benchmark)
        except FileNotFoundError as e:
            log(f"ERROR: {e}", "main")
            continue

        entries = get_model_entries(config)

        log(f"Found {len(entries)} configurations", "main")

        # List configs mode
        if args.list_configs:
            print(f"\n{'='*70}")
            print(f"Available configurations for {benchmark}")
            print(f"{'='*70}\n")

            # Group by agent
            by_agent: Dict[str, List[Tuple[str, Dict]]] = {}
            for key, entry in entries.items():
                agent_dir = entry.get("agent_dir", "unknown")
                agent_name = Path(agent_dir).name
                by_agent.setdefault(agent_name, []).append((key, entry))

            for agent_name, configs in sorted(by_agent.items()):
                print(f"\n{agent_name}:")
                for key, entry in configs:
                    model_id = entry.get("model_id", "?")
                    effort = entry.get("reasoning_effort", "")
                    effort_str = f" [reasoning={effort}]" if effort else ""
                    temp = entry.get("temperature", "")
                    temp_str = f" [temp={temp}]" if temp else ""
                    print(f"  {key}")
                    print(f"    model: {model_id}{effort_str}{temp_str}")

            print(f"\n{'='*70}")
            continue

        # Determine which configs to run
        selected: Dict[str, Dict[str, Any]] = {}

        if args.configs:
            for key in args.configs:
                if key in entries:
                    selected[key] = entries[key]
                else:
                    log(f"WARNING: Config '{key}' not found", "main")

        elif args.all_configs:
            selected = entries.copy()

        else:
            log("ERROR: Specify --config or --all-configs", "main")
            continue

        # Apply filters
        if args.agent:
            selected = {
                k: v for k, v in selected.items()
                if args.agent in v.get("agent_dir", "")
            }
            log(f"Filtered to {len(selected)} configs with agent '{args.agent}'", "main")

        if args.model_filter:
            pattern = args.model_filter.lower()
            selected = {
                k: v for k, v in selected.items()
                if pattern in v.get("model_id", "").lower() or pattern in k.lower()
            }
            log(f"Filtered to {len(selected)} configs matching '{args.model_filter}'", "main")

        if not selected:
            log(f"No configurations selected for {benchmark}", "main")
            continue

        # Ensure prefix ends with underscore
        prefix = args.prefix
        if not prefix.endswith("_"):
            prefix = prefix + "_"

        # Dry run
        if args.dry_run:
            print(f"\n{'='*70}")
            print(f"DRY RUN - {benchmark} (HAL: {hal_benchmark})")
            print(f"{'='*70}")

            if args.no_fix:
                print(f"\nMODE: --no-fix (run ENTIRE benchmark WITHOUT fixes)")
                if dataset:
                    print(f"Total tasks in dataset: {len(dataset)}")
            elif args.all_tasks:
                print(f"\nMODE: --all-tasks (run ENTIRE benchmark)")
                if dataset:
                    print(f"Total tasks in dataset: {len(dataset)}")
                print(f"Tasks with fixes: {len(task_ids_with_fixes)}")
                print(f"Tasks without fixes (run normally): {len(dataset) - len(task_ids_with_fixes) if dataset else '?'}")
            else:
                print(f"\nMODE: fixes-only (run only tasks with fixes)")
                print(f"Tasks with fixes ({len(task_ids_with_fixes)}):")
                for tid in task_ids_with_fixes[:10]:
                    print(f"  - {tid}")
                if len(task_ids_with_fixes) > 10:
                    print(f"  ... and {len(task_ids_with_fixes) - 10} more")

            print(f"\nConfigurations ({len(selected)}):")
            for key, entry in selected.items():
                agent_path, agent_func = get_agent_info(entry)
                agent_args = build_agent_args(entry)
                model_id = entry.get('model_id', '?')
                quirks = get_model_quirks(model_id)
                print(f"\n  {key}:")
                print(f"    Agent: {agent_path.name}")
                print(f"    Model: {agent_args.get('model_name')}")
                if "reasoning_effort" in agent_args:
                    print(f"    Reasoning: {agent_args['reasoning_effort']}")
                if "temperature" in agent_args:
                    print(f"    Temperature: {agent_args['temperature']}")
                print(f"    Quirks: temp={quirks['supports_temperature']}, stop={quirks['supports_stop']}")

            print(f"\n{'='*70}")
            print(f"Parallel models: {args.parallel_models}")
            print(f"Parallel tasks (per model): {args.parallel_tasks}")
            if args.sample_tasks:
                seed_label = args.sample_seed if args.sample_seed is not None else "random"
                print(f"Sample tasks: {args.sample_tasks} (seed={seed_label})")
            print(f"Resume: {args.resume}")
            print(f"Docker: {args.docker}")
            print(f"All tasks mode: {args.all_tasks}")
            if args.trace_mode:
                print(f"HAL_TRACE_MODE: {args.trace_mode}")
            print(f"{'='*70}")

            # Clean up temp dataset if created for dry run
            if modified_dataset_path and modified_dataset_path.exists():
                modified_dataset_path.unlink()

            continue

        # Run evaluations
        log(f"Running {len(selected)} configurations with prefix '{prefix}'", "main")
        log(f"Parallel models: {args.parallel_models}, Parallel tasks: {args.parallel_tasks}", "main")
        log(f"Resume: {args.resume}", "main")
        if args.all_tasks:
            log(f"Mode: ALL TASKS (fixes auto-applied where available)", "main")
        else:
            log(f"Mode: FIXES ONLY ({len(task_ids_with_fixes)} tasks)", "main")

        results: List[Tuple[str, bool, str, Optional[Path]]] = []
        lock = threading.Lock()

        # Capture these for the closure
        _dataset_path = modified_dataset_path
        _all_tasks_mode = args.all_tasks

        def run_job(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, bool, str, Optional[Path]]:
            key, entry = item
            success, msg, trace = run_hal_eval(
                benchmark=hal_benchmark,
                config_key=key,
                entry=entry,
                prefix=prefix,
                resume=args.resume,
                docker=args.docker,
                max_tasks=args.max_tasks,
                parallel_tasks=args.parallel_tasks,
                dataset_path=_dataset_path,  # Modified dataset with fixes applied
                all_tasks_mode=_all_tasks_mode,
                trace_mode=args.trace_mode,
                timeout=args.timeout,
            )
            return key, success, msg, trace

        jobs = list(selected.items())

        # Per-model timeout (seconds) - ensures no single model blocks the entire run
        # Default: None (infinite) to allow full runs
        MODEL_TIMEOUT = None

        try:
            if args.parallel_models > 1:
                from concurrent.futures import TimeoutError as FuturesTimeoutError
                with ThreadPoolExecutor(max_workers=args.parallel_models) as executor:
                    futures = {executor.submit(run_job, job): job for job in jobs}
                    for future in as_completed(futures):
                        job_key, job_entry = futures[future]
                        try:
                            result = future.result(timeout=MODEL_TIMEOUT)
                            with lock:
                                results.append(result)
                                key, success, msg, _ = result
                                status = "SUCCESS" if success else "FAILED"
                                log(f"[{len(results)}/{len(jobs)}] {status}: {key}", "main")
                        except FuturesTimeoutError:
                            # Model run timed out - log and continue with other models
                            with lock:
                                error_msg = f"TIMEOUT after {MODEL_TIMEOUT}s"
                                results.append((job_key, False, error_msg, None))
                                log(f"[{len(results)}/{len(jobs)}] TIMEOUT: {job_key} ({error_msg})", "main")
                        except Exception as e:
                            with lock:
                                error_msg = f"Exception: {str(e)}"
                                results.append((job_key, False, error_msg, None))
                                log(f"[{len(results)}/{len(jobs)}] ERROR: {job_key} ({error_msg})", "main")
            else:
                for i, job in enumerate(jobs):
                    result = run_job(job)
                    results.append(result)
                    key, success, msg, _ = result
                    status = "SUCCESS" if success else "FAILED"
                    log(f"[{i+1}/{len(jobs)}] {status}: {key}", "main")
        finally:
            # Clean up temp dataset file after all jobs complete
            if modified_dataset_path and modified_dataset_path.exists():
                try:
                    modified_dataset_path.unlink()
                    log(f"Cleaned up temp dataset: {modified_dataset_path.name}", "main")
                except Exception as e:
                    log(f"Warning: Failed to clean up temp dataset: {e}", "main")

        # Collect results
        for r in results:
            all_results.append((benchmark, r[0], r[1], r[2], r[3]))

    # Final Summary
    if all_results:
        succeeded = [r for r in all_results if r[2]]
        failed = [r for r in all_results if not r[2]]

        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total: {len(all_results)}")
        print(f"Succeeded: {len(succeeded)}")
        print(f"Failed: {len(failed)}")

        if succeeded:
            print("\nSuccessful runs:")
            for benchmark, key, _, _, trace in succeeded:
                trace_name = trace.name if trace else "no trace"
                print(f"  - [{benchmark}] {key}: {trace_name}")

        if failed:
            print("\nFailed runs:")
            for benchmark, key, _, msg, _ in failed:
                print(f"  - [{benchmark}] {key}: {msg}")

        print(f"{'='*70}\n")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
