#!/usr/bin/env python3
"""
Apply CoreBench fix packages, run HAL evaluations, and re-grade the new traces.

Usage:
    python scripts/run_corebench_fixes.py \
        --fixes-root fixes/corebench_hard \
        --agent-dir hal-harness/agents/hal_generalist_agent \
        --agent-args agent_args.azure.json \
        --rubric-model azure_openai:o3-mini
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env file early to ensure Azure config is available
from dotenv import load_dotenv
load_dotenv()

# Remove proxy URLs if USE_DIRECT_AZURE is enabled
if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
    for key in ['OPENAI_BASE_URL', 'OPENAI_API_BASE', 'LITELLM_BASE_URL']:
        if key in os.environ:
            del os.environ[key]
    print("[INFO] Direct Azure mode: removed proxy URLs from environment")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Track temporary conda environments for cleanup
_temp_envs: Set[str] = set()
_temp_envs_lock = Lock()
_cleanup_done = False


def _register_temp_env(env_name: str) -> None:
    """Register a temporary environment for cleanup."""
    with _temp_envs_lock:
        _temp_envs.add(env_name)


def _unregister_temp_env(env_name: str) -> None:
    """Unregister a temporary environment after cleanup."""
    with _temp_envs_lock:
        _temp_envs.discard(env_name)


def _cleanup_all_temp_envs() -> None:
    """Clean up all registered temporary conda environments."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    with _temp_envs_lock:
        envs_to_clean = list(_temp_envs)

    if not envs_to_clean:
        return

    print(f"\n[cleanup] Cleaning up {len(envs_to_clean)} temporary conda environment(s)...")
    for env_name in envs_to_clean:
        try:
            _delete_conda_env(env_name, quiet=True)
        except Exception as e:
            print(f"[cleanup] Warning: Failed to clean up {env_name}: {e}")


def _signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    print(f"\n[signal] Received signal {signum}, cleaning up...")
    _cleanup_all_temp_envs()
    sys.exit(128 + signum)


# Register cleanup handlers
atexit.register(_cleanup_all_temp_envs)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
sys.path.insert(0, str(REPO_ROOT / "hal-harness"))

DEFAULT_COREBENCH_DATA = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "corebench" / "core_test.json"
SMOLAGENTS_SRC = REPO_ROOT / "hal-harness" / "agents" / "open_deep_research" / "src" / "smolagents"

from hal.debugger.fix_loader import (  # type: ignore
    load_fix_package,
    apply_agent_overlay,
    apply_agent_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run HAL eval for CoreBench fixes.")
    parser.add_argument(
        "--fixes-root",
        default=str(REPO_ROOT / "fixes" / "corebench_hard"),
        help="Directory containing per-capsule fix packages (default: fixes/corebench_hard).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(REPO_ROOT / "hal-harness" / "agents" / "hal_generalist_agent"),
        help="Path to the base agent directory.",
    )
    parser.add_argument(
        "--agent-args",
        default=str(REPO_ROOT / "agent_args.azure.json"),
        help="Path to the JSON file with agent arguments (model_name, etc.).",
    )
    parser.add_argument(
        "--rubric-model",
        required=False,
        help="Model identifier for rubric evaluation (e.g., azure_openai:o3-mini or openai:o3-mini).",
    )
    parser.add_argument(
        "--skip-rubrics",
        action="store_true",
        help="Skip running main.py evaluate (per-task and merged trace).",
    )
    parser.add_argument(
        "--benchmark",
        default="corebench_hard",
        help="Benchmark variant to evaluate (default: corebench_hard).",
    )
    parser.add_argument(
        "--corebench-dataset",
        help=(
            "Path to the CoreBench task dataset JSON (core_test.json). "
            "If omitted, auto-detect under hal-harness/hal/benchmarks."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix to prepend to per-capsule run_ids and the WANDB project/group (default: empty).",
    )
    parser.add_argument(
        "--merged-trace-output",
        help="Optional output path for the merged trace JSON (must end with .json).",
    )
    parser.add_argument(
        "--merged-run-id",
        help="Optional run_id to store in the merged trace metadata (defaults to auto-generated).",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run hal-eval with the --docker flag.",
    )
    parser.add_argument(
        "--vm",
        action="store_true",
        help="Run hal-eval with the --vm flag (use VM execution, e.g. for GPU tasks).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help=(
            "Optionally override WANDB_MODE for HAL runs. "
            "Use 'online' to enable Weave trace fetching, 'offline' to avoid network, "
            "or 'disabled' to turn off wandb/weave entirely."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary agent directories (useful for debugging).",
    )
    parser.add_argument(
        "--max-parallel-capsules",
        type=int,
        default=1,
        help="Maximum number of capsules to run concurrently within this rerun (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing agent requirements before running hal-eval.",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Only run the specified capsule ID(s). Can be repeated.",
    )
    parser.add_argument(
        "--isolated-env",
        action="store_true",
        help="Create a fresh isolated conda environment for this run (recommended for clean evaluations).",
    )
    parser.add_argument(
        "--isolated-env-python",
        default="3.12",
        help="Python version to use for isolated conda environment (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-isolated-env",
        action="store_true",
        help="Do not delete the isolated conda environment after the run (useful for debugging).",
    )
    parser.add_argument(
        "--rubric-csv",
        help="Path to rubric CSV to determine which model failed for each task (for model-specific reruns).",
    )
    parser.add_argument(
        "--model-config",
        default=str(REPO_ROOT / "configs" / "model_to_baseline_corebench.json"),
        help="Path to model_to_baseline_corebench.json mapping model names to agent_args.",
    )
    parser.add_argument(
        "--model",
        help="Force a specific model for all tasks (overrides rubric CSV). Use model key from config.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all models from model config for each task with fixes.",
    )
    parser.add_argument(
        "--list-fixes",
        action="store_true",
        help="List available fixes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without running HAL eval.",
    )
    parser.add_argument(
        "--verify-fixes",
        action="store_true",
        help="Verify fixes are applied correctly without running HAL eval.",
    )
    return parser.parse_args()


def load_agent_args(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Agent args JSON must be an object, got {type(data)}")
    return data


def load_model_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model_to_baseline.json mapping model names to configs."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Model config JSON must be an object, got {type(data)}")
    return data


def load_rubric_task_models(csv_path: Path) -> List[Tuple[str, str]]:
    """Load rubric CSV and return list of (task_id, model) pairs for all failed models.

    Note: o4-mini-04-16 is expanded to both 'openai/o4-mini-2025-04-16' (high) and
    'o4-mini-2025-04-16' (low) since the rubric collapses them.
    """
    import csv
    task_model_pairs: List[Tuple[str, str]] = []
    seen: set = set()

    # Expansion map: models that should be expanded to multiple variants
    expand_models = {
        "o4-mini-04-16": ["openai/o4-mini-2025-04-16", "o4-mini-2025-04-16"],  # high and low
    }

    if not csv_path.exists():
        return task_model_pairs
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            # Get all models that failed from the models_failed column (semicolon-separated)
            models_failed = row.get("models_failed", "").strip()
            if task_id and models_failed:
                for model in models_failed.split(";"):
                    model = model.strip()
                    if not model:
                        continue
                    # Expand if needed (e.g., o4-mini-04-16 -> high and low variants)
                    models_to_add = expand_models.get(model, [model])
                    for m in models_to_add:
                        if (task_id, m) not in seen:
                            task_model_pairs.append((task_id, m))
                            seen.add((task_id, m))
    return task_model_pairs


def get_agent_args_for_model(
    model_key: str,
    model_config: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[Dict[str, object], str]]:
    """Get the agent_args for a specific model.

    Args:
        model_key: The model name/key to look up (used for pricing)
        model_config: The model_to_baseline_corebench.json config

    Returns:
        Tuple of (agent_args dict, pricing_key) where:
        - agent_args contains model_name (from model_id), reasoning_effort, etc.
        - pricing_key is the config key used for pricing lookup
        Returns None if model not found in config.
    """
    if not model_key:
        return None

    # Direct lookup
    pricing_key = model_key
    model_entry = model_config.get(model_key)

    # Try matching by short_name
    if not model_entry:
        for key, entry in model_config.items():
            if entry.get("short_name") == model_key:
                model_entry = entry
                pricing_key = key
                print(f"[model] Short name match: '{model_key}' -> '{key}'")
                break

    # Try matching by model_id
    if not model_entry:
        for key, entry in model_config.items():
            if entry.get("model_id") == model_key:
                model_entry = entry
                pricing_key = key
                break

    if not model_entry:
        return None

    # Build agent_args entirely from model config (no fallback)
    if "model_id" not in model_entry:
        return None

    agent_args: Dict[str, object] = {
        "model_name": model_entry["model_id"],
    }
    if "reasoning_effort" in model_entry:
        agent_args["reasoning_effort"] = model_entry["reasoning_effort"]
    if "max_steps" in model_entry:
        agent_args["max_steps"] = model_entry["max_steps"]
    return agent_args, pricing_key


def normalize_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return "null"
    return str(value)

def sanitize_tls_env(env: Dict[str, str]) -> None:
    """
    Some environments accidentally set invalid CA bundle paths (e.g. "...cacert.pem[/]"),
    which breaks Weave/W&B/requests TLS and can cause long retries/hangs.
    Remove obviously broken overrides and allow defaults to take over.
    """
    for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        value = env.get(key)
        if not value:
            continue
        if "[/]" in value or "[" in value or "]" in value:
            env.pop(key, None)

def _split_channels(raw: str | None) -> List[str]:
    if not raw:
        return []
    parts: List[str] = []
    for item in shlex.split(raw):
        parts.extend(x.strip() for x in item.split(",") if x.strip())
    return [p for p in parts if p]


def _split_packages(raw: str | None) -> List[str]:
    if not raw:
        return []
    aliases = {
        # Some older fix packages used this name; conda-forge ships `xorg-server-xvfb`.
        "xorg-xvfb": "xorg-server-xvfb",
    }
    packages: List[str] = []
    for token in shlex.split(raw):
        token = token.strip()
        if not token:
            continue
        packages.append(aliases.get(token, token))
    return packages


def _conda_executable() -> str | None:
    # Prefer mamba over conda for faster dependency resolution
    # Also check in current env's bin directory
    env_bin = Path(sys.prefix) / "bin"
    mamba_in_env = env_bin / "mamba"
    if mamba_in_env.exists():
        return str(mamba_in_env)
    return shutil.which("mamba") or shutil.which("conda") or os.environ.get("CONDA_EXE")


def _create_isolated_conda_env(
    base_env: str,
    capsule_id: str,
    python_version: str = "3.12",
) -> str:
    """Create an isolated conda environment for a capsule.

    Clones from the base environment to get core dependencies,
    then returns the new environment name.
    """
    conda = _conda_executable()
    if not conda:
        raise FileNotFoundError("conda/mamba executable not found")

    # Generate unique env name
    env_name = f"hal_capsule_{capsule_id}_{uuid.uuid4().hex[:8]}"

    print(f"[conda][create] Creating isolated environment: {env_name}")

    # Clone from base env to get core dependencies (faster than fresh install)
    cmd = [conda, "create", "-n", env_name, "--clone", base_env, "-y"]
    print(f"[conda][create] Cloning from {base_env}...")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # If clone fails, create fresh env with Python
        print(f"[conda][create] Clone failed, creating fresh environment...")
        cmd = [conda, "create", "-n", env_name, f"python={python_version}", "-y"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create conda env: {proc.stderr}")

    # Register for cleanup
    _register_temp_env(env_name)
    print(f"[conda][create] Created environment: {env_name}")

    return env_name


def _delete_conda_env(env_name: str, quiet: bool = False) -> None:
    """Delete a conda environment."""
    conda = _conda_executable()
    if not conda:
        return

    if not quiet:
        print(f"[conda][delete] Removing environment: {env_name}")

    cmd = [conda, "env", "remove", "-n", env_name, "-y"]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode == 0:
        _unregister_temp_env(env_name)
        if not quiet:
            print(f"[conda][delete] Removed environment: {env_name}")
    else:
        if not quiet:
            print(f"[conda][delete] Warning: Failed to remove {env_name}: {proc.stderr}")


def _conda_spec_name(spec: str) -> str:
    for sep in ("==", ">=", "<=", "=", ">", "<"):
        if sep in spec:
            return spec.split(sep, 1)[0].strip()
    return spec.strip()


def _conda_available_package_names(conda: str, *, channels: List[str], names: List[str]) -> set[str] | None:
    unique = sorted({name for name in names if name})
    if not unique:
        return set()
    cmd = [conda, "search", "--json"]
    for channel in channels:
        cmd += ["-c", channel]
    cmd += unique
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    available: set[str] = set()
    # Handle mamba format: {"result": {"pkgs": [{"name": "...", ...}, ...]}}
    if "result" in payload and isinstance(payload.get("result"), dict):
        pkgs = payload["result"].get("pkgs", [])
        if isinstance(pkgs, list):
            for pkg in pkgs:
                if isinstance(pkg, dict) and pkg.get("name") in unique:
                    available.add(pkg["name"])
        return available
    # Handle conda format: {"package_name": [...], ...}
    for name in unique:
        entries = payload.get(name)
        if isinstance(entries, list) and entries:
            available.add(name)
    return available


def _drop_missing_packages_from_conda_output(output: str) -> List[str]:
    if "PackagesNotFoundError" not in output:
        return []
    missing: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("- "):
            missing.append(line[2:].strip())
    return missing


def _install_conda_packages(
    *,
    env_name: str,
    channels: List[str],
    packages: List[str],
) -> None:
    conda = _conda_executable()
    conda_name = Path(conda).name if conda else "conda"
    if not conda:
        raise FileNotFoundError("conda executable not found (needed for HAL_CONDA_PACKAGES).")
    if not packages:
        print(f"[setup][conda] No packages to install")
        return

    print(f"[setup][conda] Using {conda_name} from: {conda}")
    print(f"[setup][conda] Preparing to install {len(packages)} packages into env '{env_name}'")
    print(f"[setup][conda] Channels: {', '.join(channels)}")
    print(f"[setup][conda] Packages: {', '.join(packages[:10])}{'...' if len(packages) > 10 else ''}")

    lock_dir = REPO_ROOT / "log" / "conda_installs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    cache_key = json.dumps(
        {"env": env_name, "channels": channels, "packages": packages},
        sort_keys=True,
    ).encode("utf-8")
    cache_id = hashlib.sha256(cache_key).hexdigest()[:16]
    sentinel = lock_dir / f"{env_name}_{cache_id}.ok"
    lock_path = lock_dir / f"{env_name}.lock"

    if sentinel.exists():
        print(f"[setup][conda] Cache hit: packages already installed (sentinel: {sentinel.name})")
        return

    print(f"[setup][conda] Cache miss: will install packages (cache_id: {cache_id})")

    # Best-effort filter: skip package specs whose names are not present in the configured channels.
    # Some environments mirror only a subset of conda-forge (e.g. no xorg packages).
    print(f"[setup][conda] Checking package availability in channels...")
    names = [_conda_spec_name(spec) for spec in packages]
    available = _conda_available_package_names(conda, channels=channels, names=names)
    if available is not None:
        print(f"[setup][conda] Found {len(available)}/{len(names)} packages available")
        filtered = [spec for spec in packages if _conda_spec_name(spec) in available]
        dropped = sorted({spec for spec in packages if spec not in filtered})
        if dropped:
            print(f"[setup][conda] Skipping unavailable package specs: {', '.join(dropped)}")
        packages = filtered
        if not packages:
            print(f"[setup][conda] No packages available to install, creating sentinel")
            sentinel.write_text("ok (no available packages)\n", encoding="utf-8")
            return
    else:
        print(f"[setup][conda] Could not check availability, proceeding with all packages")

    print(f"[setup][conda] Acquiring install lock...")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            import fcntl  # type: ignore

            # Serialize conda installs across concurrent specs without hanging forever on flock().
            wait_count = 0
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    wait_count += 1
                    if wait_count == 1:
                        print(f"[setup][conda] Waiting for lock (another install in progress)...")
                    elif wait_count % 20 == 0:  # Every 10 seconds
                        print(f"[setup][conda] Still waiting for lock ({wait_count * 0.5:.0f}s elapsed)...")
                    time.sleep(0.5)
            if wait_count > 0:
                print(f"[setup][conda] Lock acquired after {wait_count * 0.5:.1f}s")
            else:
                print(f"[setup][conda] Lock acquired immediately")
        except Exception:
            print(f"[setup][conda] Lock not available on this platform, proceeding without lock")

        if sentinel.exists():
            print(f"[setup][conda] Another process already installed packages, skipping")
            return

        def run_install(current: List[str]) -> subprocess.CompletedProcess[str]:
            cmd = [conda, "install", "-y", "-n", env_name]
            for channel in channels:
                cmd += ["-c", channel]
            cmd += current
            print(f"[setup][conda] Running: {' '.join(map(str, cmd))}")
            print(f"[setup][conda] Installing {len(current)} packages (this may take several minutes)...")
            start_time = time.time()
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            elapsed = time.time() - start_time
            print(f"[setup][conda] Install command finished in {elapsed:.1f}s (exit code: {result.returncode})")
            return result

        remaining = list(packages)
        for attempt in range(2):
            print(f"[setup][conda] Attempt {attempt + 1}/2: installing {len(remaining)} packages...")
            proc = run_install(remaining)
            if proc.returncode == 0:
                print(f"[setup][conda] SUCCESS: All packages installed successfully")
                print(f"[setup][conda] Creating sentinel file: {sentinel.name}")
                sentinel.write_text("ok\n", encoding="utf-8")
                return
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            missing = _drop_missing_packages_from_conda_output(output)
            if not missing:
                print(f"[setup][conda] FAILED: Install failed with no recoverable missing packages")
                print(f"[setup][conda] stdout: {proc.stdout[:500] if proc.stdout else '(empty)'}...")
                print(f"[setup][conda] stderr: {proc.stderr[:500] if proc.stderr else '(empty)'}...")
                raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
            missing_names = {_conda_spec_name(item) for item in missing if item}
            new_remaining = [
                spec for spec in remaining if _conda_spec_name(spec) not in missing_names and spec not in missing
            ]
            if new_remaining == remaining:
                print(f"[setup][conda] FAILED: Could not reduce package list, giving up")
                raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
            print(f"[setup][conda] Dropping {len(missing_names)} missing packages and retrying: {', '.join(sorted(missing_names))}")
            remaining = new_remaining
            if not remaining:
                print(f"[setup][conda] All packages were unavailable, creating sentinel")
                sentinel.write_text("ok (all packages unavailable)\n", encoding="utf-8")
                return

        print(f"[setup][conda] FAILED: Exhausted all retry attempts")
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


def _python_can_import(module: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def ensure_hal_cli_dependencies(env_name: str, channels: List[str]) -> None:
    # hal.cli requires click at import-time, but it is not consistently present in all envs.
    if _python_can_import("click"):
        return
    print("[setup][hal] Missing python module 'click'; attempting to install via conda.")
    try:
        _install_conda_packages(env_name=env_name, channels=channels or ["conda-forge"], packages=["click"])
        if _python_can_import("click"):
            return
    except Exception as exc:
        print(f"[setup][hal] Conda install for click failed: {exc}")
    print("[setup][hal] Falling back to pip install click.")
    subprocess.run([sys.executable, "-m", "pip", "install", "click"], check=True, cwd=REPO_ROOT)


def build_hal_eval_cmd(
    *,
    benchmark: str,
    agent_name: str,
    agent_dir: Path,
    agent_args: Dict[str, object],
    run_id: str,
    docker: bool,
    vm: bool,
    max_concurrent: int = 1,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", agent_name,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", str(max_concurrent),
    ]
    for key, value in agent_args.items():
        cmd += ["-A", f"{key}={normalize_value(value)}"]
    if docker:
        cmd.append("--docker")
    if vm:
        cmd.append("--vm")
    return cmd


def copy_agent_with_fix(
    base_dir: Path,
    task_id: str,
    fix_root: Path,
    *,
    benchmark: str,
    env_override: Dict[str, str] | None = None,
) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="corebench_fix_"))
    shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True)
    if SMOLAGENTS_SRC.exists():
        shutil.copytree(SMOLAGENTS_SRC, temp_dir / "smolagents", dirs_exist_ok=True)
    fix = load_fix_package(task_id, fixes_root=fix_root, benchmark=benchmark)
    if not fix:
        pass  # Continue to check env_override for pip packages
    else:
        if fix.agent_overlay:
            apply_agent_overlay(temp_dir, fix.agent_overlay)
        if fix.agent_patch:
            apply_agent_patch(temp_dir, fix.agent_patch)

    # Add pip packages from env_override to requirements.txt (for Docker builds)
    if env_override:
        pip_packages = env_override.get("HAL_PIP_PACKAGES", "").split()
        if pip_packages:
            requirements_path = temp_dir / "requirements.txt"
            existing = requirements_path.read_text() if requirements_path.exists() else ""
            # Append pip packages if not already present
            new_packages = [p for p in pip_packages if p not in existing]
            if new_packages:
                with requirements_path.open("a") as f:
                    f.write("\n# Added by fix package\n")
                    for pkg in new_packages:
                        f.write(f"{pkg}\n")
                print(f"[agent] Added {len(new_packages)} pip packages to requirements.txt for {task_id}")

    return temp_dir


def resolve_corebench_dataset_path(override: str | None) -> Path:
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"--corebench-dataset not found: {candidate}")
        return candidate

    if DEFAULT_COREBENCH_DATA.exists():
        return DEFAULT_COREBENCH_DATA

    root = REPO_ROOT / "hal-harness" / "hal" / "benchmarks"
    if root.exists():
        candidates = sorted(root.rglob("core_test.json"))
        if candidates:
            for path in candidates:
                if "corebench" in {part.lower() for part in path.parts}:
                    return path
            return candidates[0]

    raise FileNotFoundError(
        "CoreBench dataset file core_test.json not found. "
        "Ensure hal-harness is present (and any submodules/assets are initialized), "
        "or pass --corebench-dataset /path/to/core_test.json."
    )


def load_corebench_dataset(dataset_path: Path) -> List[dict]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def build_single_task_payload(
    *,
    original_tasks: Dict[str, dict],
    capsule_id: str,
    input_override: Dict[str, object] | None,
) -> str:
    if capsule_id not in original_tasks:
        raise ValueError(f"Capsule {capsule_id} not found in core_test.json")
    task = json.loads(json.dumps(original_tasks[capsule_id]))  # deep copy
    if input_override:
        task.update(input_override)
        if "problem_statement" in input_override:
            original_prompt = task.get("task_prompt", "")
            extra = input_override["problem_statement"]
            task["task_prompt"] = f"{original_prompt}\n\n{extra}" if original_prompt else extra
    return json.dumps([task], indent=2)


def write_filtered_dataset(
    *,
    original_tasks: Dict[str, dict],
    capsule_id: str,
    input_override: Dict[str, object] | None,
    dataset_path: Path,
) -> None:
    payload = build_single_task_payload(
        original_tasks=original_tasks,
        capsule_id=capsule_id,
        input_override=input_override,
    )
    tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(dataset_path)


def recover_dataset_if_needed(dataset_path: Path, dataset_backup: Path) -> None:
    """Restore dataset_path from a leftover backup if a prior run was interrupted."""
    if not dataset_backup.exists():
        return
    print(
        f"[WARN] Found leftover dataset backup at {dataset_backup}. "
        f"Restoring {dataset_path} from backup before continuing."
    )
    dataset_path.unlink(missing_ok=True)
    shutil.move(dataset_backup, dataset_path)

def install_agent_requirements(agent_dir: Path) -> None:
    requirements = agent_dir / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"requirements.txt not found at {requirements}")
    print(f"[setup][pip] Checking requirements from {requirements}")
    req_fingerprint = hashlib.sha256(
        (sys.executable + "\n" + requirements.read_text(encoding="utf-8")).encode("utf-8")
    ).hexdigest()[:16]
    sentinel = REPO_ROOT / ".agent_requirements_installed" / f"{agent_dir.name}_{req_fingerprint}.ok"
    lock_path = REPO_ROOT / ".agent_requirements_install.lock"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        print(f"[setup][pip] Acquiring pip install lock...")
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            print(f"[setup][pip] Lock acquired")
        except Exception:
            # Best-effort: on platforms without fcntl, run without a lock.
            print(f"[setup][pip] Lock not available, proceeding without lock")

        if sentinel.exists():
            print(f"[setup][pip] Cache hit: requirements already installed (sentinel: {sentinel.name})")
            return

        print(f"[setup][pip] Cache miss: installing requirements (fingerprint: {req_fingerprint})")
        env = os.environ.copy()
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PIP_NO_INPUT", "1")

        try:
            print(f"[setup][pip] Running: pip install -r {requirements}")
            start_time = time.time()
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                check=True,
                env=env,
            )
            elapsed = time.time() - start_time
            print(f"[setup][pip] SUCCESS: Requirements installed in {elapsed:.1f}s")
        except subprocess.CalledProcessError as exc:
            # Common in parallel runs if the env is mid-mutation or already partially corrupted.
            print(f"[setup][pip] WARN: pip install failed, attempting minimal repair: {exc}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--ignore-installed", "rich"],
                check=False,
                env=env,
            )
            print(f"[setup][pip] Retrying pip install after repair...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                check=True,
                env=env,
            )
            print(f"[setup][pip] SUCCESS: Requirements installed after repair")
        print(f"[setup][pip] Creating sentinel: {sentinel.name}")
        sentinel.write_text("ok\n", encoding="utf-8")
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass

FAILED_TASKS_FILE = REPO_ROOT / ".tmp" / "corebench_failed_tasks.json"


def is_retryable_error(error_msg: str) -> bool:
    """Check if an error is retryable (TRAPI timeout, auth, rate limit)."""
    retryable_patterns = [
        # Timeouts
        "timeout", "timed out", "connection reset", "connection refused", "connection error",
        # HTTP errors
        "503", "504", "502", "500", "429", "rate limit",
        # Auth/Token errors (CRITICAL: TRAPI token expiration)
        "401", "403", "unauthorized", "authentication",
        "invalid token", "expired token", "invalid or expired token", "token expired", "authenticationerror",
        # Other retryable errors
        "invalid_request_error", "insufficient_quota", "overloaded", "overloaded_error",
        "network error", "broken pipe", "EOF occurred", "service unavailable", "bad gateway",
    ]
    error_lower = str(error_msg).lower()
    return any(pattern in error_lower for pattern in retryable_patterns)


def save_failed_task(task_id: str, model_id: str, error: str) -> None:
    """Save failed task to file for later retry."""
    failed_tasks = {}
    if FAILED_TASKS_FILE.exists():
        try:
            failed_tasks = json.loads(FAILED_TASKS_FILE.read_text())
        except Exception:
            pass

    key = f"{task_id}_{model_id}"
    failed_tasks[key] = {
        "task_id": task_id,
        "model_id": model_id,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    FAILED_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILED_TASKS_FILE.write_text(json.dumps(failed_tasks, indent=2))


def run_with_retry(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    task_id: str,
    model_id: str,
    max_retries: int = 3,
) -> Tuple[bool, str, Optional[subprocess.CompletedProcess]]:
    """Run subprocess with retry logic for TRAPI timeouts and auth errors."""
    import random

    for attempt in range(max_retries):
        try:
            # Only print retry message on actual retries (not first attempt)
            if attempt > 0:
                print(f"[retry] Attempt {attempt + 1}/{max_retries}")
            result = subprocess.run(
                cmd, cwd=cwd, env=env, capture_output=True, text=True,
            )

            stderr, stdout = result.stderr or "", result.stdout or ""
            combined_output = stderr + stdout

            # Success
            if result.returncode == 0:
                return True, "Success", result

            # Check if error is retryable
            if is_retryable_error(combined_output):
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 5))
                print(f"[retry] Retryable error detected. Waiting {wait_time:.1f}s before retry...")
                print(f"[retry] Error snippet: {combined_output[:200]}")
                time.sleep(wait_time)
                continue
            else:
                print(f"[retry] Non-retryable error: {combined_output[:200]}")
                save_failed_task(task_id, model_id, combined_output[:500])
                return False, f"Exit code {result.returncode}", result

        except Exception as e:
            error_msg = str(e)
            print(f"[retry] Exception on attempt {attempt + 1}/{max_retries}: {error_msg}")

            if is_retryable_error(error_msg) and attempt < max_retries - 1:
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 5))
                print(f"[retry] Retrying after {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                save_failed_task(task_id, model_id, error_msg)
                return False, error_msg, None

    save_failed_task(task_id, model_id, "Max retries exceeded")
    return False, "Max retries exceeded", None


def _slugify(value: str, fallback: str) -> str:
    # Keep readable identifiers for run_ids / filenames.
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or fallback


def build_task_run_id(benchmark: str, agent_args: Dict[str, object], capsule_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    api_model = str(agent_args.get("api_model_id") or agent_args.get("model_name") or "model")
    model_slug = _slugify(api_model.replace("/", "_"), "model")
    effort = str(agent_args.get("reasoning_effort") or "").strip().lower()
    effort_part = f"_{_slugify(effort, '')}" if effort else ""
    capsule_slug = _slugify(capsule_id, "task")
    bench_slug = _slugify(benchmark, "benchmark")
    return f"{model_slug}{effort_part}_{capsule_slug}_{bench_slug}_{timestamp}"


def merge_trace_files(
    trace_paths: List[Path],
    benchmark: str,
    agent_dir: Path,
    *,
    output_path: Path | None = None,
    merged_run_id: str | None = None,
) -> Path:
    first_trace = json.loads(trace_paths[0].read_text(encoding="utf-8"))
    config: Dict[str, Any] = first_trace.get("config", {})
    agent_name = config.get("agent_name") or agent_dir.name
    model_name = (
        config.get("agent_args", {}).get("model_name")
        if isinstance(config.get("agent_args"), dict)
        else None
    ) or "model"

    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        agent_slug = _slugify(agent_dir.name, "agent")
        model_slug = _slugify(str(model_name), "model")
        run_id = merged_run_id or f"{benchmark}_{agent_slug}{model_slug}_{timestamp}_FIXED"
        output_path = REPO_ROOT / "traces" / f"{run_id}_UPLOAD.json"
    else:
        if output_path.suffix != ".json":
            raise ValueError(f"--merged-trace-output must be a .json file path, got {output_path}")
        run_id = merged_run_id or output_path.stem
        if run_id.endswith("_UPLOAD"):
            run_id = run_id[: -len("_UPLOAD")]

    merge_cmd: List[str] = [
        sys.executable,
        "scripts/merge_traces.py",
    ]
    for path in trace_paths:
        merge_cmd += ["--input", str(path)]
    merge_cmd += [
        "--output",
        str(output_path),
        "--run-id",
        run_id,
        "--agent-name",
        agent_name,
        "--date",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "--force",
    ]
    subprocess.run(merge_cmd, check=True, cwd=REPO_ROOT)
    return output_path


def evaluate_trace(trace_path: Path, rubric_model: str, output_dir: Path) -> None:
    eval_cmd = [
        sys.executable,
        "main.py",
        "evaluate",
        "--trace-file",
        str(trace_path),
        "--rubrics-dir",
        str(REPO_ROOT / "rubrics"),
        "--output-dir",
        str(output_dir),
        "--rubric-model",
        rubric_model,
        "--yes",
    ]
    subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)


def run_one_capsule(
    *,
    capsule_id: str,
    fixes_base: Path,
    dataset_dir: Path,
    dataset_payload: str,
    agent_dir: Path,
    agent_args: Dict[str, object],
    benchmark: str,
    prefix: str,
    docker: bool,
    vm: bool,
    wandb_mode: str | None,
    env_override: Dict[str, str] | None,
    rubric_model: str,
    rubric_output_dir: Path,
    skip_rubrics: bool,
    keep_temp: bool,
    conda_env: str = "hal",
    use_isolated_env: bool = False,
    keep_isolated_env: bool = False,
) -> Path:
    dataset_path = dataset_dir / f"core_test_{capsule_id}_{uuid.uuid4().hex}.json"
    dataset_path.write_text(dataset_payload, encoding="utf-8")

    # For Docker mode: packages are installed INSIDE the Docker container via requirements.txt
    # For non-Docker mode: install packages on host conda env
    isolated_env_name: str | None = None
    active_conda_env = conda_env

    if not docker:
        # Only do host-side conda installation for non-Docker mode
        if use_isolated_env:
            try:
                isolated_env_name = _create_isolated_conda_env(
                    base_env=conda_env,
                    capsule_id=capsule_id,
                )
                active_conda_env = isolated_env_name
            except Exception as exc:
                print(f"[capsule][conda] WARNING: Failed to create isolated env for {capsule_id}: {exc}")
                print(f"[capsule][conda] Falling back to shared env: {conda_env}")

        # Install per-capsule conda packages into the host environment
        if env_override:
            channels = _split_channels(env_override.get("HAL_CONDA_CHANNELS")) or ["conda-forge"]
            packages = _split_packages(env_override.get("HAL_CONDA_PACKAGES"))
            if packages:
                print(f"[capsule][conda] Installing {len(packages)} packages for {capsule_id} into {active_conda_env}...")
                try:
                    _install_conda_packages(env_name=active_conda_env, channels=channels, packages=packages)
                except Exception as exc:
                    print(f"[capsule][conda] WARNING: Failed to install packages for {capsule_id}: {exc}")
    else:
        print(f"[capsule] Docker mode: packages will be installed inside container via requirements.txt")

    # Copy agent and apply fixes (also adds pip packages to requirements.txt for Docker)
    temp_agent_dir = copy_agent_with_fix(
        agent_dir, capsule_id, fixes_base,
        benchmark=benchmark,
        env_override=env_override,
    )
    try:
        prefix_value = str(prefix or "").strip()
        prefix_part = f"{_slugify(prefix_value, '')}_" if prefix_value else ""
        run_id = prefix_part + build_task_run_id(benchmark, agent_args, capsule_id)
        print(f"[capsule][start] capsule_id={capsule_id} run_id={run_id} benchmark={benchmark}")
        cmd = build_hal_eval_cmd(
            benchmark=benchmark,
            agent_name=f"hal_generalist_agent ({capsule_id})",
            agent_dir=temp_agent_dir,
            agent_args=agent_args,
            run_id=run_id,
            docker=docker,
            vm=vm,
        )
        print(f"[hal-eval] {' '.join(cmd)}")
        hal_env = os.environ.copy()
        sanitize_tls_env(hal_env)
        if env_override:
            for key, value in env_override.items():
                if value is None:
                    continue
                hal_env[str(key)] = str(value)
        extra_path = str(REPO_ROOT / "hal-harness")
        hal_env["PYTHONPATH"] = (
            f"{extra_path}{os.pathsep}{hal_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        )
        hal_env["HAL_COREBENCH_DATASET_PATH"] = str(dataset_path)
        hal_env.setdefault("HAL_PRICING_MODEL_NAME", str(agent_args.get("model_name", "")))
        hal_env.setdefault("WANDB_SILENT", "true")
        if prefix_value:
            base_project = hal_env.get("WANDB_PROJECT") or "hal"
            prefix_slug = _slugify(prefix_value, "run")
            if not base_project.startswith(f"{prefix_slug}_"):
                hal_env["WANDB_PROJECT"] = f"{prefix_slug}_{base_project}"
            hal_env.setdefault("WANDB_RUN_GROUP", prefix_slug)
        if wandb_mode == "disabled":
            hal_env["WANDB_MODE"] = "disabled"
        elif wandb_mode:
            hal_env["WANDB_MODE"] = wandb_mode

        # Use retry wrapper with exponential backoff
        model_id_str = str(agent_args.get("model_name", "unknown"))
        success, error_msg, proc = run_with_retry(
            cmd=cmd,
            env=hal_env,
            cwd=REPO_ROOT,
            task_id=capsule_id,
            model_id=model_id_str,
            max_retries=3,
        )

        if not success:
            output = error_msg if proc is None else ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if output:
                print(f"[hal-eval][error] capsule_id={capsule_id} run_id={run_id}\n{output}\n")
            print(f"[hal-eval][failed] Failed after retries: {error_msg}")
            raise subprocess.CalledProcessError(
                proc.returncode if proc else 1,
                cmd,
                output=proc.stdout if proc else None,
                stderr=proc.stderr if proc else error_msg
            )

        results_dir = REPO_ROOT / "results" / benchmark / run_id
        trace_path = results_dir / f"{run_id}_UPLOAD.json"
        if not trace_path.exists():
            raise FileNotFoundError(f"Expected trace {trace_path} not found.")

        final_trace = REPO_ROOT / "traces" / trace_path.name
        shutil.copyfile(trace_path, final_trace)
        print(f"[trace] Copied {trace_path} -> {final_trace}")
        print(f"[capsule][done] capsule_id={capsule_id} run_id={run_id} trace={final_trace}")

        if not skip_rubrics:
            eval_cmd = [
                sys.executable,
                "main.py",
                "evaluate",
                "--trace-file", str(final_trace),
                "--rubrics-dir", str(REPO_ROOT / "rubrics"),
                "--output-dir", str(rubric_output_dir),
                "--rubric-model", rubric_model,
                "--yes",
            ]
            print(f"[rubric] {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)
            print(f"[capsule][rubric_done] capsule_id={capsule_id} run_id={run_id}")

        return final_trace
    finally:
        dataset_path.unlink(missing_ok=True)
        if not keep_temp:
            shutil.rmtree(temp_agent_dir, ignore_errors=True)
        # Clean up isolated conda environment
        if isolated_env_name and not keep_isolated_env:
            try:
                _delete_conda_env(isolated_env_name)
            except Exception as exc:
                print(f"[capsule][conda] WARNING: Failed to clean up isolated env {isolated_env_name}: {exc}")


def list_available_fixes(fixes_root: Path) -> List[str]:
    """List all task IDs that have fix directories."""
    if not fixes_root.exists():
        return []
    return sorted([p.name for p in fixes_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def ensure_capsules_extracted(capsule_ids: Set[str], benchmark: str = "corebench_hard") -> None:
    """
    Ensure all required capsules are extracted before parallel execution.

    This prevents race conditions when multiple processes try to extract the same
    capsule simultaneously during parallel runs.

    Args:
        capsule_ids: Set of capsule IDs that need to be available
        benchmark: Benchmark name (default: corebench_hard)
    """
    capsules_dir = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "corebench" / "capsules"

    # Check which capsules are missing
    missing_capsules = []
    for capsule_id in sorted(capsule_ids):
        capsule_dir = capsules_dir / capsule_id
        if not capsule_dir.exists():
            missing_capsules.append(capsule_id)

    if not missing_capsules:
        print(f"[capsules] All {len(capsule_ids)} required capsules already extracted")
        return

    print(f"[capsules] Found {len(missing_capsules)} capsules that need extraction:")
    for capsule_id in missing_capsules[:5]:  # Show first 5
        print(f"[capsules]   - {capsule_id}")
    if len(missing_capsules) > 5:
        print(f"[capsules]   ... and {len(missing_capsules) - 5} more")

    print(f"[capsules] Initializing benchmark to extract capsules (this may take a few minutes)...")

    # Run a minimal hal-eval to trigger capsule extraction
    # We use max_tasks=1 but the benchmark __init__ will still check/extract all capsules
    agent_dir = REPO_ROOT / "hal-harness" / "agents" / "hal_generalist_agent"

    try:
        cmd = [
            "hal-eval",
            "--benchmark", benchmark,
            "--agent_dir", str(agent_dir),
            "--agent_function", "main.run",
            "--agent_name", "capsule_extractor",
            "--max_tasks", "0",  # Don't actually run any tasks
            "-A", "model_name=gpt-4o",
        ]

        print(f"[capsules] Running: {' '.join(cmd)}")

        # Run the command - it will initialize the benchmark which extracts capsules
        # We expect this to fail since max_tasks=0, but capsules will be extracted
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        # Check if capsules were extracted successfully
        still_missing = []
        for capsule_id in missing_capsules:
            capsule_dir = capsules_dir / capsule_id
            if not capsule_dir.exists():
                still_missing.append(capsule_id)

        if still_missing:
            print(f"[capsules] WARNING: {len(still_missing)} capsules still not extracted:")
            for capsule_id in still_missing[:5]:
                print(f"[capsules]   - {capsule_id}")
            print(f"[capsules] Stderr: {result.stderr[:500]}")
        else:
            print(f"[capsules] ✓ Successfully extracted all {len(missing_capsules)} capsules")

    except Exception as e:
        print(f"[capsules] WARNING: Failed to pre-extract capsules: {e}")
        print(f"[capsules] Will attempt extraction during individual runs (may cause race conditions)")


def main() -> None:
    args = parse_args()
    fixes_root = Path(args.fixes_root)

    # Handle --list-fixes early
    if args.list_fixes:
        available = list_available_fixes(fixes_root)
        print(f"\n{'='*60}")
        print(f"AVAILABLE FIXES in {fixes_root}")
        print(f"{'='*60}")
        print(f"\nFound {len(available)} fix directories:")
        for fix_id in available:
            print(f"  - {fix_id}")
        print()
        return

    if not args.skip_rubrics and not args.rubric_model:
        raise SystemExit("--rubric-model is required unless --skip-rubrics is set.")
    agent_dir = Path(args.agent_dir)
    dataset_source = resolve_corebench_dataset_path(args.corebench_dataset)
    rubric_output_dir = REPO_ROOT / "rubrics_output"
    rubric_output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config (required - all models must be defined here)
    model_config_path = Path(args.model_config)
    model_config = load_model_config(model_config_path)
    if model_config:
        print(f"[model] Loaded {len(model_config)} model configs from {model_config_path.name}")
        print(f"[model] Available models: {list(model_config.keys())}")
    else:
        print(f"[model] ERROR: No model config found at {model_config_path}")
        print(f"[model] All models must be defined in model_to_baseline_corebench.json")
        return

    # Load task->model pairs from rubric CSV (each task can have multiple models)
    task_model_pairs: List[Tuple[str, str]] = []
    if args.all_models:
        # Will be populated after loading available_fixes
        print(f"[model] --all-models specified, will run all models for each task")
    elif args.rubric_csv:
        rubric_csv_path = Path(args.rubric_csv)
        task_model_pairs = load_rubric_task_models(rubric_csv_path)
        print(f"[model] Loaded {len(task_model_pairs)} (task, model) pairs from {rubric_csv_path.name}")
    elif args.model:
        print(f"[model] Using forced model '{args.model}' for all tasks")
    else:
        print(f"[model] WARNING: No --rubric-csv, --model, or --all-models specified")

    if not args.skip_install:
        install_agent_requirements(agent_dir)

    temp_root = Path(tempfile.mkdtemp(prefix="corebench_dataset_"))
    tasks_data = load_corebench_dataset(dataset_source)
    capsule_map = {item["capsule_id"]: item for item in tasks_data}

    # Build set of available fix capsule IDs
    fix_dirs = sorted([p for p in fixes_root.iterdir() if p.is_dir()])
    available_fixes = {p.name for p in fix_dirs}
    if args.task_ids:
        requested = set(args.task_ids)
        available_fixes = available_fixes & requested
        missing = requested - available_fixes
        if missing:
            print(f"[WARN] Requested task(s) not found under {fixes_root}: {', '.join(sorted(missing))}")
    if not available_fixes:
        print(f"No fix directories found in {fixes_root}")
        return
    print(f"[fixes] Found {len(available_fixes)} fix directories")

    # Pre-extract all required capsules to avoid race conditions during parallel execution
    ensure_capsules_extracted(available_fixes, benchmark=args.benchmark)

    try:
        fixes_base = fixes_root.parent
        # capsule_jobs: (capsule_id, payload, env_override, task_agent_args, pricing_key)
        capsule_jobs: List[tuple[str, str, Dict[str, str] | None, Dict[str, object], str]] = []
        conda_env = "hal"

        # Build task_model_pairs based on flags
        if args.all_models:
            task_model_pairs = [(fix_id, model_key) for fix_id in available_fixes for model_key in model_config.keys()]
            print(f"[model] --all-models: {len(model_config)} models × {len(available_fixes)} tasks = {len(task_model_pairs)} jobs")
        elif args.model:
            task_model_pairs = [(fix_id, args.model) for fix_id in available_fixes]
            print(f"[model] Forced model '{args.model}' for all {len(task_model_pairs)} tasks")

        # Cache loaded fixes to avoid reloading for same capsule with different models
        fix_cache: Dict[str, Any] = {}

        for capsule_id, model_key in task_model_pairs:
            # Skip if no fix for this capsule
            if capsule_id not in available_fixes:
                continue

            # Get agent_args for this model (returns tuple of agent_args, pricing_key)
            result = get_agent_args_for_model(model_key, model_config)
            if result is None:
                print(f"[skip] Skipping {capsule_id} + {model_key} - model not in config")
                continue
            task_agent_args, pricing_key = result

            print(f"\n=== Queuing {capsule_id} + {model_key} ===")
            print(f"[model] Will use: {task_agent_args.get('model_name')} "
                  f"(reasoning_effort={task_agent_args.get('reasoning_effort', 'N/A')}, pricing_key={pricing_key})")

            if capsule_id not in capsule_map:
                print(f"[WARN] Capsule {capsule_id} not found in dataset. Skipping.")
                continue

            # Load fix (cached)
            if capsule_id not in fix_cache:
                try:
                    fix_cache[capsule_id] = load_fix_package(
                        capsule_id, fixes_root=fixes_base, benchmark=args.benchmark
                    )
                except Exception as exc:
                    print(f"[WARN] Failed to load fix for {capsule_id}: {exc}. Skipping.")
                    fix_cache[capsule_id] = None

            fix = fix_cache[capsule_id]
            if fix is None:
                continue

            input_override = fix.input_override if fix else None
            env_override = fix.env_override if fix else None
            if env_override:
                conda_env = str(env_override.get("HAL_FORCE_CONDA_ENV") or conda_env)
            payload = build_single_task_payload(
                original_tasks=capsule_map,
                capsule_id=capsule_id,
                input_override=input_override,
            )
            task_agent_args["benchmark_name"] = args.benchmark
            capsule_jobs.append((capsule_id, payload, env_override, task_agent_args, pricing_key))

        if not capsule_jobs:
            print("No runnable capsules found after filtering.")
            return

        # Handle --dry-run
        if args.dry_run:
            print(f"\n{'='*60}")
            print("DRY RUN - Would execute the following jobs:")
            print(f"{'='*60}\n")
            for capsule_id, payload, env_override, task_agent_args, pricing_key in capsule_jobs:
                print(f"  - {capsule_id}")
                print(f"      Model: {task_agent_args.get('model_name')}")
                print(f"      Pricing key: {pricing_key}")
                if env_override:
                    print(f"      Env overrides: {list(env_override.keys())}")
            print(f"\nTotal jobs: {len(capsule_jobs)}")
            print(f"Prefix: {args.prefix or '(none)'}")
            print(f"Docker: {args.docker}")
            print(f"{'='*60}\n")
            return

        # Handle --verify-fixes
        if args.verify_fixes:
            print(f"\n{'='*60}")
            print("VERIFY FIXES - Checking fix packages:")
            print(f"{'='*60}\n")
            for capsule_id, payload, env_override, task_agent_args, pricing_key in capsule_jobs:
                print(f"\n--- {capsule_id} ---")
                fix = fix_cache.get(capsule_id)
                if fix:
                    if fix.input_override:
                        print(f"  input_override: present")
                    if fix.env_override:
                        print(f"  env_override: {list(fix.env_override.keys())}")
                    if fix.agent_overlay:
                        print(f"  agent_overlay: present")
                    if fix.agent_patch:
                        print(f"  agent_patch: present")
                else:
                    print(f"  (no fix loaded)")
            print(f"\n{'='*60}\n")
            return

        # Ensure basic CLI dependencies (click) are available
        ensure_hal_cli_dependencies(conda_env, ["conda-forge"])

        # NOTE: Per-capsule conda packages are now installed inside run_one_capsule()
        # This avoids conflicts when different capsules need different package versions (e.g., r-base=4.0 vs r-base=4.2)

        generated_traces: List[Path] = []
        max_workers = max(1, int(args.max_parallel_capsules))
        if max_workers <= 1 or len(capsule_jobs) <= 1:
            for capsule_id, payload, env_override, task_agent_args, pricing_key in capsule_jobs:
                print(f"\n=== Processing {capsule_id} (pricing_key={pricing_key}) ===")
                generated_traces.append(
                    run_one_capsule(
                        capsule_id=capsule_id,
                        fixes_base=fixes_base,
                        dataset_dir=temp_root,
                        dataset_payload=payload,
                        agent_dir=agent_dir,
                        agent_args=task_agent_args,
                        benchmark=args.benchmark,
                        prefix=args.prefix,
                        docker=args.docker,
                        vm=args.vm,
                        wandb_mode=args.wandb_mode,
                        env_override=env_override,
                        rubric_model=args.rubric_model,
                        rubric_output_dir=rubric_output_dir,
                        skip_rubrics=args.skip_rubrics,
                        keep_temp=args.keep_temp,
                        conda_env=conda_env,
                        use_isolated_env=args.isolated_env,
                        keep_isolated_env=args.keep_isolated_env,
                    )
                )
        else:
            print(f"\n=== Running {len(capsule_jobs)} capsules with max_parallel_capsules={max_workers} ===")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        run_one_capsule,
                        capsule_id=capsule_id,
                        fixes_base=fixes_base,
                        dataset_dir=temp_root,
                        dataset_payload=payload,
                        agent_dir=agent_dir,
                        agent_args=task_agent_args,
                        benchmark=args.benchmark,
                        prefix=args.prefix,
                        docker=args.docker,
                        vm=args.vm,
                        wandb_mode=args.wandb_mode,
                        env_override=env_override,
                        rubric_model=args.rubric_model,
                        rubric_output_dir=rubric_output_dir,
                        skip_rubrics=args.skip_rubrics,
                        keep_temp=args.keep_temp,
                        conda_env=conda_env,
                        use_isolated_env=args.isolated_env,
                        keep_isolated_env=args.keep_isolated_env,
                    ): (capsule_id, pricing_key)
                    for capsule_id, payload, env_override, task_agent_args, pricing_key in capsule_jobs
                }
                for future in as_completed(futures):
                    try:
                        generated_traces.append(future.result())
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise

        if generated_traces:
            # Group traces by model (extract model from filename)
            # Filename format: iter1_openai_gpt-4_1_2025-04-14_capsule-XXX_...
            # or: iter1_openai_o4-mini_2025-04-16_high_capsule-XXX_...
            traces_by_model: Dict[str, List[Path]] = {}
            for trace_path in generated_traces:
                # Extract model identifier from filename
                name = trace_path.name
                # Pattern: prefix_MODEL_capsule-XXX_benchmark_timestamp_UPLOAD.json
                # Find the part between prefix and capsule-
                parts = name.split("_capsule-")
                if len(parts) >= 2:
                    model_part = parts[0]  # e.g., "iter1_openai_gpt-4_1_2025-04-14" or "iter1_openai_o4-mini_2025-04-16_high"
                    # Remove the prefix (iter1_)
                    if model_part.startswith(args.prefix):
                        model_part = model_part[len(args.prefix):]
                    traces_by_model.setdefault(model_part, []).append(trace_path)
                else:
                    # Fallback: put in "unknown" group
                    traces_by_model.setdefault("unknown", []).append(trace_path)

            print(f"\n[merge] Grouping {len(generated_traces)} traces into {len(traces_by_model)} model groups:")
            for model_key, traces in traces_by_model.items():
                print(f"  - {model_key}: {len(traces)} traces")

            # Merge each model group separately
            merged_traces: List[Path] = []
            # Build prefix part for merged run_id (same pattern as run_one_capsule)
            prefix_value = str(args.prefix or "").strip()
            prefix_part = f"{_slugify(prefix_value, '')}_" if prefix_value else ""

            for model_key, traces in sorted(traces_by_model.items()):
                traces = sorted(traces, key=lambda p: p.name)
                merged_trace = merge_trace_files(
                    traces,
                    benchmark=args.benchmark,
                    agent_dir=agent_dir,
                    output_path=None,  # Auto-generate per model
                    merged_run_id=f"{prefix_part}{args.benchmark}_{model_key}_FIXED" if model_key != "unknown" else None,
                )
                merged_traces.append(merged_trace)
                print(f"[merge] Saved merged trace for {model_key} -> {merged_trace}")

            # Run rubrics on each merged trace
            if not args.skip_rubrics:
                for merged_trace in merged_traces:
                    evaluate_trace(merged_trace, args.rubric_model, rubric_output_dir)
                    print(f"[merge] Completed rubric evaluation for {merged_trace}")

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
