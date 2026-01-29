#!/usr/bin/env python3
"""
Apply ColBench fix packages and re-run HAL evaluations.

This script:
1. Loads model configurations from model_to_baseline_colbench.json
2. Loads fixes from fixes/colbench/<task_id>/
3. Creates a modified dataset with instruction clarifications injected
4. Runs HAL evaluation for fixed tasks using the original failing model
5. Outputs traces with a configurable prefix

Parallelism: Each (task, model) pair runs in a separate HAL instance.
Use --parallel N to run N HAL instances concurrently.

Usage:
    # List available fixes
    python scripts/run_colbench_fixes.py --list-fixes

    # Verify fixes are applied correctly (without running HAL)
    python scripts/run_colbench_fixes.py --verify-fixes

    # Dry run - see what would happen
    python scripts/run_colbench_fixes.py --task-id 1 --dry-run

    # Run fixes for specific tasks
    python scripts/run_colbench_fixes.py --task-id 1 --prefix fixed_ --docker

    # Run fixes for all failed tasks that have fixes (auto-detects model from rubric)
    python scripts/run_colbench_fixes.py --prefix iter1_ \
        --rubric-csv rubrics_output/colbench/colbench_combined.csv --docker

    # Run with parallelism (5 concurrent HAL instances)
    python scripts/run_colbench_fixes.py --all-models --prefix iter1_ --parallel 5 --docker
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
FIXES_DIR = REPO_ROOT / "fixes" / "colbench"
TRACES_DIR = REPO_ROOT / "traces"
HAL_HARNESS = REPO_ROOT / "hal-harness"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model_to_baseline_colbench.json"
TMP_DIR = REPO_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

# ColBench data paths
COLBENCH_DATA_DIR = HAL_HARNESS / "hal" / "benchmarks" / "colbench" / "data"
BACKEND_TASK_PATH = COLBENCH_DATA_DIR / "backend_test.jsonl"
FRONTEND_TASK_PATH = COLBENCH_DATA_DIR / "frontend_test.jsonl"


def log(msg: str, prefix: str = "main") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def _slugify(value: str, fallback: str) -> str:
    """Keep readable identifiers for run_ids / filenames."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or fallback


def load_model_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model_to_baseline_colbench.json."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))

    # Flat dict format - expected
    if isinstance(data, dict) and "models" not in data:
        return data

    # Legacy format with "models" array
    if "models" in data and isinstance(data["models"], list):
        result = {}
        for model_entry in data["models"]:
            model_id = model_entry.get("model_id")
            if model_id:
                result[model_id] = model_entry
        return result

    return {}


def load_rubric_task_models(csv_path: Path) -> List[Tuple[str, str]]:
    """Load rubric CSV and return list of (task_id, model_id) pairs for failed tasks."""
    task_model_pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    if not csv_path.exists():
        return task_model_pairs

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            correct = row.get("correct", "").strip()
            model_run = row.get("model_run", "").strip()

            # Only include failed tasks (correct=0)
            if task_id and correct == "0" and model_run:
                model_id = _extract_model_from_run_name(model_run)
                if model_id and (task_id, model_id) not in seen:
                    task_model_pairs.append((task_id, model_id))
                    seen.add((task_id, model_id))

    return task_model_pairs


def _extract_model_from_run_name(model_run: str) -> Optional[str]:
    """Extract a model config key from the model_run column."""
    has_high = "_high_" in model_run.lower() or "_high" in model_run.lower()
    has_low = "_low_" in model_run.lower() or "_low" in model_run.lower()

    patterns = [
        (r"gpt[\-_]?4[\-_]?1[\-_]?2025|gpt4120250414|gpt41", "gpt-4.1-2025-04-14"),
        (r"o3[\-_]?2025[\-_]?04[\-_]?16|o320250416|o3low|o4low", "o3-2025-04-16"),
        (r"o4[\-_]?mini[\-_]?2025|o4mini20250416|o4minihigh", "o4-mini-2025-04-16-high"),
    ]

    for pattern, config_key in patterns:
        if re.search(pattern, model_run, re.IGNORECASE):
            return config_key

    return None


def get_agent_args_for_model(
    model_key: str,
    model_config: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Get the agent_args for a specific model."""
    if not model_key:
        return None

    pricing_key = model_key
    model_entry = model_config.get(model_key)

    # Try matching by model_id
    if not model_entry:
        for key, entry in model_config.items():
            if entry.get("model_id") == model_key:
                model_entry = entry
                pricing_key = key
                break

    if not model_entry:
        return None

    model_id = model_entry.get("model_id")
    if not model_id:
        return None

    agent_args: Dict[str, Any] = {
        "model_name": model_id,
    }
    if "reasoning_effort" in model_entry:
        agent_args["reasoning_effort"] = model_entry["reasoning_effort"]
    if "budget" in model_entry:
        agent_args["budget"] = model_entry["budget"]

    return agent_args, pricing_key


def load_fix_package(task_id: str) -> Dict[str, Any]:
    """Load all fix files for a task."""
    fix_dir = FIXES_DIR / task_id
    if not fix_dir.exists():
        return {}

    fix = {"task_id": task_id, "fix_dir": fix_dir}

    # Load different override types
    for name in ["instruction_override", "simulated_user_override", "env_override", "evaluation_override"]:
        path = fix_dir / f"{name}.json"
        if path.exists():
            fix[name] = json.loads(path.read_text())

    # Load README
    readme = fix_dir / "README.md"
    if readme.exists():
        fix["readme"] = readme.read_text()

    return fix


def list_available_fixes(include_readme_only: bool = False) -> List[str]:
    """List all task IDs that have actual fix files."""
    if not FIXES_DIR.exists():
        return []

    task_ids = []
    for item in FIXES_DIR.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            actual_fix_files = [
                "instruction_override.json", "simulated_user_override.json",
                "env_override.json", "evaluation_override.json",
            ]
            has_actual_fix = any((item / f).exists() for f in actual_fix_files)

            if has_actual_fix:
                task_ids.append(item.name)
            elif include_readme_only and (item / "README.md").exists():
                task_ids.append(item.name)

    return sorted(task_ids, key=lambda x: int(x) if x.isdigit() else float('inf'))


def load_colbench_dataset(benchmark: str) -> List[Dict[str, Any]]:
    """Load the ColBench dataset from JSONL file."""
    if "frontend" in benchmark:
        task_path = FRONTEND_TASK_PATH
    else:
        task_path = BACKEND_TASK_PATH

    if not task_path.exists():
        log(f"Task file not found: {task_path}", "data")
        return []

    tasks = []
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    return tasks


def apply_fix_to_task(task: Dict[str, Any], task_index: int, fix: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Apply a fix to a task, modifying task fields.

    Returns:
        Tuple of (modified_task, list of changes made)
    """
    modified = json.loads(json.dumps(task))  # Deep copy
    changes_made = []

    # =============================================================
    # 1. Apply instruction_override - Add clarifications to problem_description
    # =============================================================
    if fix.get("instruction_override"):
        instr = fix["instruction_override"]
        clarifications = instr.get("clarifications", [])

        if clarifications:
            clarification_text = "\n\n## IMPORTANT CLARIFICATIONS\n" + "\n".join(f"- {c}" for c in clarifications)
            modified["problem_description"] = modified.get("problem_description", "") + clarification_text
            changes_made.append(f"Added {len(clarifications)} clarification(s) to problem_description")

    # =============================================================
    # 2. Apply simulated_user_override - Additional guidance
    # =============================================================
    if fix.get("simulated_user_override"):
        user_fix = fix["simulated_user_override"]
        clarifications = user_fix.get("clarifications", [])

        if clarifications:
            clarification_text = "\n\n## ADDITIONAL NOTES\n" + "\n".join(f"- {c}" for c in clarifications)
            modified["problem_description"] = modified.get("problem_description", "") + clarification_text
            changes_made.append(f"Added {len(clarifications)} simulated_user clarification(s)")

    # =============================================================
    # 3. Apply env_override - Store for harness to read
    # =============================================================
    if fix.get("env_override"):
        modified["_env_override"] = fix["env_override"]
        changes_made.append(f"Environment: Added overrides {list(fix['env_override'].keys())}")

    return modified, changes_made


def create_filtered_dataset(
    tasks: List[Dict[str, Any]],
    fixes: Dict[str, Dict[str, Any]],
    task_ids: List[str],
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Create a filtered dataset with only the specified tasks, with fixes applied."""
    filtered = []
    all_changes: Dict[str, List[str]] = {}

    for i, task in enumerate(tasks):
        task_id = str(i)  # ColBench uses index as task_id
        if task_id in task_ids:
            if task_id in fixes:
                modified, changes = apply_fix_to_task(task, i, fixes[task_id])
                all_changes[task_id] = changes

                if verbose and changes:
                    log(f"Applied {len(changes)} fix(es) to task {task_id}:", "fix")
                    for change in changes:
                        log(f"  - {change}", "fix")
                elif verbose:
                    log(f"No changes applied to task {task_id} (fix had no applicable overrides)", "fix")
            else:
                modified = task
            filtered.append(modified)

    return filtered, all_changes


def run_hal_eval(
    agent_dir: Path,
    agent_args: Dict[str, Any],
    dataset_path: Path,
    output_prefix: str,
    benchmark: str = "colbench_backend_programming",
    docker: bool = False,
    conda_env: Optional[str] = None,
    task_id: Optional[str] = None,
    pricing_key: Optional[str] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Run HAL evaluation with a custom dataset.

    Args:
        task_id: Single task ID being run (for run_id naming)
        pricing_key: The config key used for pricing lookup (may differ from model_id).
                     If None, falls back to model_name from agent_args.
    """
    model_name = str(agent_args.get("model_name", "model"))
    model_slug = _slugify(model_name.replace("/", "_"), "model")
    effort = str(agent_args.get("reasoning_effort") or "").strip().lower()
    effort_part = f"_{_slugify(effort, '')}" if effort else ""
    task_part = f"_{task_id}" if task_id else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{output_prefix}{model_slug}{effort_part}{task_part}_{benchmark}_{timestamp}"

    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", output_prefix,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]

    # Note: task filtering is done via the dataset file (COLBENCH_*_DATASET_PATH)
    # The dataset only contains the tasks we want to run

    if docker:
        cmd.append("--docker")

    for key, value in agent_args.items():
        if isinstance(value, (dict, list)):
            cmd += ["-A", f"{key}={json.dumps(value)}"]
        else:
            cmd += ["-A", f"{key}={value}"]

    # Set environment (like scienceagentbench does)
    env = os.environ.copy()

    # Set dataset path for colbench
    if "frontend" in benchmark:
        env["COLBENCH_FRONTEND_DATASET_PATH"] = str(dataset_path)
    else:
        env["COLBENCH_BACKEND_DATASET_PATH"] = str(dataset_path)

    # Add hal-harness to PYTHONPATH
    extra_path = str(HAL_HARNESS)
    env["PYTHONPATH"] = f"{extra_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault("WANDB_SILENT", "true")

    # Enable direct Azure TRAPI access (bypass LiteLLM proxy)
    # USE_DIRECT_AZURE tells HAL Docker runner to pre-fetch Azure AD token
    # USE_TRAPI tells the ColBench agent to use Azure TRAPI
    env["USE_DIRECT_AZURE"] = "true"
    env["USE_TRAPI"] = "true"

    # Use pricing_key for cost tracking (config key), model_name is for API calls (model_id)
    effective_pricing_key = pricing_key or str(agent_args.get("model_name", ""))
    env["HAL_PRICING_MODEL_NAME"] = effective_pricing_key

    # Set weave project name based on prefix
    weave_project = f"{output_prefix.rstrip('_')}_{benchmark}"
    env["HAL_WEAVE_PROJECT"] = weave_project

    log(f"Running HAL eval with run_id: {run_id}", "hal")
    log(f"Model (API): {agent_args.get('model_name')} | Pricing key: {effective_pricing_key}", "hal")
    log(f"Dataset: {dataset_path}", "hal")
    log(f"Command: {' '.join(cmd)}", "hal")

    # Results directory (at REPO_ROOT, not HAL_HARNESS)
    results_dir = REPO_ROOT / "results" / benchmark / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(os.environ.get("HAL_TASK_TIMEOUT_SECONDS", "600"))
    log(f"Task timeout: {timeout_seconds}s (HAL_TASK_TIMEOUT_SECONDS)", "hal")
    try:
        # Run from REPO_ROOT (like scienceagentbench) - let output stream to console
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, timeout=timeout_seconds)

        # Find output trace in REPO_ROOT/results
        trace_path = results_dir / f"{run_id}_UPLOAD.json"

        # Also check HAL_HARNESS results as fallback
        if not trace_path.exists():
            hal_results_dir = HAL_HARNESS / "results" / benchmark / run_id
            alt_trace = hal_results_dir / f"{run_id}_UPLOAD.json"
            if alt_trace.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(alt_trace, trace_path)
                # Also copy logs if they exist
                for log_file in hal_results_dir.glob("*.log"):
                    shutil.copy(log_file, results_dir / log_file.name)

        if result.returncode == 0:
            if trace_path.exists():
                # Copy trace to our traces directory with prefix
                dest = TRACES_DIR / f"{output_prefix}_{trace_path.name}"
                TRACES_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(trace_path, dest)
                log(f"Trace saved to: {dest}", "hal")
                log(f"Results in: {results_dir}", "hal")
                return True, "Success", dest
            return True, "Success (no trace found)", None
        else:
            log(f"HAL eval failed with exit code {result.returncode}", "hal")
            # Check for verbose log to find error
            verbose_log = results_dir / f"{run_id}_verbose.log"
            if verbose_log.exists():
                log(f"Check verbose log: {verbose_log}", "hal")
            return False, f"Exit code {result.returncode}", None

    except subprocess.TimeoutExpired:
        return False, "Timeout", None
    except Exception as e:
        return False, str(e), None


def main():
    parser = argparse.ArgumentParser(
        description="Apply ColBench fixes and re-run HAL evaluations."
    )

    parser.add_argument("--agent-dir", default=str(HAL_HARNESS / "agents" / "colbench_example_agent"),
                        help="Path to the agent directory.")
    parser.add_argument("--benchmark", default="colbench_backend_programming",
                        choices=["colbench_backend_programming", "colbench_frontend_design"],
                        help="Benchmark variant (default: colbench_backend_programming).")
    parser.add_argument("--output-prefix", default="fixed",
                        help="Prefix for output traces (default: fixed).")
    parser.add_argument("--prefix", help="Prefix for run IDs. Takes precedence over --output-prefix.")
    parser.add_argument("--task-id", dest="task_ids", action="append",
                        help="Task ID(s) to process. Can be repeated.")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to process.")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be done.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel HAL instances to run (default: 1). Each (task, model) pair runs in a separate HAL instance.")
    parser.add_argument("--list-fixes", action="store_true", help="List available fixes and exit.")
    parser.add_argument("--show-fix", help="Show details of a specific fix and exit.")
    parser.add_argument("--docker", action="store_true",
                        help="Run HAL evaluation with --docker flag. NOTE: ColBench frontend tasks need Firefox/Selenium, which may not work in Docker.")
    parser.add_argument("--conda-env", default=None,
                        help="Conda environment name to use for local execution (recommended for frontend tasks that need Firefox).")
    parser.add_argument("--rubric-csv", help="Path to rubric CSV to determine which model failed.")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG),
                        help="Path to model configuration JSON.")
    parser.add_argument("--model", help="Force a specific model for all tasks.")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all models from config for each task with fixes.")
    parser.add_argument("--verify-fixes", action="store_true",
                        help="Verify fixes are applied correctly without running HAL eval.")

    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = REPO_ROOT / agent_dir

    # List fixes mode
    if args.list_fixes:
        with_fixes = list_available_fixes(include_readme_only=False)
        all_analyzed = list_available_fixes(include_readme_only=True)
        readme_only = set(all_analyzed) - set(with_fixes)

        print(f"\n{'='*60}")
        print(f"FIXES AVAILABLE in fixes/colbench/")
        print(f"{'='*60}\n")

        print(f"Tasks with ACTUAL FIXES ({len(with_fixes)}):")
        for task_id in with_fixes:
            fix = load_fix_package(task_id)
            types = []
            if fix.get("instruction_override"): types.append("instr")
            if fix.get("simulated_user_override"): types.append("user")
            if fix.get("env_override"): types.append("env")
            if fix.get("evaluation_override"): types.append("eval")
            print(f"  {task_id}: [{', '.join(types)}]")

        if readme_only:
            print(f"\nTasks analyzed - NO FIX NEEDED ({len(readme_only)}):")
            for task_id in sorted(readme_only, key=lambda x: int(x) if x.isdigit() else float('inf')):
                print(f"  {task_id}: [readme only]")

        print(f"\n{'='*60}")
        print(f"Summary: {len(with_fixes)} fixes to apply, {len(readme_only)} analyzed (no fix needed)")
        print(f"{'='*60}\n")
        return

    # Show specific fix
    if args.show_fix:
        fix = load_fix_package(args.show_fix)
        if not fix:
            print(f"No fix found for task {args.show_fix}")
            return
        print(f"\n=== Fix for task {args.show_fix} ===\n")
        for key, value in fix.items():
            if key in ["fix_dir"]:
                continue
            if key == "readme":
                print(f"README:\n{value[:500]}...")
            elif isinstance(value, dict):
                print(f"{key}: {json.dumps(value, indent=2)}")
            else:
                print(f"{key}: {value[:200] if isinstance(value, str) else value}")
        return

    # Verify fixes mode
    if args.verify_fixes:
        print(f"\n{'='*60}")
        print("VERIFY FIXES MODE")
        print(f"{'='*60}\n")

        log(f"Loading ColBench dataset for {args.benchmark}...", "verify")
        dataset = load_colbench_dataset(args.benchmark)
        if not dataset:
            log("Failed to load dataset", "verify")
            return
        log(f"Loaded {len(dataset)} tasks", "verify")

        with_fixes = list_available_fixes(include_readme_only=False)
        if args.task_ids:
            with_fixes = [t for t in with_fixes if t in args.task_ids]

        log(f"Verifying {len(with_fixes)} fixes...\n", "verify")

        all_success = True
        for task_id in with_fixes:
            fix = load_fix_package(task_id)
            if not fix:
                continue

            try:
                task_idx = int(task_id)
                if task_idx >= len(dataset):
                    print(f"  Task {task_id}: OUT OF RANGE (only {len(dataset)} tasks)")
                    all_success = False
                    continue
                original_task = dataset[task_idx]
            except ValueError:
                print(f"  Task {task_id}: INVALID ID (must be numeric)")
                all_success = False
                continue

            modified_task, changes = apply_fix_to_task(original_task, task_idx, fix)

            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print(f"{'='*60}")

            if not changes:
                print(f"  NO CHANGES APPLIED (fix may not match dataset structure)")
                all_success = False
            else:
                print(f"  {len(changes)} change(s) applied:")
                for change in changes:
                    print(f"    - {change}")

                # Show before/after for problem_description
                orig_desc = original_task.get("problem_description", "")[:200]
                mod_desc = modified_task.get("problem_description", "")
                if orig_desc != mod_desc[:200]:
                    print(f"\n  problem_description (truncated):")
                    print(f"    BEFORE: {orig_desc}...")
                    # Show the added part
                    added = mod_desc[len(original_task.get('problem_description', '')):]
                    if added:
                        print(f"    ADDED: {added[:300]}...")

        print(f"\n{'='*60}")
        if all_success:
            print("All fixes verified successfully!")
        else:
            print("Some fixes had issues - review output above")
        print(f"{'='*60}\n")
        return

    # Determine effective prefix
    prefix = args.prefix or args.output_prefix or "fixed"
    if not prefix.endswith("_"):
        prefix = prefix + "_"

    # Load model config
    model_config_path = Path(args.model_config)
    if not model_config_path.is_absolute():
        model_config_path = REPO_ROOT / model_config_path
    model_config = load_model_config(model_config_path)
    if model_config:
        log(f"Loaded {len(model_config)} model configs from {model_config_path.name}", "model")
    else:
        log(f"WARNING: No model config found at {model_config_path}", "model")

    # Build list of available fixes
    available_fixes = set(list_available_fixes())
    if args.task_ids:
        requested = set(args.task_ids)
        available_fixes = available_fixes & requested
        missing = requested - available_fixes
        if missing:
            log(f"WARN: Requested task(s) not found: {', '.join(sorted(missing))}", "main")

    if not available_fixes:
        log("No fix directories found.", "main")
        return

    log(f"Found {len(available_fixes)} available fixes", "main")

    # Build (task_id, model_id) pairs to run
    task_model_pairs: List[Tuple[str, str]] = []

    if args.all_models:
        if not model_config:
            log("ERROR: --all-models requires model config file", "model")
            return
        for task_id in sorted(available_fixes, key=lambda x: int(x) if x.isdigit() else float('inf')):
            for model_key in model_config.keys():
                task_model_pairs.append((task_id, model_key))
        log(f"Running all {len(model_config)} models for {len(available_fixes)} tasks = {len(task_model_pairs)} total jobs", "model")
    elif args.model:
        for task_id in sorted(available_fixes, key=lambda x: int(x) if x.isdigit() else float('inf')):
            task_model_pairs.append((task_id, args.model))
        log(f"Forced model '{args.model}' for all {len(task_model_pairs)} tasks", "model")
    elif args.rubric_csv:
        rubric_csv_path = Path(args.rubric_csv)
        if not rubric_csv_path.is_absolute():
            rubric_csv_path = REPO_ROOT / rubric_csv_path
        all_failed_pairs = load_rubric_task_models(rubric_csv_path)
        log(f"Loaded {len(all_failed_pairs)} failed (task, model) pairs from {rubric_csv_path.name}", "model")

        for task_id, model_id in all_failed_pairs:
            if task_id in available_fixes:
                task_model_pairs.append((task_id, model_id))
        log(f"Filtered to {len(task_model_pairs)} pairs with available fixes", "model")
    elif model_config:
        # Default: run all models from config for each task with fixes
        for task_id in sorted(available_fixes, key=lambda x: int(x) if x.isdigit() else float('inf')):
            for model_key in model_config.keys():
                task_model_pairs.append((task_id, model_key))
        log(f"Running all {len(model_config)} models for {len(available_fixes)} tasks = {len(task_model_pairs)} total jobs", "model")
        for model_key in model_config.keys():
            log(f"  - {model_key}", "model")
    else:
        log("No model config found, cannot proceed", "model")
        return

    if args.max_tasks:
        task_model_pairs = task_model_pairs[:args.max_tasks]

    if not task_model_pairs:
        log("No tasks to process after filtering.", "main")
        return

    # Load fixes
    fixes: Dict[str, Dict[str, Any]] = {}
    for task_id in set(t[0] for t in task_model_pairs):
        fix = load_fix_package(task_id)
        if fix:
            fixes[task_id] = fix

    if not fixes:
        log("No valid fixes found for specified tasks.", "main")
        return

    log(f"Loaded {len(fixes)} fixes", "main")

    # Dry run - show what would be done
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Would apply the following fixes:")
        print("="*60)

        # Group by task
        tasks_to_models: Dict[str, List[str]] = {}
        for task_id, model_id in task_model_pairs:
            tasks_to_models.setdefault(task_id, []).append(model_id)

        for task_id in sorted(tasks_to_models.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            models = tasks_to_models[task_id]
            fix = fixes.get(task_id, {})
            print(f"\n--- Task {task_id} ---")
            print(f"  Models: {models}")
            if fix.get("instruction_override"):
                instr = fix["instruction_override"]
                if instr.get("clarifications"):
                    print("  Clarifications:")
                    for c in instr["clarifications"]:
                        print(f"    - {c[:80]}...")
            if fix.get("simulated_user_override"):
                user = fix["simulated_user_override"]
                if user.get("clarifications"):
                    print(f"  Simulated user clarifications: {len(user['clarifications'])} items")
            if fix.get("env_override"):
                print(f"  Environment: {list(fix['env_override'].keys())}")

        print(f"\n{'='*60}")
        print(f"Would run HAL eval with prefix: {prefix}")
        print(f"Docker: {args.docker}")
        print(f"Total jobs: {len(task_model_pairs)}")
        print(f"Parallel HAL instances: {args.parallel}")
        return

    # Load ColBench dataset
    log(f"Loading ColBench dataset for {args.benchmark}...", "data")
    dataset = load_colbench_dataset(args.benchmark)
    if not dataset:
        log("Failed to load dataset", "main")
        return
    log(f"Loaded {len(dataset)} tasks from dataset", "data")

    # Load fallback agent args (for legacy mode)
    fallback_agent_args: Dict[str, Any] = {"model_name": "gpt-4o"}

    # Define job runner function for parallel execution
    def run_single_job(job_info: Tuple[int, str, str]) -> Tuple[str, str, bool, Optional[Path], str]:
        """Run a single (task, model) job. Returns (task_id, model_id, success, trace_path, message)."""
        job_idx, task_id, model_id = job_info

        # Get agent args for this model
        # Returns (agent_args, pricing_key) where pricing_key is config key for cost tracking
        pricing_key = None
        if model_id and model_config:
            result = get_agent_args_for_model(model_id, model_config)
            if result is None:
                return (task_id, model_id, False, None, "model not in config")
            task_agent_args, pricing_key = result
        else:
            task_agent_args = fallback_agent_args.copy()

        # Create filtered dataset with just this task
        fix = fixes.get(task_id)
        if not fix:
            return (task_id, model_id, False, None, "no fix")

        filtered, changes_applied = create_filtered_dataset(dataset, {task_id: fix}, [task_id], verbose=False)
        if not filtered:
            return (task_id, model_id, False, None, "task not in dataset")

        # Write temporary dataset file (JSONL format for ColBench)
        temp_dataset = TMP_DIR / f"colbench_{task_id}_{model_id.replace('/', '_')}_{time.time()}.jsonl"
        with temp_dataset.open("w", encoding="utf-8") as f:
            for task in filtered:
                # Remove internal override fields before saving
                clean_task = {k: v for k, v in task.items() if not k.startswith("_")}
                f.write(json.dumps(clean_task) + "\n")

        try:
            success, message, trace_path = run_hal_eval(
                agent_dir=agent_dir,
                agent_args=task_agent_args,
                dataset_path=temp_dataset,
                output_prefix=prefix,
                benchmark=args.benchmark,
                docker=args.docker,
                task_id=task_id,
                pricing_key=pricing_key,
            )
            return (task_id, model_id, success and trace_path is not None, trace_path, message)
        finally:
            temp_dataset.unlink(missing_ok=True)

    # Run jobs
    generated_traces: List[Path] = []
    failed_runs: List[Tuple[str, str, str]] = []
    total_jobs = len(task_model_pairs)

    # Prepare job list with indices
    jobs = [(i, task_id, model_id) for i, (task_id, model_id) in enumerate(task_model_pairs)]

    if args.parallel > 1:
        # Parallel execution - run separate HAL instances
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        log(f"Running {total_jobs} jobs with {args.parallel} parallel HAL instances", "main")
        completed = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_job = {executor.submit(run_single_job, job): job for job in jobs}

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                job_idx, task_id, model_id = job

                try:
                    task_id, model_id, success, trace_path, message = future.result()
                    with lock:
                        completed += 1
                        if success and trace_path:
                            generated_traces.append(trace_path)
                            log(f"[{completed}/{total_jobs}] SUCCESS: task={task_id} model={model_id}", "main")
                        else:
                            failed_runs.append((task_id, model_id, message))
                            log(f"[{completed}/{total_jobs}] FAILED: task={task_id} model={model_id} - {message}", "main")
                except Exception as e:
                    with lock:
                        completed += 1
                        failed_runs.append((task_id, model_id, str(e)))
                        log(f"[{completed}/{total_jobs}] ERROR: task={task_id} model={model_id} - {e}", "main")
    else:
        # Sequential execution
        for i, (task_id, model_id) in enumerate(task_model_pairs):
            log(f"\n=== [{i+1}/{total_jobs}] task={task_id}, model={model_id or 'default'} ===", "main")

            task_id, model_id, success, trace_path, message = run_single_job((i, task_id, model_id))

            if success and trace_path:
                generated_traces.append(trace_path)
                log(f"SUCCESS: {trace_path}", "main")
            else:
                failed_runs.append((task_id, model_id, message))
                log(f"FAILED: {message}", "main")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Total jobs: {len(task_model_pairs)}")
    print(f"Successful: {len(generated_traces)}")
    print(f"Failed: {len(failed_runs)}")

    if generated_traces:
        print(f"\nGenerated traces:")
        for trace in generated_traces:
            print(f"  - {trace}")

    if failed_runs:
        print(f"\nFailed runs:")
        for task_id, model_id, reason in failed_runs:
            print(f"  - {task_id} ({model_id}): {reason}")


if __name__ == "__main__":
    main()
