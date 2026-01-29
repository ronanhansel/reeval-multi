#!/usr/bin/env python3
"""
Apply USACO fix packages and re-run HAL evaluations.

This script:
1. Loads fixes from fixes/usaco/<task_id>/
2. Creates a modified dataset with problem statement clarifications injected
3. Runs HAL evaluation for fixed tasks only (filtered by model failures)
4. Outputs traces with a configurable prefix

Usage:
    # List available fixes
    python scripts/run_usaco_fixes.py --list-fixes

    # Dry run - see what would happen
    python scripts/run_usaco_fixes.py \
        --task-id usaco_standin-bronze-2023-jan \
        --dry-run

    # Run fixes for all failed tasks that have fixes (recommended)
    python scripts/run_usaco_fixes.py \
        --prefix usaco_lime_ \
        --rubric-csv rubrics_output/usaco/usaco_combined.csv \
        --docker

    # Force a specific model for all tasks
    python scripts/run_usaco_fixes.py \
        --prefix usaco_lime_ \
        --model openai/gpt-4.1-2025-04-14 \
        --docker

    # Run with specific agent function (zeroshot, retrieval, reflexion)
    python scripts/run_usaco_fixes.py \
        --prefix usaco_lime_ \
        --agent-function main.run_usaco_zeroshot \
        --docker
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
FIXES_DIR = REPO_ROOT / "fixes"
TRACES_DIR = REPO_ROOT / "traces"
HAL_HARNESS = REPO_ROOT / "hal-harness"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model_to_baseline_usaco.json"
USACO_DATASET_PATH = HAL_HARNESS / "hal" / "benchmarks" / "USACO" / "data" / "datasets" / "usaco_subset307_dict.json"
TMP_DIR = REPO_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)


def log(msg: str, prefix: str = "main") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def _slugify(value: str, fallback: str) -> str:
    """Keep readable identifiers for run_ids / filenames."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or fallback


def load_model_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model_to_baseline_usaco.json and return a dict mapping model keys to config.

    Format:
    {
      "openai/gpt-4.1-2025-04-14": {
        "model_id": "openai/gpt-4.1_2025-04-14",
        "short_name": "gpt-4.1-04-14",
        "baseline_trace": "usaco_..._UPLOAD.json",
        "reasoning_effort": "high",  // optional
        "max_steps": 5
      },
      ...
    }
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))

    # Flat dict format - this is the expected format
    if isinstance(data, dict) and "models" not in data:
        return data

    # Legacy format with "models" array - convert to flat dict
    if "models" in data and isinstance(data["models"], list):
        result = {}
        for model_entry in data["models"]:
            model_id = model_entry.get("model_id")
            if model_id:
                result[model_id] = model_entry
        return result

    return {}


def load_rubric_task_models(csv_path: Path) -> List[Tuple[str, str]]:
    """Load rubric CSV and return list of (task_id, model_id) pairs for failed tasks.

    CSV format: task_id,criteria,grade,correct,explanation,model_run
    We extract failed tasks (correct=0) and parse model_id from model_run column.
    """
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
    """Extract a model config key from the model_run column.

    Returns the key that matches model_to_baseline_usaco.json.
    """
    # Check for reasoning effort suffix
    has_high = "_high_" in model_run.lower() or "_high" in model_run.lower()
    has_low = "_low_" in model_run.lower() or "_low" in model_run.lower()

    # Known model patterns and their config keys
    patterns = [
        # GPT models
        (r"gpt[\-_]?5|gpt5", "openai/gpt-5-0125"),
        (r"gpt[\-_]?4[\-_]?1[\-_]?2025[\-_]?04[\-_]?14|gpt4120250414", "openai/gpt-4.1-2025-04-14"),
        (r"gpt4o[\-_]?2024[\-_]?11[\-_]?20", "openai/gpt-4o-2024-11-20"),
        # OpenAI reasoning models
        (r"o3[\-_]?2025[\-_]?04[\-_]?16|o320250416", "openai/o3-2025-04-16"),
        (r"o4[\-_]?mini[\-_]?2025[\-_]?04[\-_]?16|o4mini20250416", None),  # Handle separately
        # Claude models
        (r"claude[\-_]?opus[\-_]?4[\-_]?1|claudeopus41", None),
        (r"claude[\-_]?sonnet[\-_]?4[\-_]?5|claudesonnet45", None),
        (r"claude[\-_]?3[\-_]?7[\-_]?sonnet|claude37sonnet", None),
        # Other models
        (r"deepseek[\-_]?v3|DeepSeekV3|deepseekaiDeepSeekV3", "deepseek/DeepSeek-V3"),
        (r"gemini[\-_]?2[\-_]?0[\-_]?flash|gemini20flash", "google/gemini-2.0-flash"),
    ]

    for pattern, config_key in patterns:
        if re.search(pattern, model_run, re.IGNORECASE):
            if config_key is not None:
                return config_key

            # Handle models with effort suffixes
            if "o4" in pattern and "mini" in pattern:
                if has_high:
                    return "openai/o4-mini-2025-04-16-high"
                elif has_low:
                    return "openai/o4-mini-2025-04-16-low"
                else:
                    return "openai/o4-mini-2025-04-16-low"  # Default to low

            if "opus" in pattern:
                if has_high:
                    return "anthropic/claude-opus-4-1-20250514-high"
                else:
                    return "anthropic/claude-opus-4-1-20250514"

            if "sonnet" in pattern and "4" in pattern and "5" in pattern:
                if has_high:
                    return "anthropic/claude-sonnet-4-5-20250929-high"
                else:
                    return "anthropic/claude-sonnet-4-5-20250929"

            if "3" in pattern and "7" in pattern and "sonnet" in pattern:
                if has_high:
                    return "anthropic/claude-3-7-sonnet-20250219-high"
                else:
                    return "anthropic/claude-3-7-sonnet-20250219"

    return None


def get_agent_args_for_model(
    model_key: str,
    model_config: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Get the agent_args for a specific model.

    Args:
        model_key: The model name/key to look up (used for pricing)
        model_config: The model config dict

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
    if "temperature" in model_entry:
        agent_args["temperature"] = model_entry["temperature"]

    return agent_args, pricing_key


def load_fix_package(task_id: str, fixes_root: Path) -> Dict[str, Any]:
    """Load all fix files for a task."""
    fix_dir = fixes_root / task_id
    if not fix_dir.exists():
        return {}

    fix = {"task_id": task_id, "fix_dir": fix_dir}

    # Load different override types
    for name in ["instruction_override", "evaluation_override", "input_override", "test_case_override"]:
        path = fix_dir / f"{name}.json"
        if path.exists():
            fix[name] = json.loads(path.read_text())

    # Load problem_statement.txt if exists
    ps_path = fix_dir / "problem_statement.txt"
    if ps_path.exists():
        fix["problem_statement"] = ps_path.read_text()

    # Load README
    readme = fix_dir / "README.md"
    if readme.exists():
        fix["readme"] = readme.read_text()

    return fix


def list_available_fixes(fixes_root: Path, include_readme_only: bool = False) -> List[str]:
    """List all task IDs that have actual fix files (not just READMEs).

    Args:
        fixes_root: Directory containing fix packages
        include_readme_only: If True, include tasks with only README (no actual fixes)
    """
    if not fixes_root.exists():
        return []

    task_ids = []
    for item in fixes_root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Actual fix files that modify the benchmark
            actual_fix_files = [
                "instruction_override.json", "evaluation_override.json",
                "input_override.json", "test_case_override.json",
                "problem_statement.txt",
            ]
            has_actual_fix = any((item / f).exists() for f in actual_fix_files)

            if has_actual_fix:
                task_ids.append(item.name)
            elif include_readme_only and (item / "README.md").exists():
                task_ids.append(item.name)

    return sorted(task_ids)


def load_usaco_dataset(dataset_path: Path = None) -> Dict[str, Dict[str, Any]]:
    """Load the USACO dataset from JSON file.

    Returns:
        Dictionary mapping task_id to task data
    """
    if dataset_path is None:
        dataset_path = USACO_DATASET_PATH

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to load USACO dataset: {e}", "data")
        return {}


def apply_fix_to_task(task: Dict[str, Any], fix: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Apply a fix to a USACO task.

    USACO tasks are simpler than SciCode - they're flat dictionaries with
    problem description, test cases, etc.

    Returns:
        Tuple of (modified_task, list of changes made)
    """
    modified = json.loads(json.dumps(task))  # Deep copy
    changes_made = []
    task_id = fix.get("task_id", "")

    # =============================================================
    # 1. Apply instruction_override - Fix problem description
    # =============================================================
    if fix.get("instruction_override"):
        instr = fix["instruction_override"]

        # Add clarifications to problem description
        if instr.get("clarifications"):
            clarifications = instr["clarifications"]
            clarification_text = "\n\n## IMPORTANT CLARIFICATIONS\n" + "\n".join(f"- {c}" for c in clarifications)

            # USACO tasks have 'description' field
            if "description" in modified:
                modified["description"] += clarification_text
                changes_made.append(f"Added {len(clarifications)} clarifications to description")
            elif "problem_description" in modified:
                modified["problem_description"] += clarification_text
                changes_made.append(f"Added {len(clarifications)} clarifications to problem_description")

        # Replace entire problem description if provided
        if instr.get("corrected_description"):
            if "description" in modified:
                modified["description"] = instr["corrected_description"]
                changes_made.append("Replaced problem description")
            elif "problem_description" in modified:
                modified["problem_description"] = instr["corrected_description"]
                changes_made.append("Replaced problem description")

        # Fix constraints
        if instr.get("corrected_constraints"):
            modified["constraints"] = instr["corrected_constraints"]
            changes_made.append("Fixed constraints")

        # Fix input/output format
        if instr.get("corrected_input_format"):
            modified["input_format"] = instr["corrected_input_format"]
            changes_made.append("Fixed input format")
        if instr.get("corrected_output_format"):
            modified["output_format"] = instr["corrected_output_format"]
            changes_made.append("Fixed output format")

    # =============================================================
    # 2. Apply input_override - Additional clarifications
    # =============================================================
    if fix.get("input_override"):
        inp = fix["input_override"]
        clarifications = inp.get("clarifications", [])

        if clarifications:
            clarification_text = "\n\n## ADDITIONAL NOTES\n" + "\n".join(f"- {c}" for c in clarifications)

            if "description" in modified:
                modified["description"] += clarification_text
            elif "problem_description" in modified:
                modified["problem_description"] += clarification_text
            changes_made.append(f"Added {len(clarifications)} input clarifications")

    # =============================================================
    # 3. Apply evaluation_override - Fix test cases or judge behavior
    # =============================================================
    if fix.get("evaluation_override"):
        eval_fix = fix["evaluation_override"]

        # Add notes about evaluation
        if eval_fix.get("notes"):
            notes_text = f"\n\n## EVALUATION NOTES\n{eval_fix['notes']}"
            if "description" in modified:
                modified["description"] += notes_text
            changes_made.append("Added evaluation notes")

        # Store time limit override if specified
        if eval_fix.get("time_limit_override"):
            modified["_time_limit_override"] = eval_fix["time_limit_override"]
            changes_made.append(f"Time limit override: {eval_fix['time_limit_override']}")

        # Store precision override if specified
        if eval_fix.get("precision_override"):
            modified["_precision_override"] = eval_fix["precision_override"]
            changes_made.append(f"Precision override: {eval_fix['precision_override']}")

    # =============================================================
    # 4. Apply test_case_override - Fix specific test cases
    # =============================================================
    if fix.get("test_case_override"):
        tc_fix = fix["test_case_override"]

        # Fix specific test case inputs/outputs
        if tc_fix.get("fixes"):
            for tc_idx, tc_changes in tc_fix["fixes"].items():
                if "test_cases" in modified:
                    try:
                        idx = int(tc_idx)
                        if 0 <= idx < len(modified["test_cases"]):
                            if tc_changes.get("expected_output"):
                                modified["test_cases"][idx]["output"] = tc_changes["expected_output"]
                                changes_made.append(f"Fixed test case {idx} expected output")
                            if tc_changes.get("input"):
                                modified["test_cases"][idx]["input"] = tc_changes["input"]
                                changes_made.append(f"Fixed test case {idx} input")
                    except (ValueError, IndexError):
                        pass

    # =============================================================
    # 5. Apply problem_statement.txt - Raw text addition
    # =============================================================
    if fix.get("problem_statement"):
        ps_text = f"\n\n## ADDITIONAL CONTEXT\n{fix['problem_statement']}"
        if "description" in modified:
            modified["description"] += ps_text
        elif "problem_description" in modified:
            modified["problem_description"] += ps_text
        changes_made.append("Added problem statement context")

    return modified, changes_made


def create_filtered_dataset(
    dataset: Dict[str, Dict[str, Any]],
    fixes: Dict[str, Dict[str, Any]],
    task_ids: List[str],
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Create a filtered dataset with only the specified tasks, with fixes applied.

    Returns:
        Tuple of (filtered_dataset, dict mapping task_id to list of changes made)
    """
    filtered = {}
    all_changes: Dict[str, List[str]] = {}

    for task_id in task_ids:
        if task_id in dataset:
            task = dataset[task_id]
            if task_id in fixes:
                modified, changes = apply_fix_to_task(task, fixes[task_id])
                all_changes[task_id] = changes

                if verbose and changes:
                    log(f"Applied {len(changes)} fix(es) to task {task_id}:", "fix")
                    for change in changes:
                        log(f"  - {change}", "fix")
                elif verbose:
                    log(f"No changes applied to task {task_id} (fix had no applicable overrides)", "fix")
            else:
                modified = task
            filtered[task_id] = modified

    return filtered, all_changes


def run_hal_eval(
    agent_dir: Path,
    agent_args: Dict[str, Any],
    dataset_path: Path,
    output_prefix: str,
    agent_function: str = "main.run_usaco_zeroshot",
    docker: bool = True,
    task_id: Optional[str] = None,
    pricing_key: Optional[str] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Run HAL evaluation with a custom dataset.

    Args:
        pricing_key: The config key used for pricing lookup (may differ from model_id).
    """
    # Build run_id with model info
    model_name = str(agent_args.get("model_name", "model"))
    model_slug = _slugify(model_name.replace("/", "_"), "model")
    effort = str(agent_args.get("reasoning_effort") or "").strip().lower()
    effort_part = f"_{_slugify(effort, '')}" if effort else ""
    task_part = f"_{task_id}" if task_id else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{output_prefix}{model_slug}{effort_part}{task_part}_usaco_{timestamp}"

    cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", "usaco",
        "--agent_name", output_prefix,
        "--agent_function", agent_function,
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]

    if docker:
        cmd.append("--docker")

    for key, value in agent_args.items():
        if isinstance(value, (dict, list)):
            cmd += ["-A", f"{key}={json.dumps(value)}"]
        else:
            cmd += ["-A", f"{key}={value}"]

    # Set environment
    env = os.environ.copy()
    env["USACO_DATASET_PATH"] = str(dataset_path)

    # Add hal-harness to PYTHONPATH
    extra_path = str(HAL_HARNESS)
    env["PYTHONPATH"] = f"{extra_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault("WANDB_SILENT", "true")

    effective_pricing_key = pricing_key or str(agent_args.get("model_name", ""))
    env["HAL_PRICING_MODEL_NAME"] = effective_pricing_key

    weave_project = f"{output_prefix.rstrip('_')}_usaco"
    env["HAL_WEAVE_PROJECT"] = weave_project

    log(f"Running HAL eval with run_id: {run_id}", "hal")
    log(f"Model (API): {agent_args.get('model_name')} | Pricing key: {effective_pricing_key}", "hal")
    log(f"Agent function: {agent_function}", "hal")
    log(f"Dataset: {dataset_path}", "hal")
    log(f"Command: {' '.join(cmd)}", "hal")

    # Results directory
    results_dir = REPO_ROOT / "results" / "usaco" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            timeout=7200,  # 2 hour timeout for USACO (longer due to Docker)
        )

        # Find output trace
        trace_path = results_dir / f"{run_id}_UPLOAD.json"

        # Check HAL_HARNESS results as fallback
        if not trace_path.exists():
            hal_results_dir = HAL_HARNESS / "results" / "usaco" / run_id
            alt_trace = hal_results_dir / f"{run_id}_UPLOAD.json"
            if alt_trace.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(alt_trace, trace_path)
                for log_file in hal_results_dir.glob("*.log"):
                    shutil.copy(log_file, results_dir / log_file.name)

        if result.returncode == 0:
            if trace_path.exists():
                # Copy trace to traces directory
                dest = TRACES_DIR / f"{output_prefix}_{trace_path.name}"
                TRACES_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(trace_path, dest)
                log(f"Trace saved to: {dest}", "hal")
                log(f"Results in: {results_dir}", "hal")
                return True, "Success", dest

            return True, "Success (no trace found)", None
        else:
            log(f"HAL eval failed with exit code {result.returncode}", "hal")
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
        description="Apply USACO fixes and re-run HAL evaluations."
    )

    parser.add_argument(
        "--fixes-root",
        default=str(FIXES_DIR / "usaco"),
        help="Directory containing fix packages (default: fixes/usaco).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(HAL_HARNESS / "agents" / "USACO"),
        help="Path to the USACO agent directory.",
    )
    parser.add_argument(
        "--agent-function",
        default="main.run_usaco_zeroshot",
        choices=[
            "main.run_usaco_zeroshot",
            "main.run_usaco_episodic_semantic_retrieval",
            "main.run_usaco_episodic_semantic_retrieval_reflexion",
        ],
        help="Agent function to use (default: main.run_usaco_zeroshot).",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(USACO_DATASET_PATH),
        help="Path to USACO dataset JSON file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="usaco_fixed",
        help="Prefix for output traces (default: usaco_fixed).",
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for run IDs and output files (e.g., 'usaco_lime_'). Takes precedence over --output-prefix.",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Task ID(s) to process. Can be repeated.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel HAL evaluations (default: 1).",
    )
    parser.add_argument(
        "--list-fixes",
        action="store_true",
        help="List available fixes and exit.",
    )
    parser.add_argument(
        "--show-fix",
        help="Show details of a specific fix and exit.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        default=True,
        help="Run HAL evaluation with --docker flag (default: True for USACO).",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker for HAL evaluation (not recommended for USACO).",
    )
    parser.add_argument(
        "--rubric-csv",
        help="Path to rubric CSV to determine which model failed for each task.",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model_to_baseline_usaco.json mapping model names to configs.",
    )
    parser.add_argument(
        "--model",
        help="Force a specific model for all tasks (overrides rubric CSV).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all models from model config for each task with fixes.",
    )
    parser.add_argument(
        "--verify-fixes",
        action="store_true",
        help="Verify fixes are applied correctly without running HAL eval.",
    )
    parser.add_argument(
        "--merge-traces",
        action="store_true",
        help="Merge existing trace files by model (no HAL runs).",
    )

    args = parser.parse_args()

    # Handle docker flag
    use_docker = args.docker and not args.no_docker

    # Resolve paths
    fixes_root = Path(args.fixes_root)
    if not fixes_root.is_absolute():
        fixes_root = REPO_ROOT / fixes_root

    agent_dir = Path(args.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = REPO_ROOT / agent_dir

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    # List fixes mode
    if args.list_fixes:
        with_fixes = list_available_fixes(fixes_root, include_readme_only=False)
        all_analyzed = list_available_fixes(fixes_root, include_readme_only=True)
        readme_only = set(all_analyzed) - set(with_fixes)

        print(f"\n{'='*60}")
        print(f"FIXES AVAILABLE in {fixes_root.name}/")
        print(f"{'='*60}\n")

        print(f"Tasks with ACTUAL FIXES ({len(with_fixes)}):")
        for task_id in with_fixes:
            fix = load_fix_package(task_id, fixes_root)
            types = []
            if fix.get("instruction_override"): types.append("instr")
            if fix.get("evaluation_override"): types.append("eval")
            if fix.get("input_override"): types.append("input")
            if fix.get("test_case_override"): types.append("test")
            if fix.get("problem_statement"): types.append("prompt")
            print(f"  ✓ {task_id}: [{', '.join(types)}]")

        if readme_only:
            print(f"\nTasks analyzed - NO FIX NEEDED ({len(readme_only)}):")
            for task_id in sorted(readme_only):
                print(f"  ⊘ {task_id}: [readme only - not a benchmark issue]")

        print(f"\n{'='*60}")
        print(f"Summary: {len(with_fixes)} fixes to apply, {len(readme_only)} analyzed (no fix needed)")
        print(f"{'='*60}\n")
        return

    # Show specific fix
    if args.show_fix:
        fix = load_fix_package(args.show_fix, fixes_root)
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

        log("Loading USACO dataset...", "verify")
        dataset = load_usaco_dataset(dataset_path)
        if not dataset:
            log("Failed to load dataset", "verify")
            return
        log(f"Loaded {len(dataset)} tasks", "verify")

        with_fixes = list_available_fixes(fixes_root, include_readme_only=False)
        if args.task_ids:
            with_fixes = [t for t in with_fixes if t in args.task_ids]

        log(f"Verifying {len(with_fixes)} fixes...\n", "verify")

        all_success = True
        for task_id in with_fixes:
            fix = load_fix_package(task_id, fixes_root)
            if not fix:
                continue

            original_task = dataset.get(task_id)
            if not original_task:
                print(f"  ✗ Task {task_id}: NOT FOUND in dataset")
                all_success = False
                continue

            modified_task, changes = apply_fix_to_task(original_task, fix)

            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print(f"{'='*60}")

            if not changes:
                print(f"  ⚠ NO CHANGES APPLIED (fix may not match dataset structure)")
                all_success = False
            else:
                print(f"  ✓ {len(changes)} change(s) applied:")
                for change in changes:
                    print(f"    - {change}")

                # Show before/after for description
                orig_desc = original_task.get("description", "")[:200]
                mod_desc = modified_task.get("description", "")[:200]
                if orig_desc != mod_desc:
                    print(f"\n  Description (first 200 chars):")
                    print(f"    BEFORE: {orig_desc}...")
                    print(f"    AFTER:  {mod_desc}...")

        print(f"\n{'='*60}")
        if all_success:
            print("✓ All fixes verified successfully!")
        else:
            print("⚠ Some fixes had issues - review output above")
        print(f"{'='*60}\n")
        return

    # Merge traces mode
    if args.merge_traces:
        prefix = args.prefix or args.output_prefix or "usaco_fixed"
        if not prefix.endswith("_"):
            prefix = prefix + "_"

        print(f"\n{'='*60}")
        print(f"MERGE TRACES MODE - prefix: {prefix}")
        print(f"{'='*60}\n")

        trace_pattern = f"{prefix}*_UPLOAD.json"
        all_trace_files = list(TRACES_DIR.glob(trace_pattern))
        trace_files = [f for f in all_trace_files if "_MERGED_" not in f.name]

        if not trace_files:
            log(f"No trace files found matching {trace_pattern} in {TRACES_DIR}", "merge")
            return

        log(f"Found {len(trace_files)} trace files (excluding merged)", "merge")

        # Group traces by model
        model_traces: Dict[str, List[Path]] = {}

        for trace_file in trace_files:
            name = trace_file.name
            model_key = None

            if "gpt-4_1_2025-04-14" in name or "gpt-4.1-2025-04-14" in name:
                model_key = "openai/gpt-4.1-2025-04-14"
            elif "gpt-5" in name:
                model_key = "openai/gpt-5"
            elif "o3_2025-04-16" in name or "o3-2025-04-16" in name:
                model_key = "openai/o3-2025-04-16"
            elif "o4-mini" in name and "_high_" in name:
                model_key = "openai/o4-mini-2025-04-16-high"
            elif "o4-mini" in name and "_low_" in name:
                model_key = "openai/o4-mini-2025-04-16-low"
            elif "o4-mini" in name:
                model_key = "openai/o4-mini-2025-04-16"

            if model_key:
                model_traces.setdefault(model_key, []).append(trace_file)
            else:
                log(f"Could not determine model for: {name}", "merge")

        log(f"Grouped into {len(model_traces)} models", "merge")

        for model_key, traces in sorted(model_traces.items()):
            log(f"Merging {len(traces)} traces for {model_key}", "merge")

            all_successful: Set[str] = set()
            all_failed: Set[str] = set()
            merged_config: Dict[str, Any] = {}
            total_cost = 0.0

            for trace_file in traces:
                try:
                    data = json.loads(trace_file.read_text())

                    if "results" in data:
                        results = data["results"]
                        for task_id in results.get("successful_tasks", []):
                            all_successful.add(str(task_id))
                            all_failed.discard(str(task_id))
                        for task_id in results.get("failed_tasks", []):
                            if str(task_id) not in all_successful:
                                all_failed.add(str(task_id))
                        total_cost += results.get("total_cost", 0)

                    if not merged_config and "config" in data:
                        merged_config = data["config"]

                except Exception as e:
                    log(f"Error reading {trace_file.name}: {e}", "merge")

            total_tasks = len(all_successful) + len(all_failed)
            accuracy = len(all_successful) / total_tasks if total_tasks > 0 else 0.0

            merged_trace = {
                "config": merged_config,
                "results": {
                    "accuracy": accuracy,
                    "successful_tasks": sorted(all_successful),
                    "failed_tasks": sorted(all_failed),
                    "total_cost": total_cost,
                    "latencies": {},
                },
                "raw_eval_results": {},
                "raw_logging_results": [],
                "total_usage": {},
                "total_cost": total_cost,
            }

            model_slug = model_key.replace("/", "_").replace("-", "_")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            merged_filename = f"{prefix}{model_slug}_MERGED_usaco_{timestamp}_UPLOAD.json"
            merged_path = TRACES_DIR / merged_filename

            merged_path.write_text(json.dumps(merged_trace, indent=2))
            log(f"Saved: {merged_path.name}", "merge")
            log(f"  Tasks: {len(all_successful)} successful, {len(all_failed)} failed ({total_tasks} total)", "merge")
            log(f"  Accuracy: {accuracy:.1%}", "merge")

        print(f"\n{'='*60}")
        print(f"Merged {len(model_traces)} model traces")
        print(f"{'='*60}\n")
        return

    # Determine effective prefix
    prefix = args.prefix or args.output_prefix or "usaco_fixed"
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
    available_fixes = set(list_available_fixes(fixes_root))
    if args.task_ids:
        requested = set(args.task_ids)
        available_fixes = available_fixes & requested
        missing = requested - available_fixes
        if missing:
            log(f"WARN: Requested task(s) not found under {fixes_root}: {', '.join(sorted(missing))}", "main")

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
        for task_id in sorted(available_fixes):
            for model_key in model_config.keys():
                task_model_pairs.append((task_id, model_key))
        log(f"Running all {len(model_config)} models for {len(available_fixes)} tasks = {len(task_model_pairs)} total jobs", "model")
    elif args.model:
        for task_id in sorted(available_fixes):
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
        for task_id in sorted(available_fixes):
            for model_key in model_config.keys():
                task_model_pairs.append((task_id, model_key))
        log(f"Running all {len(model_config)} models for {len(available_fixes)} tasks = {len(task_model_pairs)} total jobs", "model")
    else:
        # Default model
        log("No model config found, using default gpt-4o", "model")
        for task_id in sorted(available_fixes):
            task_model_pairs.append((task_id, "openai/gpt-4o"))

    if args.max_tasks:
        task_model_pairs = task_model_pairs[:args.max_tasks]

    if not task_model_pairs:
        log("No tasks to process after filtering.", "main")
        return

    # Load fixes
    fixes: Dict[str, Dict[str, Any]] = {}
    for task_id in set(t[0] for t in task_model_pairs):
        fix = load_fix_package(task_id, fixes_root)
        if fix:
            fixes[task_id] = fix

    if not fixes:
        log("No valid fixes found for specified tasks.", "main")
        return

    log(f"Loaded {len(fixes)} fixes", "main")

    # Dry run
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Would apply the following fixes:")
        print("="*60)

        tasks_to_models: Dict[str, List[str]] = {}
        for task_id, model_id in task_model_pairs:
            tasks_to_models.setdefault(task_id, []).append(model_id)

        for task_id in sorted(tasks_to_models.keys()):
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
            if fix.get("problem_statement"):
                print(f"  Problem statement addition: {fix['problem_statement'][:100]}...")

        print(f"\n{'='*60}")
        print(f"Would run HAL eval with prefix: {prefix}")
        print(f"Agent function: {args.agent_function}")
        print(f"Docker: {use_docker}")
        print(f"Total jobs: {len(task_model_pairs)}")
        return

    # Load USACO dataset
    log("Loading USACO dataset...", "data")
    dataset = load_usaco_dataset(dataset_path)
    if not dataset:
        log("Failed to load dataset", "main")
        return

    log(f"Loaded {len(dataset)} tasks from dataset", "data")

    # Define job runner function
    def run_single_job(job_info: Tuple[int, str, str]) -> Tuple[str, str, bool, Optional[Path], str]:
        """Run a single (task, model) job."""
        job_idx, task_id, model_id = job_info

        pricing_key = None
        if model_id and model_config:
            result = get_agent_args_for_model(model_id, model_config)
            if result is None:
                # Try using model_id directly
                task_agent_args = {"model_name": model_id}
                pricing_key = model_id
            else:
                task_agent_args, pricing_key = result
        else:
            task_agent_args = {"model_name": model_id or "openai/gpt-4o"}

        fix = fixes.get(task_id)
        if not fix:
            return (task_id, model_id, False, None, "no fix")

        filtered, changes_applied = create_filtered_dataset(dataset, {task_id: fix}, [task_id], verbose=False)
        if not filtered:
            return (task_id, model_id, False, None, "task not in dataset")

        # Write temporary dataset file
        temp_dataset = TMP_DIR / f"usaco_{task_id}_{model_id.replace('/', '_')}_{time.time()}.json"
        temp_dataset.write_text(json.dumps(filtered, indent=2))

        try:
            success, message, trace_path = run_hal_eval(
                agent_dir=agent_dir,
                agent_args=task_agent_args,
                dataset_path=temp_dataset,
                output_prefix=prefix,
                agent_function=args.agent_function,
                docker=use_docker,
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

    jobs = [(i, task_id, model_id) for i, (task_id, model_id) in enumerate(task_model_pairs)]

    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        log(f"Running {total_jobs} jobs with {args.parallel} parallel workers", "main")
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
