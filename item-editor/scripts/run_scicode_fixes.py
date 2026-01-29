#!/usr/bin/env python3
"""
Apply SciCode fix packages and re-run HAL evaluations.

This script:
1. Loads fixes from fixes/scicode/<task_id>/
2. Creates a modified dataset with instruction clarifications injected
3. Runs HAL evaluation for fixed tasks only (filtered by model failures)
4. Outputs traces with a configurable prefix

Usage:
    # List available fixes
    python scripts/run_scicode_fixes.py --list-fixes

    # Dry run - see what would happen
    python scripts/run_scicode_fixes.py \
        --task-id 11 --task-id 12 \
        --dry-run

    # Run fixes for all failed tasks that have fixes (recommended)
    python scripts/run_scicode_fixes.py \
        --prefix iter1_ \
        --rubric-csv rubrics_output/scicode/scicode_combined.csv \
        --docker

    # Force a specific model for all tasks
    python scripts/run_scicode_fixes.py \
        --prefix iter1_ \
        --model gpt-4o-2024-11-20 \
        --docker

    # Actually run fixes (legacy mode - all fixes, default model)
    python scripts/run_scicode_fixes.py \
        --agent-dir hal-harness/agents/scicode_tool_calling_agent \
        --agent-args agent_args.json \
        --task-id 11 \
        --output-prefix fixed_scicode
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

# If using direct Azure, remove proxy URLs that might be set in shell environment
if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
    for key in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_BASE_URL', 'LITELLM_BASE_URL'):
        os.environ.pop(key, None)
    print("[INFO] Direct Azure mode: removed proxy URLs from environment")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXES_DIR = REPO_ROOT / "fixes"
TRACES_DIR = REPO_ROOT / "traces"
HAL_HARNESS = REPO_ROOT / "hal-harness"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model_to_baseline_scicode.json"
TMP_DIR = REPO_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)  # Ensure .tmp directory exists
FAILED_TASKS_FILE = REPO_ROOT / ".tmp" / "scicode_failed_tasks.json"


def log(msg: str, prefix: str = "main") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{prefix}] {msg}", flush=True)


def is_retryable_error(error_msg: str) -> bool:
    """Check if an error is retryable (TRAPI timeout, auth, rate limit)."""
    retryable_patterns = [
        # Timeouts
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "connection error",
        # HTTP errors
        "503",
        "504",
        "502",
        "500",  # Internal server error
        "429",  # Rate limit
        "rate limit",
        # Auth/Token errors (CRITICAL: TRAPI token expiration)
        "401",
        "403",
        "unauthorized",
        "authentication",
        "invalid token",
        "expired token",
        "invalid or expired token",
        "token expired",
        "authenticationerror",
        # Other retryable errors
        "invalid_request_error",
        "insufficient_quota",
        "overloaded",
        "overloaded_error",
        "network error",
        "broken pipe",
        "EOF occurred",
        "service unavailable",
        "bad gateway",
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

    FAILED_TASKS_FILE.write_text(json.dumps(failed_tasks, indent=2))


def run_with_retry(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    task_id: str,
    model_id: str,
    max_retries: int = 3,
    base_timeout: int = 3600,
) -> Tuple[bool, str, Optional[subprocess.CompletedProcess]]:
    """Run subprocess with retry logic for TRAPI timeouts and auth errors.

    Returns:
        (success, error_msg, result)
    """
    import random

    for attempt in range(max_retries):
        timeout = base_timeout * (attempt + 1)  # Increase timeout on retries

        try:
            # Only print retry message on actual retries (not first attempt)
            if attempt > 0:
                log(f"Attempt {attempt + 1}/{max_retries} (timeout={timeout}s)", "retry")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )

            # Check stderr for errors
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            combined_output = stderr + stdout

            # Success
            if result.returncode == 0:
                return True, "Success", result

            # Check if error is retryable
            if is_retryable_error(combined_output):
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 5))
                log(f"Retryable error detected. Waiting {wait_time:.1f}s before retry...", "retry")
                log(f"Error snippet: {combined_output[:200]}", "retry")
                time.sleep(wait_time)
                continue
            else:
                # Non-retryable error, fail immediately
                log(f"Non-retryable error: {combined_output[:200]}", "retry")
                save_failed_task(task_id, model_id, combined_output[:500])
                return False, f"Exit code {result.returncode}", result

        except subprocess.TimeoutExpired as e:
            log(f"Timeout on attempt {attempt + 1}/{max_retries}", "retry")
            if attempt < max_retries - 1:
                wait_time = min(30, (2 ** attempt) + random.uniform(0, 3))
                log(f"Retrying after {wait_time:.1f}s...", "retry")
                time.sleep(wait_time)
            else:
                save_failed_task(task_id, model_id, "Timeout after retries")
                return False, "Timeout after retries", None

        except Exception as e:
            error_msg = str(e)
            log(f"Exception on attempt {attempt + 1}/{max_retries}: {error_msg}", "retry")

            if is_retryable_error(error_msg) and attempt < max_retries - 1:
                wait_time = min(60, (2 ** attempt) + random.uniform(0, 5))
                log(f"Retrying after {wait_time:.1f}s...", "retry")
                time.sleep(wait_time)
            else:
                save_failed_task(task_id, model_id, error_msg)
                return False, error_msg, None

    save_failed_task(task_id, model_id, "Max retries exceeded")
    return False, "Max retries exceeded", None


def _slugify(value: str, fallback: str) -> str:
    """Keep readable identifiers for run_ids / filenames."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or fallback


def load_model_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load model_to_baseline_scicode.json and return a dict mapping model keys to config.

    Format (same as corebench):
    {
      "openai/gpt-4.1-2025-04-14": {
        "model_id": "openai/gpt-4.1_2025-04-14",
        "short_name": "gpt-4.1-04-14",
        "baseline_trace": "scicode_..._UPLOAD.json",
        "reasoning_effort": "high",  // optional
        "max_steps": 5
      },
      ...
    }
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))

    # Flat dict format (corebench style) - this is the expected format
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
                # Extract model_id from model_run
                # Format examples:
                # - scicode_hal_generalist_agent_gpt4120250414_1745597325_UPLOAD
                # - scicode_scicode_tool_calling_agent_claudesonnet45_high_1759429729_UPLOAD
                model_id = _extract_model_from_run_name(model_run)
                if model_id and (task_id, model_id) not in seen:
                    task_model_pairs.append((task_id, model_id))
                    seen.add((task_id, model_id))

    return task_model_pairs


def _extract_model_from_run_name(model_run: str) -> Optional[str]:
    """Extract a model config key from the model_run column.

    Returns the key that matches model_to_baseline_scicode.json (e.g., "openai/gpt-4.1-2025-04-14").

    Examples:
    - scicode_hal_generalist_agent_gpt4120250414_... -> openai/gpt-4.1-2025-04-14
    - scicode_scicode_tool_calling_agent_claudesonnet45_high_... -> anthropic/claude-sonnet-4-5-20250929-high
    - scicode_hal_generalist_agent_o4mini20250416_low_... -> openai/o4-mini-2025-04-16-low
    """
    # Check for reasoning effort suffix
    has_high = "_high_" in model_run.lower() or "_high" in model_run.lower()
    has_low = "_low_" in model_run.lower() or "_low" in model_run.lower()

    # Known model patterns and their config keys (order matters - more specific first)
    patterns = [
        # GPT models
        (r"gpt[\-_]?5|gpt5", "openai/gpt-5-0125"),
        (r"gpt[\-_]?4[\-_]?1[\-_]?2025[\-_]?04[\-_]?14|gpt4120250414", "openai/gpt-4.1-2025-04-14"),
        (r"gpt4o[\-_]?2024[\-_]?11[\-_]?20", "openai/gpt-4o-2024-11-20"),
        # OpenAI reasoning models - check effort suffix
        (r"o3[\-_]?2025[\-_]?04[\-_]?16|o320250416", "openai/o3-2025-04-16"),
        (r"o4[\-_]?mini[\-_]?2025[\-_]?04[\-_]?16|o4mini20250416", None),  # Handle separately
        # Claude models - check effort suffix
        (r"claude[\-_]?opus[\-_]?4[\-_]?1|claudeopus41", None),  # Handle separately
        (r"claude[\-_]?sonnet[\-_]?4[\-_]?5|claudesonnet45", None),  # Handle separately
        (r"claude[\-_]?3[\-_]?7[\-_]?sonnet|claude37sonnet", None),  # Handle separately
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
                pricing_key = key  # Use the config key for pricing
                break

    if not model_entry:
        return None

    # Build agent_args from model config
    # model_id is for API calls, pricing_key is for cost tracking
    model_id = model_entry.get("model_id")
    if not model_id:
        return None

    agent_args: Dict[str, Any] = {
        "model_name": model_id,  # Used for API calls
    }
    if "reasoning_effort" in model_entry:
        agent_args["reasoning_effort"] = model_entry["reasoning_effort"]
    if "max_steps" in model_entry:
        agent_args["max_steps"] = model_entry["max_steps"]

    return agent_args, pricing_key


def load_fix_package(task_id: str, fixes_root: Path) -> Dict[str, Any]:
    """Load all fix files for a task."""
    fix_dir = fixes_root / task_id
    if not fix_dir.exists():
        return {}

    fix = {"task_id": task_id, "fix_dir": fix_dir}

    # Load different override types
    for name in ["dependency_override", "instruction_override", "evaluation_override", "env_override", "input_override"]:
        path = fix_dir / f"{name}.json"
        if path.exists():
            fix[name] = json.loads(path.read_text())

    # Load problem_statement.txt if exists (common format)
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
                "dependency_override.json", "instruction_override.json",
                "evaluation_override.json", "env_override.json",
                "input_override.json", "problem_statement.txt",
            ]
            has_actual_fix = any((item / f).exists() for f in actual_fix_files)

            if has_actual_fix:
                task_ids.append(item.name)
            elif include_readme_only and (item / "README.md").exists():
                task_ids.append(item.name)

    return sorted(task_ids)


def load_scicode_dataset() -> List[Dict[str, Any]]:
    """Load the SciCode dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        dataset = list(load_dataset("SciCode1/SciCode", split="test"))
        return dataset
    except Exception as e:
        log(f"Failed to load SciCode dataset: {e}", "data")
        return []


def apply_fix_to_task(task: Dict[str, Any], fix: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Apply a fix to a task, modifying function headers, dependencies, and test cases.

    Returns:
        Tuple of (modified_task, list of changes made)
    """
    modified = json.loads(json.dumps(task))  # Deep copy
    changes_made = []
    task_id = str(task.get("problem_id", ""))

    # =============================================================
    # 1. Apply instruction_override - Fix function headers
    # =============================================================
    if fix.get("instruction_override"):
        instr = fix["instruction_override"]

        # Handle both "overrides" and "step_overrides" keys
        overrides = instr.get("overrides", {})
        step_overrides = instr.get("step_overrides", {})
        overrides.update(step_overrides)

        # Also check for step-specific keys at top level (e.g., "58.2": {...})
        for key, value in instr.items():
            if key not in ["task_id", "description", "overrides", "step_overrides", "notes", "clarifications", "problem_id", "reason"] and isinstance(value, dict):
                overrides[key] = value

        for step_key, step_fix in overrides.items():
            # step_key is like "71.1" or "71.2"
            parts = step_key.split(".")
            if len(parts) == 2:
                try:
                    step_idx = int(parts[1]) - 1  # Convert to 0-indexed
                except ValueError:
                    continue

                if 0 <= step_idx < len(modified.get("sub_steps", [])):
                    step = modified["sub_steps"][step_idx]

                    # Fix function header - method 1: original/fixed pair
                    if step_fix.get("original_header") and step_fix.get("fixed_header"):
                        original = step_fix["original_header"]
                        fixed = step_fix["fixed_header"]
                        old_header = step.get("function_header", "")

                        if original in old_header:
                            step["function_header"] = old_header.replace(original, fixed)
                            changes_made.append(f"Step {step_key}: Fixed header '{original}' -> '{fixed}'")
                        else:
                            # Try more flexible matching (just the def line)
                            lines = old_header.split("\n")
                            for i, line in enumerate(lines):
                                if line.strip().startswith("def ") and original.strip() in line:
                                    lines[i] = line.replace(original.strip(), fixed.strip())
                                    step["function_header"] = "\n".join(lines)
                                    changes_made.append(f"Step {step_key}: Fixed header '{original}' -> '{fixed}'")
                                    break

                    # Fix function header - method 2: full replacement
                    elif step_fix.get("function_header"):
                        new_header = step_fix["function_header"]
                        old_header = step.get("function_header", "")
                        if old_header != new_header:
                            step["function_header"] = new_header
                            changes_made.append(f"Step {step_key}: Replaced function_header entirely")

                    # Fix step description - method 1: fixed_description key
                    if step_fix.get("fixed_description"):
                        step["step_description_prompt"] = step_fix["fixed_description"]
                        changes_made.append(f"Step {step_key}: Replaced step description")
                    # Fix step description - method 2: step_description_prompt key (direct replacement)
                    elif step_fix.get("step_description_prompt"):
                        step["step_description_prompt"] = step_fix["step_description_prompt"]
                        changes_made.append(f"Step {step_key}: Replaced step description")

                    # Add clarification to step description if provided
                    if step_fix.get("clarification"):
                        clarification = step_fix["clarification"]
                        step["step_description_prompt"] = (
                            step.get("step_description_prompt", "") +
                            f"\n\n**IMPORTANT**: {clarification}"
                        )
                        changes_made.append(f"Step {step_key}: Added clarification")

                    # Add reason as clarification if provided
                    if step_fix.get("reason") and not step_fix.get("clarification"):
                        reason = step_fix["reason"]
                        step["step_description_prompt"] = (
                            step.get("step_description_prompt", "") +
                            f"\n\n**NOTE**: {reason}"
                        )
                        changes_made.append(f"Step {step_key}: Added reason note")

                    # Fix output description in docstring if provided
                    if step_fix.get("corrected_output_description"):
                        # This would require parsing the docstring - add as clarification for now
                        pass

    # =============================================================
    # 2. Apply dependency_override - Fix required_dependencies
    # =============================================================
    if fix.get("dependency_override"):
        dep = fix["dependency_override"]

        if dep.get("fixed_dependencies"):
            old_deps = modified.get("required_dependencies", "")
            new_deps = dep["fixed_dependencies"]
            modified["required_dependencies"] = new_deps

            # Log what changed
            old_imports = set(old_deps.strip().split("\n"))
            new_imports = set(new_deps.strip().split("\n"))
            added = new_imports - old_imports
            removed = old_imports - new_imports

            if removed:
                changes_made.append(f"Dependencies: Removed {removed}")
            if added:
                changes_made.append(f"Dependencies: Added {added}")
            if not added and not removed and old_deps != new_deps:
                changes_made.append(f"Dependencies: Modified imports")

        # Handle additional_imports (append to existing)
        if dep.get("additional_imports"):
            old_deps = modified.get("required_dependencies", "")
            for imp in dep["additional_imports"]:
                if imp not in old_deps:
                    modified["required_dependencies"] = old_deps + f"\n{imp}"
                    changes_made.append(f"Dependencies: Added '{imp}'")

    # =============================================================
    # 3. Apply evaluation_override - Fix test cases and add preambles
    # =============================================================
    if fix.get("evaluation_override"):
        eval_fix = fix["evaluation_override"]

        # Handle preamble - code to PREPEND to required_dependencies for compatibility
        # Must be prepended so shims run BEFORE the original imports
        if eval_fix.get("preamble"):
            preamble = eval_fix["preamble"]
            old_deps = modified.get("required_dependencies", "")
            if preamble not in old_deps:
                modified["required_dependencies"] = preamble + "\n" + old_deps
                changes_made.append(f"Evaluation: Prepended compatibility preamble")

        # Handle apply_to_all_steps flag (for preamble that affects all steps)
        if eval_fix.get("apply_to_all_steps") and eval_fix.get("preamble"):
            changes_made.append(f"Evaluation: Preamble applies to all steps")

        for step_key, test_fix in eval_fix.items():
            # Skip non-step keys
            if step_key in ["preamble", "apply_to_all_steps", "description", "notes", "reason"]:
                continue

            if not isinstance(test_fix, dict):
                continue

            parts = step_key.split(".")
            if len(parts) == 2:
                try:
                    step_idx = int(parts[1]) - 1
                except ValueError:
                    continue

                if 0 <= step_idx < len(modified.get("sub_steps", [])):
                    step = modified["sub_steps"][step_idx]
                    test_cases = step.get("test_cases", [])

                    if test_fix.get("test_case_fix") == "replace_function_call":
                        original_func = test_fix.get("original_function", "")
                        corrected_func = test_fix.get("corrected_function", "")

                        if original_func and corrected_func:
                            new_test_cases = []
                            for tc in test_cases:
                                if original_func in tc:
                                    new_tc = tc.replace(original_func, corrected_func)
                                    new_test_cases.append(new_tc)
                                    changes_made.append(
                                        f"Step {step_key}: Fixed test case '{original_func}' -> '{corrected_func}'"
                                    )
                                else:
                                    new_test_cases.append(tc)
                            step["test_cases"] = new_test_cases

                    # Handle direct test case replacement
                    if test_fix.get("original_test") and test_fix.get("fixed_test"):
                        original_test = test_fix["original_test"]
                        fixed_test = test_fix["fixed_test"]
                        new_test_cases = []
                        for tc in test_cases:
                            if original_test in tc:
                                new_test_cases.append(tc.replace(original_test, fixed_test))
                                changes_made.append(f"Step {step_key}: Replaced test case")
                            else:
                                new_test_cases.append(tc)
                        step["test_cases"] = new_test_cases

    # =============================================================
    # 4. Apply env_override - Environment variables (for harness)
    # =============================================================
    if fix.get("env_override"):
        # Store env overrides in a special field for the harness to read
        modified["_env_override"] = fix["env_override"]
        changes_made.append(f"Environment: Added overrides {list(fix['env_override'].keys())}")

    # =============================================================
    # 5. Apply input_override - Additional clarifications
    # =============================================================
    if fix.get("input_override"):
        inp = fix["input_override"]
        clarifications = inp.get("clarifications", [])

        if clarifications:
            clarification_text = "\n\n## IMPORTANT CLARIFICATIONS\n" + "\n".join(f"- {c}" for c in clarifications)

            if "problem_description_main" in modified:
                modified["problem_description_main"] += clarification_text
            elif modified.get("sub_steps"):
                modified["sub_steps"][0]["step_description_prompt"] = (
                    modified["sub_steps"][0].get("step_description_prompt", "") + clarification_text
                )
            changes_made.append(f"Added {len(clarifications)} clarifications to prompt")

        if inp.get("corrected_instructions"):
            # Replace the main problem description
            modified["problem_description_main"] = inp["corrected_instructions"]
            changes_made.append("Replaced problem description with corrected instructions")

    # =============================================================
    # 6. Apply problem_statement.txt - Raw text addition
    # =============================================================
    if fix.get("problem_statement"):
        ps_text = f"\n\n## ADDITIONAL CONTEXT\n{fix['problem_statement']}"
        if "problem_description_main" in modified:
            modified["problem_description_main"] += ps_text
        changes_made.append("Added problem statement context")

    return modified, changes_made


def create_filtered_dataset(
    tasks: List[Dict[str, Any]],
    fixes: Dict[str, Dict[str, Any]],
    task_ids: List[str],
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Create a filtered dataset with only the specified tasks, with fixes applied.

    Returns:
        Tuple of (filtered_tasks, dict mapping task_id to list of changes made)
    """
    filtered = []
    all_changes: Dict[str, List[str]] = {}

    for task in tasks:
        problem_id = str(task.get("problem_id", ""))
        if problem_id in task_ids:
            if problem_id in fixes:
                modified, changes = apply_fix_to_task(task, fixes[problem_id])
                all_changes[problem_id] = changes

                if verbose and changes:
                    log(f"Applied {len(changes)} fix(es) to task {problem_id}:", "fix")
                    for change in changes:
                        log(f"  - {change}", "fix")
                elif verbose:
                    log(f"No changes applied to task {problem_id} (fix had no applicable overrides)", "fix")
            else:
                modified = task
            filtered.append(modified)

    return filtered, all_changes


def run_hal_eval(
    agent_dir: Path,
    agent_args: Dict[str, Any],
    dataset_path: Path,
    output_prefix: str,
    benchmark: str = "scicode",
    docker: bool = False,
    task_id: Optional[str] = None,
    pricing_key: Optional[str] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Run HAL evaluation with a custom dataset.

    Runs from REPO_ROOT (like corebench) so results go to results/scicode/.

    Args:
        pricing_key: The config key used for pricing lookup (may differ from model_id).
                     If None, falls back to model_name from agent_args.
    """

    # Build run_id with model info for better traceability
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

    if docker:
        cmd.append("--docker")

    for key, value in agent_args.items():
        if isinstance(value, (dict, list)):
            cmd += ["-A", f"{key}={json.dumps(value)}"]
        else:
            cmd += ["-A", f"{key}={value}"]

    # Set environment (like corebench does)
    env = os.environ.copy()
    env["SCICODE_DATASET_PATH"] = str(dataset_path)

    # Add hal-harness to PYTHONPATH so hal.cli can be found when running from REPO_ROOT
    extra_path = str(HAL_HARNESS)
    env["PYTHONPATH"] = f"{extra_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env.setdefault("WANDB_SILENT", "true")
    # Use pricing_key for cost tracking (config key), model_name is for API calls (model_id)
    effective_pricing_key = pricing_key or str(agent_args.get("model_name", ""))
    env["HAL_PRICING_MODEL_NAME"] = effective_pricing_key  # Use assignment, not setdefault
    # Set weave project name based on prefix (e.g., scicode_lime_scicode)
    weave_project = f"{output_prefix.rstrip('_')}_{benchmark}"
    env["HAL_WEAVE_PROJECT"] = weave_project

    log(f"Running HAL eval with run_id: {run_id}", "hal")
    log(f"Model (API): {agent_args.get('model_name')} | Pricing key: {effective_pricing_key}", "hal")
    log(f"Dataset: {dataset_path}", "hal")
    log(f"Command: {' '.join(cmd)}", "hal")

    # Results directory (at REPO_ROOT, not HAL_HARNESS)
    results_dir = REPO_ROOT / "results" / benchmark / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use retry wrapper with exponential backoff
    task_id_str = task_id or "unknown"
    model_id_str = str(agent_args.get("model_name", "unknown"))
    success, error_msg, result = run_with_retry(
        cmd=cmd,
        env=env,
        cwd=REPO_ROOT,
        task_id=task_id_str,
        model_id=model_id_str,
        max_retries=3,
        base_timeout=3600,
    )

    if not success:
        log(f"HAL eval failed after retries: {error_msg}", "hal")
        return False, error_msg, None

    # Find output trace in REPO_ROOT/results (like corebench)
    trace_path = results_dir / f"{run_id}_UPLOAD.json"

    # Also check HAL_HARNESS results as fallback
    if not trace_path.exists():
        hal_results_dir = HAL_HARNESS / "results" / benchmark / run_id
        alt_trace = hal_results_dir / f"{run_id}_UPLOAD.json"
        if alt_trace.exists():
            # Copy to expected location
            results_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(alt_trace, trace_path)
            # Also copy logs if they exist
            for log_file in hal_results_dir.glob("*.log"):
                shutil.copy(log_file, results_dir / log_file.name)

    if trace_path.exists():
        # Copy trace to our traces directory with prefix
        dest = TRACES_DIR / f"{output_prefix}_{trace_path.name}"
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(trace_path, dest)
        log(f"Trace saved to: {dest}", "hal")
        log(f"Results in: {results_dir}", "hal")
        return True, "Success", dest

    return True, "Success (no trace found)", None


def main():
    parser = argparse.ArgumentParser(
        description="Apply SciCode fixes and re-run HAL evaluations."
    )

    parser.add_argument(
        "--fixes-root",
        default=str(FIXES_DIR / "scicode"),
        help="Directory containing fix packages (default: fixes/scicode).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(HAL_HARNESS / "agents" / "scicode_tool_calling_agent"),
        help="Path to the agent directory.",
    )
    parser.add_argument(
        "--agent-args",
        help="Path to JSON file with agent arguments.",
    )
    parser.add_argument(
        "--benchmark",
        default="scicode",
        help="Benchmark name (default: scicode).",
    )
    parser.add_argument(
        "--output-prefix",
        default="fixed",
        help="Prefix for output traces (default: fixed).",
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for run IDs and output files (e.g., 'iter1_'). Takes precedence over --output-prefix.",
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
        help="Run HAL evaluation with --docker flag.",
    )
    parser.add_argument(
        "--rubric-csv",
        help="Path to rubric CSV to determine which model failed for each task.",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model_to_baseline_scicode.json mapping model names to configs.",
    )
    parser.add_argument(
        "--model",
        help="Force a specific model for all tasks (overrides rubric CSV). Use config key like 'openai/gpt-4.1-2025-04-14'.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all models from model_to_baseline_scicode.json for each task with fixes.",
    )
    parser.add_argument(
        "--verify-fixes",
        action="store_true",
        help="Verify fixes are applied correctly without running HAL eval. Shows before/after for each fix.",
    )
    parser.add_argument(
        "--merge-traces",
        action="store_true",
        help="Merge existing trace files by model (no HAL runs). Looks for traces with the given prefix.",
    )

    args = parser.parse_args()

    # Resolve paths
    fixes_root = Path(args.fixes_root)
    if not fixes_root.is_absolute():
        fixes_root = REPO_ROOT / fixes_root

    agent_dir = Path(args.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = REPO_ROOT / agent_dir

    # List fixes mode
    if args.list_fixes:
        # Get tasks with actual fixes
        with_fixes = list_available_fixes(fixes_root, include_readme_only=False)
        # Get all analyzed tasks (including README-only)
        all_analyzed = list_available_fixes(fixes_root, include_readme_only=True)
        readme_only = set(all_analyzed) - set(with_fixes)

        print(f"\n{'='*60}")
        print(f"FIXES AVAILABLE in {fixes_root.name}/")
        print(f"{'='*60}\n")

        print(f"Tasks with ACTUAL FIXES ({len(with_fixes)}):")
        for task_id in with_fixes:
            fix = load_fix_package(task_id, fixes_root)
            types = []
            if fix.get("dependency_override"): types.append("dep")
            if fix.get("instruction_override"): types.append("instr")
            if fix.get("evaluation_override"): types.append("eval")
            if fix.get("problem_statement"): types.append("prompt")
            if fix.get("env_override"): types.append("env")
            if fix.get("input_override"): types.append("input")
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

    # Verify fixes mode - test fix application without running HAL eval
    if args.verify_fixes:
        print(f"\n{'='*60}")
        print("VERIFY FIXES MODE")
        print(f"{'='*60}\n")

        log("Loading SciCode dataset from HuggingFace...", "verify")
        dataset = load_scicode_dataset()
        if not dataset:
            log("Failed to load dataset", "verify")
            return
        log(f"Loaded {len(dataset)} tasks", "verify")

        # Get tasks with fixes
        with_fixes = list_available_fixes(fixes_root, include_readme_only=False)
        if args.task_ids:
            with_fixes = [t for t in with_fixes if t in args.task_ids]

        log(f"Verifying {len(with_fixes)} fixes...\n", "verify")

        all_success = True
        for task_id in with_fixes:
            fix = load_fix_package(task_id, fixes_root)
            if not fix:
                continue

            # Find original task
            original_task = None
            for task in dataset:
                if str(task.get("problem_id")) == task_id:
                    original_task = task
                    break

            if not original_task:
                print(f"  ✗ Task {task_id}: NOT FOUND in dataset")
                all_success = False
                continue

            # Apply fix
            modified_task, changes = apply_fix_to_task(original_task, fix)

            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print(f"{'='*60}")

            if not changes:
                print(f"  ⚠ NO CHANGES APPLIED (fix may not match dataset structure)")
                all_success = False

                # Debug: show what the fix contains vs what the task has
                if fix.get("instruction_override"):
                    instr = fix["instruction_override"]
                    overrides = instr.get("overrides", {})
                    for step_key, step_fix in overrides.items():
                        if step_fix.get("original_header"):
                            print(f"\n  Fix expects to find: '{step_fix['original_header']}'")
                            parts = step_key.split(".")
                            if len(parts) == 2:
                                try:
                                    step_idx = int(parts[1]) - 1
                                    if 0 <= step_idx < len(original_task.get("sub_steps", [])):
                                        actual = original_task["sub_steps"][step_idx].get("function_header", "")[:100]
                                        print(f"  Actual in dataset: '{actual}...'")
                                except:
                                    pass
            else:
                print(f"  ✓ {len(changes)} change(s) applied:")
                for change in changes:
                    print(f"    - {change}")

                # Show before/after for key fields
                for i, step in enumerate(original_task.get("sub_steps", [])):
                    step_num = f"{task_id}.{i+1}"
                    orig_header = step.get("function_header", "")
                    mod_header = modified_task["sub_steps"][i].get("function_header", "")

                    if orig_header != mod_header:
                        # Extract just the def line
                        orig_def = next((l for l in orig_header.split("\n") if l.strip().startswith("def ")), "")
                        mod_def = next((l for l in mod_header.split("\n") if l.strip().startswith("def ")), "")
                        print(f"\n  Step {step_num} function_header:")
                        print(f"    BEFORE: {orig_def}")
                        print(f"    AFTER:  {mod_def}")

                    orig_tests = step.get("test_cases", [])
                    mod_tests = modified_task["sub_steps"][i].get("test_cases", [])
                    if orig_tests != mod_tests:
                        print(f"\n  Step {step_num} test_cases:")
                        for j, (ot, mt) in enumerate(zip(orig_tests, mod_tests)):
                            if ot != mt:
                                print(f"    [{j}] BEFORE: {ot[:80]}...")
                                print(f"    [{j}] AFTER:  {mt[:80]}...")

                # Check dependencies
                orig_deps = original_task.get("required_dependencies", "")
                mod_deps = modified_task.get("required_dependencies", "")
                if orig_deps != mod_deps:
                    print(f"\n  required_dependencies:")
                    print(f"    BEFORE: {orig_deps[:100]}...")
                    print(f"    AFTER:  {mod_deps[:100]}...")

        print(f"\n{'='*60}")
        if all_success:
            print("✓ All fixes verified successfully!")
        else:
            print("⚠ Some fixes had issues - review output above")
        print(f"{'='*60}\n")
        return

    # Merge traces mode - combine existing traces by model
    if args.merge_traces:
        prefix = args.prefix or args.output_prefix or "fixed"
        if not prefix.endswith("_"):
            prefix = prefix + "_"

        print(f"\n{'='*60}")
        print(f"MERGE TRACES MODE - prefix: {prefix}")
        print(f"{'='*60}\n")

        # Find all trace files with the prefix (exclude already merged files)
        trace_pattern = f"{prefix}*_UPLOAD.json"
        all_trace_files = list(TRACES_DIR.glob(trace_pattern))
        trace_files = [f for f in all_trace_files if "_MERGED_" not in f.name]

        if not trace_files:
            log(f"No trace files found matching {trace_pattern} in {TRACES_DIR}", "merge")
            return

        log(f"Found {len(trace_files)} trace files (excluding merged)", "merge")

        # Group traces by model
        # Filename format: prefix_prefix_model_task_benchmark_timestamp_UPLOAD.json
        # Example: scicode_potato__scicode_potato_openai_gpt-4_1_2025-04-14_28_scicode_20260117_123734_UPLOAD.json
        model_traces: Dict[str, List[Path]] = {}

        for trace_file in trace_files:
            name = trace_file.name
            # Extract model from filename - look for known model patterns
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

        # Merge traces for each model
        for model_key, traces in sorted(model_traces.items()):
            log(f"Merging {len(traces)} traces for {model_key}", "merge")

            # Collect all successful and failed tasks across all traces
            all_successful: Set[str] = set()
            all_failed: Set[str] = set()
            merged_config: Dict[str, Any] = {}
            merged_raw_eval: Dict[str, Any] = {"details": {}}
            total_cost = 0.0

            for trace_file in traces:
                try:
                    data = json.loads(trace_file.read_text())

                    # Collect successful/failed task IDs from results
                    if "results" in data:
                        results = data["results"]
                        for task_id in results.get("successful_tasks", []):
                            all_successful.add(str(task_id))
                            # If task was previously marked as failed, remove it (success overrides)
                            all_failed.discard(str(task_id))
                        for task_id in results.get("failed_tasks", []):
                            # Only add to failed if not already successful
                            if str(task_id) not in all_successful:
                                all_failed.add(str(task_id))
                        total_cost += results.get("total_cost", 0)

                    # Merge raw_eval_results details - combine unique subtasks
                    if "raw_eval_results" in data and "details" in data["raw_eval_results"]:
                        for task_id, subtasks in data["raw_eval_results"]["details"].items():
                            task_id_str = str(task_id)
                            if task_id_str not in merged_raw_eval["details"]:
                                merged_raw_eval["details"][task_id_str] = []
                            # Merge subtask lists, keeping unique subtasks
                            existing_subtasks = set(merged_raw_eval["details"][task_id_str])
                            for subtask in subtasks:
                                if subtask not in existing_subtasks:
                                    merged_raw_eval["details"][task_id_str].append(subtask)
                                    existing_subtasks.add(subtask)

                    # Keep config from first file
                    if not merged_config and "config" in data:
                        merged_config = data["config"]

                except Exception as e:
                    log(f"Error reading {trace_file.name}: {e}", "merge")

            # Calculate merged accuracy
            total_tasks = len(all_successful) + len(all_failed)
            accuracy = len(all_successful) / total_tasks if total_tasks > 0 else 0.0

            # Calculate subtask_accuracy from merged details
            # Count total successful subtasks across all tasks
            successful_subtasks = sum(
                len(subtasks) for subtasks in merged_raw_eval["details"].values()
            )
            # Scicode has 288 total subtasks across 80 tasks
            subtask_accuracy = successful_subtasks / 288 if successful_subtasks > 0 else 0.0

            # Create merged trace with proper structure
            merged_trace = {
                "config": merged_config,
                "results": {
                    "accuracy": accuracy,
                    "subtask_accuracy": subtask_accuracy,
                    "successful_tasks": sorted(all_successful, key=lambda x: (int(x) if x.isdigit() else float('inf'), x)),
                    "failed_tasks": sorted(all_failed, key=lambda x: (int(x) if x.isdigit() else float('inf'), x)),
                    "total_cost": total_cost,
                    "latencies": {},
                },
                "raw_eval_results": merged_raw_eval,
                "raw_logging_results": [],
                "total_usage": {},
                "total_cost": total_cost,
            }

            # Save merged trace
            # Format: {prefix}{model}_MERGED_{benchmark}_{timestamp}_UPLOAD.json
            model_slug = model_key.replace("/", "_").replace("-", "_")
            benchmark = merged_config.get("benchmark_name", "scicode")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            merged_filename = f"{prefix}{model_slug}_MERGED_{benchmark}_{timestamp}_UPLOAD.json"
            merged_path = TRACES_DIR / merged_filename

            merged_path.write_text(json.dumps(merged_trace, indent=2))
            log(f"Saved: {merged_path.name}", "merge")
            log(f"  Tasks: {len(all_successful)} successful, {len(all_failed)} failed ({total_tasks} total)", "merge")
            log(f"  Accuracy: {accuracy:.1%}", "merge")
            log(f"  Subtask accuracy: {subtask_accuracy:.1%} ({successful_subtasks}/288 subtasks)", "merge")

        print(f"\n{'='*60}")
        print(f"Merged {len(model_traces)} model traces")
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
    available_fixes = set(list_available_fixes(fixes_root))
    if args.task_ids:
        # Filter to only requested task IDs
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
        # Run all models from config for each task with fixes
        if not model_config:
            log("ERROR: --all-models requires model config file", "model")
            return
        for task_id in sorted(available_fixes):
            for model_key in model_config.keys():
                task_model_pairs.append((task_id, model_key))
        log(f"Running all {len(model_config)} models for {len(available_fixes)} tasks = {len(task_model_pairs)} total jobs", "model")
        for model_key in model_config.keys():
            log(f"  - {model_key}", "model")
    elif args.model:
        # Force a specific model for all fixes
        for task_id in sorted(available_fixes):
            task_model_pairs.append((task_id, args.model))
        log(f"Forced model '{args.model}' for all {len(task_model_pairs)} tasks", "model")
    elif args.rubric_csv:
        # Load failed (task, model) pairs from rubric CSV
        rubric_csv_path = Path(args.rubric_csv)
        if not rubric_csv_path.is_absolute():
            rubric_csv_path = REPO_ROOT / rubric_csv_path
        all_failed_pairs = load_rubric_task_models(rubric_csv_path)
        log(f"Loaded {len(all_failed_pairs)} failed (task, model) pairs from {rubric_csv_path.name}", "model")

        # Filter to only tasks with fixes
        for task_id, model_id in all_failed_pairs:
            if task_id in available_fixes:
                task_model_pairs.append((task_id, model_id))
        log(f"Filtered to {len(task_model_pairs)} pairs with available fixes", "model")
    elif model_config:
        # Default: run all models from config for each task with fixes
        for task_id in sorted(available_fixes):
            for model_key in model_config.keys():
                task_model_pairs.append((task_id, model_key))
        log(f"Running all {len(model_config)} models for {len(available_fixes)} tasks = {len(task_model_pairs)} total jobs", "model")
        for model_key in model_config.keys():
            log(f"  - {model_key}", "model")
    else:
        # Legacy mode: use all fixes with default model from agent_args
        log("No model config found, using legacy mode with agent_args", "model")
        for task_id in sorted(available_fixes):
            task_model_pairs.append((task_id, ""))  # Empty model = use agent_args

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

    # Dry run - show what would be done
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Would apply the following fixes:")
        print("="*60)

        # Group by task
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
            if fix.get("dependency_override"):
                dep = fix["dependency_override"]
                if dep.get("additional_imports"):
                    print(f"  Additional imports: {dep['additional_imports']}")
            if fix.get("problem_statement"):
                print(f"  Problem statement addition: {fix['problem_statement'][:100]}...")

        print(f"\n{'='*60}")
        print(f"Would run HAL eval with prefix: {prefix}")
        print(f"Docker: {args.docker}")
        print(f"Total jobs: {len(task_model_pairs)}")
        return

    # Load SciCode dataset
    log("Loading SciCode dataset from HuggingFace...", "data")
    dataset = load_scicode_dataset()
    if not dataset:
        log("Failed to load dataset", "main")
        return

    log(f"Loaded {len(dataset)} tasks from dataset", "data")

    # Load fallback agent args (for legacy mode)
    fallback_agent_args: Dict[str, Any] = {"model_name": "gpt-4o"}
    if args.agent_args:
        args_path = Path(args.agent_args)
        if not args_path.is_absolute():
            args_path = REPO_ROOT / args_path
        if args_path.exists():
            fallback_agent_args = json.loads(args_path.read_text())
            log(f"Loaded fallback agent args: {list(fallback_agent_args.keys())}", "main")

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

        # Write temporary dataset file
        temp_dataset = TMP_DIR / f"scicode_{task_id}_{model_id.replace('/', '_')}_{time.time()}.json"
        temp_dataset.write_text(json.dumps(filtered, indent=2))

        try:
            success, message, trace_path = run_hal_eval(
                agent_dir=agent_dir,
                agent_args=task_agent_args,
                dataset_path=temp_dataset,
                output_prefix=prefix,
                benchmark=args.benchmark,
                docker=args.docker,
                task_id=task_id,
                pricing_key=pricing_key,  # Config key for cost tracking
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
        # Parallel execution
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
