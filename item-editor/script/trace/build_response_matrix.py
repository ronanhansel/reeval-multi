#!/usr/bin/env python3
"""
Build a binary response matrix for a given prefix.

Rows: benchmark.config_key (agent row).
Columns: benchmark.task_id

Partial scoring:
- Only score tasks with non-error raw submissions.
- Missing/ERROR tasks are left blank.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HAL_HARNESS_PATH = REPO_ROOT / "hal-harness"
if str(HAL_HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HAL_HARNESS_PATH))

BENCHMARKS = ("scicode", "scienceagentbench", "corebench", "colbench", "assistantbench", "swebench_verified_mini", "usaco")
HAL_BENCHMARK_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
    "assistantbench": "assistantbench",
    "swebench_verified_mini": "swebench_verified_mini",
    "usaco": "usaco",
}
TASK_ID_FIELD = {
    "scicode": "problem_id",
    "scienceagentbench": "instance_id",
    "corebench": "capsule_id",
    "usaco": "problem_id",
    "assistantbench": "task_id",
    "swebench_verified_mini": "instance_id",
}

AUTH_ERROR_SNIPPETS = (
    "authenticationerror",
    "invalid_api_key",
    "incorrect api key",
    "unauthorized",
    "error code: 401",
)

SCICODE_CACHED_IMAGE = "scicode-eval:latest"


def _push_env(key: str, value: Optional[str]) -> Optional[str]:
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return old


def _extract_python_code(history_payload: Dict[str, object]) -> Optional[str]:
    try:
        history = history_payload.get("history", [])
    except AttributeError:
        return None
    if not history:
        return None
    last = history[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return None
    content = last.get("content", "")
    if not isinstance(content, str):
        return None
    match = re.search(r"```python(.*?)```", content, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def docker_image_exists(image_name: str) -> bool:
    try:
        import docker
    except ImportError:
        return False
    try:
        client = docker.from_env()
        client.images.get(image_name)
        return True
    except Exception:
        return False

def detect_run_root(script_dir: Path) -> Path:
    # 1. If results/logs/etc exist in CWD, use it
    cwd = Path(".").resolve()
    for d in ["result", "results", "logs", ".hal_data", ".hal-data", "output"]:
        if (cwd / d).exists():
            return cwd

    script_dir = script_dir.resolve()
    if script_dir.name == "script":
        repo_root = script_dir.parent
        project_name = repo_root.name
    else:
        repo_root = script_dir
        project_name = script_dir.name

    # 2. If results/logs exist in repo_root, use it
    for d in ["result", "results", "logs", ".hal_data", ".hal-data", "output"]:
        if (repo_root / d).exists():
            return repo_root

    # 3. Check home directory for a data root
    home = Path.home()
    for d in [".hal_data", ".hal-data"]:
        root = home / d / "hal_runs" / os.getlogin() / project_name
        if root.is_dir():
            return root

    # 4. DATA_PATH fallback
    data_path = os.environ.get("DATA_PATH") or os.environ.get("HAL_DATA_ROOT")
    if data_path and Path(data_path).is_dir():
        namespace = os.environ.get("HAL_DATA_NAMESPACE") or os.environ.get("USER") or os.getlogin() or "user"
        root = Path(data_path) / "hal_runs" / namespace / project_name
        if root.is_dir():
            return root
            
    return repo_root

def find_log_dir_for_prefix(log_base: Path, prefix_pattern: str) -> Optional[Path]:
    try:
        regex = re.compile(prefix_pattern)
    except re.error:
        regex = None

    for run_dir in sorted(log_base.glob("benchmark_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        cfg = run_dir / "config.json"
        if not cfg.exists():
            continue
        try:
            saved_prefix = json.loads(cfg.read_text()).get("prefix")
        except Exception:
            continue
        
        if regex:
            if regex.search(saved_prefix):
                return run_dir
        elif saved_prefix == prefix_pattern:
            return run_dir
    return None

def parse_run_ids(text: str) -> List[str]:
    return sorted(set(re.findall(r"Run ID: (\S+)", text)))

def parse_dataset_path(text: str, benchmark: str) -> Optional[Path]:
    # 1. Look for the "Saved modified dataset" pattern from run_benchmark_fixes.py
    # Example: [10:57:32] [main] Saved modified dataset (JSON format) to: /path/file.json
    pattern_saved = re.compile(rf"Saved modified dataset.*to:\s+([^\s]+)")
    matches_saved = pattern_saved.findall(text)
    if matches_saved:
        return Path(matches_saved[-1])

    # 2. Fallback to the environment variable pattern
    # Example: Custom dataset: SCICODE_DATASET_PATH=/path/file.json (65 tasks)
    env_var = {
        "scicode": "SCICODE_DATASET_PATH",
        "scienceagentbench": "SCIENCEAGENTBENCH_DATASET_PATH",
        "corebench": "HAL_COREBENCH_DATASET_PATH",
        "colbench": "COLBENCH_BACKEND_DATASET_PATH",
    }.get(benchmark)
    if not env_var:
        return None
    pattern_env = re.compile(rf"{env_var}=([^\s]+)")
    matches_env = pattern_env.findall(text)
    if not matches_env:
        return None
    return Path(matches_env[-1])

def find_latest_dataset(tmp_dir: Path, pattern: str) -> Optional[Path]:
    if not tmp_dir.exists():
        return None
    candidates = list(tmp_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def resolve_dataset_path(
    benchmark: str,
    log_text: str,
    run_root: Path,
    args: argparse.Namespace,
    prefix: str = "",
) -> Optional[Path]:
    cli_map = {
        "scicode": args.dataset_scicode,
        "scienceagentbench": args.dataset_scienceagentbench,
        "corebench": args.dataset_corebench,
        "colbench": args.dataset_colbench,
    }
    cli_value = cli_map.get(benchmark)
    if cli_value:
        path = Path(cli_value)
        if path.exists():
            return path

    env_map = {
        "scicode": "SCICODE_DATASET_PATH",
        "scienceagentbench": "SCIENCEAGENTBENCH_DATASET_PATH",
        "corebench": "HAL_COREBENCH_DATASET_PATH",
        "colbench": "COLBENCH_BACKEND_DATASET_PATH",
    }
    env_val = os.environ.get(env_map.get(benchmark, ""))
    if env_val:
        path = Path(env_val)
        if path.exists():
            return path

    custom_path_from_log = None
    if log_text:
        custom_path_from_log = parse_dataset_path(log_text, benchmark)
        if custom_path_from_log and custom_path_from_log.exists():
            return custom_path_from_log

    # If run_root has a logs or results symlink, use its target as a base for tmp too
    real_run_root = run_root.resolve()
    data_root = detect_run_root(REPO_ROOT)
    
    tmp_dirs = [
        run_root / "tmp", 
        run_root / ".tmp",
        real_run_root / "tmp",
        real_run_root / ".tmp",
        run_root / ".hal_data" / "tmp", 
        run_root / ".hal_data" / ".tmp",
        run_root / ".hal-data" / "tmp", 
        run_root / ".hal-data" / ".tmp",
        REPO_ROOT / "tmp", 
        REPO_ROOT / ".tmp",
        data_root / "tmp",
        data_root / ".tmp"
    ]
    path = None
    for tmp_dir in tmp_dirs:
        if not tmp_dir.exists():
            continue
            
        # If prefix is provided, try to find a dataset that might be associated with it
        if prefix:
            clean_prefix = prefix.strip("_")
            patterns = {
                "scicode": f"scicode_modified_*{{clean_prefix}}*.json",
                "scienceagentbench": f"scienceagentbench_modified_*{{clean_prefix}}*.json",
                "corebench": f"corebench_modified_*{{clean_prefix}}*.json",
                "colbench": f"colbench_modified_*{{clean_prefix}}*.jsonl",
            }
            if benchmark in patterns:
                path = find_latest_dataset(tmp_dir, patterns[benchmark])
                if path and path.exists():
                    return path

        # Fallback to generic modified dataset
        if benchmark == "scicode":
            path = find_latest_dataset(tmp_dir, "scicode_modified_*.json")
        
        if path and path.exists():
            return path

    if benchmark == "corebench":
        # If the log indicated a custom dataset was used, but we couldn't find it (likely cleaned up),
        # return None so that we fall back to inferring tasks from the results.
        # This prevents populating the matrix with 45 tasks when only a subset was run.
        if custom_path_from_log:
            return None

        default_path = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "corebench" / "core_test.json"
        if default_path.exists():
            return default_path

    if benchmark == "scicode":
        default_path = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "scicode" / "scicode.json"
        if default_path.exists():
            return default_path

    return None

def derive_config_key(run_id: str, benchmark: str, prefix_pattern: str) -> str:
    if run_id.startswith(f"{benchmark}_"):
        run_id = run_id[len(benchmark) + 1 :].lstrip("_")
    
    # Try stripping prefix as regex first
    try:
        run_id = re.sub(prefix_pattern, "", run_id, count=1).lstrip("_")
    except re.error:
        if run_id.startswith(prefix_pattern):
            run_id = run_id[len(prefix_pattern) :].lstrip("_")
            
    return re.sub(r"_[0-9]{8}_[0-9]{6}$", "", run_id)

def resolve_run_dir(run_root: Path, repo_root: Path, benchmark: str, run_id: str, traces_dir: Optional[str] = None) -> Optional[Path]:
    if traces_dir:
        # When using aggregated traces, the 'run_dir' is effectively the traces folder
        # or we just return the traces_dir itself and let loaders look for benchmark_runid files
        return Path(traces_dir)

    hal_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
    candidates = [
        run_root / "results" / hal_name / run_id,
        run_root / "results" / benchmark / run_id,
        run_root / ".hal_data" / "results" / hal_name / run_id,
        run_root / ".hal-data" / "results" / hal_name / run_id,
        repo_root / "results" / hal_name / run_id,
        repo_root / "results" / benchmark / run_id,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def find_run_ids_from_results(
    run_root: Path,
    repo_root: Path,
    benchmark: str,
    prefix_pattern: str,
    traces_dir: Optional[str] = None
) -> List[Tuple[str, str]]:
    if traces_dir:
        # Search in the aggregated traces folder
        search_dir = Path(traces_dir) / "traces"
        if not search_dir.exists():
            search_dir = Path(traces_dir)
        
        run_ids: List[Tuple[str, str]] = []
        try:
            regex = re.compile(prefix_pattern)
        except re.error:
            regex = None

        if not search_dir.exists():
            return []

        for f in search_dir.glob("*.json"):
            # Expecting name format like {benchmark}_{run_id}_UPLOAD.json
            # Robustness: Check for both short name and full mapped name
            full_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
            
            rid = None
            if f.name.startswith(f"{benchmark}_") and f.name.endswith("_UPLOAD.json"):
                rid = f.name[len(benchmark)+1 : -len("_UPLOAD.json")]
            elif f.name.startswith(f"{full_name}_") and f.name.endswith("_UPLOAD.json"):
                rid = f.name[len(full_name)+1 : -len("_UPLOAD.json")]
            
            if rid:
                if regex:
                    m = regex.search(f.name)
                    if m:
                        run_ids.append((rid, m.group(0)))
                elif prefix_pattern in f.name:
                    run_ids.append((rid, prefix_pattern))
        return sorted(list(set(run_ids)))

    hal_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
    candidates = [
        run_root / "results" / hal_name,
        run_root / "results" / benchmark,
        run_root / ".hal_data" / "results" / hal_name,
        run_root / ".hal-data" / "results" / hal_name,
        repo_root / "results" / hal_name,
        repo_root / "results" / benchmark,
    ]

    run_ids: List[Tuple[str, str]] = []
    try:
        regex = re.compile(prefix_pattern)
    except re.error:
        regex = None

    for base in candidates:
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if regex:
                m = regex.search(child.name)
                if m:
                    run_ids.append((child.name, m.group(0)))
            elif prefix_pattern in child.name:
                run_ids.append((child.name, prefix_pattern))
    return sorted(list(set(run_ids)))

def is_error_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("ERROR"):
        return True
    lowered = value.lower()
    return any(snippet in lowered for snippet in AUTH_ERROR_SNIPPETS)

def is_non_null_scalar(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True

def is_non_null_submission(benchmark: str, task_id: str, payload: object, task_meta: Optional[Dict[str, List[str]]]) -> bool:
    if payload is None:
        return False
    if benchmark == "scicode":
        if not isinstance(payload, dict):
            return False
        required = task_meta.get(task_id) if task_meta else None
        if required:
            for key in required:
                if key not in payload or not is_non_null_scalar(payload.get(key)):
                    return False
            return True
        for value in payload.values():
            if not is_non_null_scalar(value):
                return False
        return len(payload) > 0

    if benchmark == "corebench":
        if not isinstance(payload, dict):
            return False
        for value in payload.values():
            if not is_non_null_scalar(value):
                return False
        return len(payload) > 0

    if benchmark == "colbench":
        if not isinstance(payload, dict):
            return False
        answer = payload.get("answer")
        return is_non_null_scalar(answer)

    if benchmark == "scienceagentbench":
        if not isinstance(payload, dict):
            return False
        code = _extract_python_code(payload)
        return is_non_null_scalar(code)

    return is_non_null_scalar(payload)

def load_raw_ok_tasks(raw_path: Path, benchmark: str, task_meta: Optional[Dict[str, List[str]]]) -> Dict[str, object]:
    ok: Dict[str, object] = {}
    if not raw_path.exists():
        return ok
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            task_id = next(iter(obj.keys()))
            value = obj[task_id]
            
            # Robustness: attempt to extract JSON dictionary from string
            if isinstance(value, str):
                # Look for something that looks like a JSON dict
                match = re.search(r"\{.*\}", value, re.DOTALL)
                if match:
                    try:
                        value = json.loads(match.group(0))
                    except:
                        pass

            if not is_error_value(value):
                task_id = str(task_id)
                if is_non_null_submission(benchmark, task_id, value, task_meta):
                    ok[task_id] = value
    return ok

def load_task_ids(benchmark: str, dataset_path: Path) -> List[str]:
    if benchmark == "colbench":
        tasks = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(line)
        return [str(i) for i in range(len(tasks))]

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    field = TASK_ID_FIELD[benchmark]
    return [str(task[field]) for task in data]

def build_scicode_task_meta(dataset_path: Path) -> Dict[str, List[str]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    meta: Dict[str, List[str]] = {}
    for task in data:
        task_id = str(task.get("problem_id"))
        sub_steps = task.get("sub_steps", [])
        keys = [f"{task_id}.{idx + 1}" for idx in range(len(sub_steps))]
        meta[task_id] = keys
    return meta

def load_upload_success(upload_path: Path) -> Optional[set]:
    if not upload_path.exists():
        return None
    data = json.loads(upload_path.read_text(encoding="utf-8"))
    successful = data.get("results", {}).get("successful_tasks")
    if isinstance(successful, list):
        return {str(t) for t in successful}
    return None

def load_colbench_correctness_from_file(upload_path: Path) -> Optional[List[float]]:
    if not upload_path.exists():
        return None
    data = json.loads(upload_path.read_text(encoding="utf-8"))
    correctness = data.get("raw_eval_results")
    if isinstance(correctness, list):
        return [float(v) for v in correctness]
    return None

def load_scienceagentbench_eval_from_file(eval_path: Path, task_ids: List[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if not eval_path.exists():
        return results
    ordered_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    for idx, task_id in enumerate(ordered_ids):
        if idx >= len(lines):
            continue
        line = lines[idx].strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            success_rate = payload.get("success_rate")
            if isinstance(success_rate, (int, float)):
                results[task_id] = float(success_rate)
    return results

def load_scicode_subtask_scores_from_file(upload_path: Path, task_meta: Dict[str, List[str]]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if not upload_path.exists():
        return results
    try:
        data = json.loads(upload_path.read_text(encoding="utf-8"))
        details = data.get("raw_eval_results", {}).get("details", {})
        for task_id, all_sub in task_meta.items():
            if not all_sub:
                continue
            passed_sub = details.get(task_id, [])
            results[task_id] = len(passed_sub) / len(all_sub)
    except Exception:
        pass
    return results

def load_corebench_subtask_scores_from_file(upload_path: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if not upload_path.exists():
        return results
    try:
        data = json.loads(upload_path.read_text(encoding="utf-8"))
        raw = data.get("raw_eval_results", {})
        for task_id, val in raw.items():
            res = {}
            tw = val.get("total_written_questions", 0)
            if tw > 0:
                res["written_score"] = val.get("correct_written_answers", 0) / tw
            tv = val.get("total_vision_questions", 0)
            if tv > 0:
                res["vision_score"] = val.get("correct_vision_answers", 0) / tv
            if res:
                results[task_id] = res
    except Exception:
        pass
    return results

def load_scienceagentbench_subtask_scores_from_file(eval_path: Path, task_ids: List[str]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if not eval_path.exists():
        return results
    ordered_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
    try:
        lines = eval_path.read_text(encoding="utf-8").splitlines()
        for idx, task_id in enumerate(ordered_ids):
            if idx >= len(lines):
                continue
            line = lines[idx].strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                results[task_id] = {
                    "success_rate": float(payload.get("success_rate", 0)),
                    "valid_program": float(payload.get("valid_program", 0)),
                    "codebert_score": float(payload.get("codebert_score", 0)),
                }
    except Exception:
        pass
    return results

def load_scicode_success(run_dir: Path, run_id: str) -> Optional[set]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    if not upload_path.exists():
        return None
    data = json.loads(upload_path.read_text(encoding="utf-8"))
    successful = data.get("results", {}).get("successful_tasks")
    if isinstance(successful, list):
        return {str(t) for t in successful}
    return None

def reeval_scicode(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    from hal.benchmarks.scicode import SciCodeBenchmark

    env_old = _push_env("SCICODE_DATASET_PATH", str(dataset_path))
    image_old = os.environ.get("SCICODE_EVAL_IMAGE")
    skip_old = os.environ.get("SCICODE_EVAL_SKIP_INSTALL")
    used_cached = False
    if not image_old and docker_image_exists(SCICODE_CACHED_IMAGE):
        os.environ["SCICODE_EVAL_IMAGE"] = SCICODE_CACHED_IMAGE
        if skip_old is None:
            os.environ["SCICODE_EVAL_SKIP_INSTALL"] = "1"
        used_cached = True
    try:
        bench = SciCodeBenchmark(agent_dir=".", config={}, benchmark_name="scicode")
        eval_results = bench.evaluate_output(ok_tasks, run_id="reeval_scicode")
    finally:
        _push_env("SCICODE_DATASET_PATH", env_old)
        if used_cached:
            _push_env("SCICODE_EVAL_IMAGE", image_old)
            _push_env("SCICODE_EVAL_SKIP_INSTALL", skip_old)

    details = eval_results.get("details", {})
    results: Dict[str, int] = {}
    for task_id in ok_tasks.keys():
        task = bench.benchmark.get(task_id)
        if not task:
            continue
        total = len(task.get("sub_steps", []))
        passed = len(details.get(task_id, []))
        results[task_id] = 1 if total > 0 and passed == total else 0
    return results

def reeval_corebench(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    from hal.benchmarks.corebench import CoreBenchHard

    env_old = _push_env("HAL_COREBENCH_DATASET_PATH", str(dataset_path))
    try:
        bench = CoreBenchHard(agent_dir=".", config={})
        eval_results = bench.evaluate_output(ok_tasks, run_id="reeval_corebench")
    finally:
        _push_env("HAL_COREBENCH_DATASET_PATH", env_old)

    results: Dict[str, int] = {}
    for task_id, result in eval_results.items():
        written_correct = result.get("correct_written_answers", 0)
        vision_correct = result.get("correct_vision_answers", 0)
        written_total = result.get("total_written_questions", 0)
        vision_total = result.get("total_vision_questions", 0)
        if written_total == 0 and vision_total == 0:
            results[task_id] = 0
            continue
        results[task_id] = 1 if (written_correct == written_total and vision_correct == vision_total) else 0
    return results

def reeval_colbench(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    from hal.benchmarks.sweet_rl.utils import code_evaluate

    sorted_ids = sorted(ok_tasks.keys(), key=lambda x: int(x) if x.isdigit() else x)
    trajectories = [ok_tasks[task_id] for task_id in sorted_ids]
    correctness = code_evaluate(trajectories)
    results: Dict[str, int] = {}
    for idx, task_id in enumerate(sorted_ids):
        score = correctness[idx] if idx < len(correctness) else 0
        results[task_id] = 1 if score >= 0.999 else 0
    return results

def reeval_scienceagentbench(dataset_path: Path, ok_tasks: Dict[str, object]) -> Dict[str, int]:
    # Mirror ScienceAgentBench imports/path setup
    repo_root = Path(__file__).resolve().parents[1]
    hal_harness = repo_root / "hal-harness"
    submodule_path = hal_harness / "hal" / "benchmarks" / "scienceagentbench" / "ScienceAgentBench_modified"
    if str(submodule_path) not in sys.path:
        sys.path.insert(0, str(submodule_path))

    from evaluation.harness import run_evaluation as sab_eval

    # Load dataset mapping
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    id_to_gold = {str(row["instance_id"]): row["gold_program_name"] for row in data}

    tmp_dir = Path(os.environ.get("HAL_TMP_DIR", "/tmp")) / f"sab_reeval_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pred_programs = tmp_dir / "pred_programs"
    pred_programs.mkdir(parents=True, exist_ok=True)

    instance_ids: List[str] = []
    for task_id, payload in ok_tasks.items():
        code = _extract_python_code(payload) if isinstance(payload, dict) else None
        if not code:
            continue
        gold_name = id_to_gold.get(str(task_id))
        if not gold_name:
            continue
        out_path = pred_programs / f"pred_{gold_name}"
        out_path.write_text(code, encoding="utf-8")
        instance_ids.append(str(task_id))

    if not instance_ids:
        return {}

    log_fname = tmp_dir / "sab_eval.jsonl"

    # Point eval logs into tmp
    sab_eval.RUN_EVALUATION_LOG_DIR = tmp_dir / "logs"

    env_old = _push_env("SCIENCEAGENTBENCH_DATASET_PATH", str(dataset_path))
    try:
        sab_eval.main(
            benchmark_path=str(submodule_path / "benchmark"),
            pred_program_path=str(pred_programs),
            log_fname=str(log_fname),
            dataset_name="osunlp/ScienceAgentBench",
            split="validation",
            instance_ids=instance_ids,
            max_workers=max(1, (os.cpu_count() or 2) // 2),
            force_rebuild=False,
            cache_level="instance",
            clean=False,
            open_file_limit=4096,
            run_id=f"reeval_{os.getpid()}",
            timeout=1800,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY", ""),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        )
    finally:
        _push_env("SCIENCEAGENTBENCH_DATASET_PATH", env_old)

    results: Dict[str, int] = {}
    if log_fname.exists():
        lines = log_fname.read_text(encoding="utf-8").splitlines()
        ordered_ids = sorted(instance_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
        for idx, task_id in enumerate(ordered_ids):
            if idx >= len(lines):
                continue
            line = lines[idx].strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                success_rate = payload.get("success_rate")
                if isinstance(success_rate, (int, float)):
                    results[task_id] = 1 if success_rate >= 1.0 else 0
    return results

def load_corebench_success(run_dir: Path, run_id: str) -> Optional[set]:
    return load_scicode_success(run_dir, run_id)

def load_colbench_correctness(run_dir: Path, run_id: str) -> Optional[List[float]]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    if not upload_path.exists():
        return None
    data = json.loads(upload_path.read_text(encoding="utf-8"))
    correctness = data.get("raw_eval_results")
    if isinstance(correctness, list):
        return [float(v) for v in correctness]
    return None

def load_scienceagentbench_eval(run_dir: Path, run_id: str, task_ids: List[str]) -> Dict[str, float]:
    eval_path = run_dir / f"{run_id}_eval.jsonl"
    results: Dict[str, float] = {}
    if not eval_path.exists():
        return results
    ordered_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    for idx, task_id in enumerate(ordered_ids):
        if idx >= len(lines):
            continue
        line = lines[idx].strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            success_rate = payload.get("success_rate")
            if isinstance(success_rate, (int, float)):
                results[task_id] = float(success_rate)
    return results

def load_scicode_subtask_scores(run_dir: Path, run_id: str, task_meta: Dict[str, List[str]]) -> Dict[str, float]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    results: Dict[str, float] = {}
    if not upload_path.exists():
        return results
    try:
        data = json.loads(upload_path.read_text(encoding="utf-8"))
        details = data.get("raw_eval_results", {}).get("details", {})
        for task_id, all_sub in task_meta.items():
            if not all_sub:
                continue
            passed_sub = details.get(task_id, [])
            results[task_id] = len(passed_sub) / len(all_sub)
    except Exception:
        pass
    return results

def load_corebench_subtask_scores(run_dir: Path, run_id: str) -> Dict[str, Dict[str, float]]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    results: Dict[str, Dict[str, float]] = {}
    if not upload_path.exists():
        return results
    try:
        data = json.loads(upload_path.read_text(encoding="utf-8"))
        raw = data.get("raw_eval_results", {})
        for task_id, val in raw.items():
            res = {}
            tw = val.get("total_written_questions", 0)
            if tw > 0:
                res["written_score"] = val.get("correct_written_answers", 0) / tw
            tv = val.get("total_vision_questions", 0)
            if tv > 0:
                res["vision_score"] = val.get("correct_vision_answers", 0) / tv
            if res:
                results[task_id] = res
    except Exception:
        pass
    return results

def load_scienceagentbench_subtask_scores(run_dir: Path, run_id: str, task_ids: List[str]) -> Dict[str, Dict[str, float]]:
    eval_path = run_dir / f"{run_id}_eval.jsonl"
    results: Dict[str, Dict[str, float]] = {}
    if not eval_path.exists():
        return results
    ordered_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else float("inf"))
    try:
        lines = eval_path.read_text(encoding="utf-8").splitlines()
        for idx, task_id in enumerate(ordered_ids):
            if idx >= len(lines):
                continue
            line = lines[idx].strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                results[task_id] = {
                    "success_rate": float(payload.get("success_rate", 0)),
                    "valid_program": float(payload.get("valid_program", 0)),
                    "codebert_score": float(payload.get("codebert_score", 0)),
                }
    except Exception:
        pass
    return results

def load_judge_verdicts(judge_dir: Path, benchmark: str, run_id_prefix: str) -> Dict[str, int]:
    """
    Load binary verdicts (final_grade) from judge_output for a specific benchmark and prefix.
    Expects formats like: 
    - [benchmark]_verdict_[prefix].csv
    - [benchmark]_[prefix]_verdict.csv
    - [prefix]_verdict.csv
    """
    verdicts: Dict[str, int] = {}
    # Clean up prefix for filename matching (remove regex chars)
    clean_prefix = re.sub(r'[^a-zA-Z0-9_]', '', run_id_prefix).rstrip("_")
    
    # Map internal benchmark keys to descriptive names used by judge
    full_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
    
    # List of candidate filenames to try
    candidates = [
        f"{full_name}_verdict_{clean_prefix}.csv",
        f"{full_name}_{clean_prefix}_verdict.csv",
        f"{benchmark}_verdict_{clean_prefix}.csv",
        f"{benchmark}_{clean_prefix}_verdict.csv",
        f"{clean_prefix}_verdict.csv",
        f"{full_name}_verdict.csv",
        f"{benchmark}_verdict.csv",
    ]
    
    for fname in sorted(set(candidates), key=len, reverse=True):
        verdict_path = judge_dir / fname
        if verdict_path.exists():
            try:
                with verdict_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get("task_id")
                        grade = row.get("final_grade")
                        if tid is not None and grade is not None:
                            try:
                                verdicts[str(tid)] = int(float(grade))
                            except ValueError:
                                pass
                if verdicts:
                    # print(f"  Loaded judge verdicts from {fname}")
                    return verdicts
            except Exception as e:
                print(f"Warning: Failed to load judge verdict from {verdict_path}: {e}")
                
    return verdicts

def load_rubric_scores(rubric_dir: Path, benchmark: str, run_id: str) -> Dict[str, float]:
    """
    Load rubric scores from eval_traces/rubrics_output/[benchmark]/[full_name]_[run_id]_UPLOAD.csv
    Returns: Dict[task_id, grade]
    """
    scores: Dict[str, float] = {}
    full_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
    
    # List of candidate paths to try
    candidate_paths = [
        rubric_dir / benchmark / f"{full_name}_{run_id}_UPLOAD.csv",
        rubric_dir / benchmark / f"{benchmark}_{run_id}_UPLOAD.csv",
        rubric_dir / benchmark / f"{full_name}_{run_id}.csv",
        rubric_dir / benchmark / f"{benchmark}_{run_id}.csv",
        rubric_dir / benchmark / f"{run_id}_UPLOAD.csv",
        rubric_dir / benchmark / f"{run_id}.csv",
    ]
    
    for rubric_path in candidate_paths:
        if rubric_path.exists():
            try:
                with rubric_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get("task_id")
                        grade = row.get("grade")
                        if tid is not None and grade is not None:
                            try:
                                scores[str(tid)] = float(grade)
                            except ValueError:
                                pass
                if scores:
                    return scores
            except Exception as e:
                print(f"Warning: Failed to load rubric from {rubric_path}: {e}")
            
    return scores

def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> Path:
    import csv
    parent = path.parent
    try:
        if parent.exists():
            if parent.is_dir():
                pass
            elif parent.is_symlink():
                target = Path(os.path.realpath(parent))
                target.mkdir(parents=True, exist_ok=True)
            else:
                raise RuntimeError(f"Output parent is not a directory: {parent}")
        else:
            if parent.is_symlink():
                target = Path(os.path.realpath(parent))
                target.mkdir(parents=True, exist_ok=True)
            else:
                parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = REPO_ROOT / "output_local" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        path = fallback
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path

# Global cache for benchmark task metadata to avoid redundant loading/HF hangs
BENCHMARK_TASK_META_CACHE: Dict[str, Dict[str, List[str]]] = {}

def main() -> None:
    parser = argparse.ArgumentParser(description="Build binary response matrix for a given prefix.")
    parser.add_argument("--prefix", required=True, help="Prefix pattern (string or regex) like beach[0-9]+_")
    parser.add_argument("--log-dir", help="Path to benchmark_run_* directory")
    parser.add_argument("--run-root", help="Override run root path")
    parser.add_argument("--output", help="CSV output path")
    parser.add_argument("--original", action="store_true", help="Treat as original (pre-revision) data")
    parser.add_argument("--reeval", action="store_true", help="Re-evaluate tasks from raw submissions")
    parser.add_argument("--extract-subscores", action="store_true", help="Extract and save detailed subtask scores")
    parser.add_argument("--traces-dir", help="Directory containing aggregated traces (e.g., eval_traces)")
    parser.add_argument("--benchmark", action="append", help="Select specific benchmark(s) to process.")
    parser.add_argument("--skip-benchmark", action="append", default=["scicode", "colbench", "corebench"], help="Skip benchmark(s) by name")
    parser.add_argument("--dataset-scicode", help="Path to SciCode dataset json")
    parser.add_argument("--dataset-scienceagentbench", help="Path to ScienceAgentBench dataset json")
    parser.add_argument("--dataset-corebench", help="Path to CoreBench dataset json")
    parser.add_argument("--dataset-colbench", help="Path to ColBench dataset jsonl")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    repo_root = script_dir
    run_root = Path(args.run_root) if args.run_root else detect_run_root(script_dir)

    logs_base = run_root / "logs"
    if not logs_base.exists():
        for d in [".hal_data", ".hal-data"]:
            cand = run_root / d / "logs"
            if cand.exists():
                logs_base = cand
                break

    if args.benchmark:
        # Map descriptive names back to internal keys
        requested = []
        for b in args.benchmark:
            if b == "colbench_backend_programming":
                requested.append("colbench")
            elif b == "corebench_hard":
                requested.append("corebench")
            else:
                requested.append(b)
        active_benchmarks = [b for b in BENCHMARKS if b in requested]
    else:
        skip_set = set(args.skip_benchmark)
        active_benchmarks = [b for b in BENCHMARKS if b not in skip_set]

    if args.extract_subscores:
        for b in active_benchmarks:
            if b not in BENCHMARKS:
                print(f"Error: --extract-subscores not implemented for {b}")
                sys.exit(1)

    # 1. Collect all runs across all active benchmarks and group by their ACTUAL prefix
    prefix_to_benchmark_runs: Dict[str, Dict[str, List[str]]] = {}

    for benchmark in active_benchmarks:
        # Find runs and their actual matching prefix
        pairs = find_run_ids_from_results(run_root, repo_root, benchmark, args.prefix, traces_dir=args.traces_dir)
        for rid, actual_pfx in pairs:
            if actual_pfx not in prefix_to_benchmark_runs:
                prefix_to_benchmark_runs[actual_pfx] = {b: [] for b in active_benchmarks}
            prefix_to_benchmark_runs[actual_pfx][benchmark].append(rid)

    if not prefix_to_benchmark_runs:
        print(f"No runs found matching prefix pattern: {args.prefix}")
        sys.exit(0)

    # 2. Iterate over each unique actual prefix
    for actual_prefix, benchmark_run_ids in sorted(prefix_to_benchmark_runs.items()):
        print(f"\nProcessing prefix: {actual_prefix}")

        # Build task columns and totals for THIS prefix
        benchmark_task_ids: Dict[str, List[str]] = {}
        benchmark_task_meta: Dict[str, Optional[Dict[str, List[str]]]] = {}

        # Find log_dir for THIS actual prefix
        current_log_dir = Path(args.log_dir) if args.log_dir else find_log_dir_for_prefix(logs_base, actual_prefix)

        for benchmark in active_benchmarks:
            run_ids = benchmark_run_ids[benchmark]
            if not run_ids:
                continue

            text = ""
            if current_log_dir:
                log_file = current_log_dir / f"{benchmark}.log"
                if log_file.exists():
                    text = log_file.read_text(errors="ignore")

            # Supplement run IDs from log if available
            if text:
                log_run_ids = [rid for rid in parse_run_ids(text) if actual_prefix in rid]
                run_ids = sorted(set(run_ids + log_run_ids))

            # If we found run IDs in results but didn't have log_dir, try to find one run's log for dataset path discovery
            if not text and run_ids:
                for rid in run_ids:
                    r_dir = resolve_run_dir(run_root, repo_root, benchmark, rid)
                    if r_dir:
                        for log_name in [f"{rid}.log", "run.log", f"{benchmark}.log"]:
                            log_path = r_dir / log_name
                            if log_path.exists():
                                text = log_path.read_text(errors="ignore")
                                break
                    if text:
                        break

            dataset_path = resolve_dataset_path(benchmark, text, run_root, args, prefix=actual_prefix)
            if dataset_path and dataset_path.exists():
                benchmark_task_ids[benchmark] = load_task_ids(benchmark, dataset_path)
                if benchmark == "scicode":
                    benchmark_task_meta[benchmark] = build_scicode_task_meta(dataset_path)
                else:
                    benchmark_task_meta[benchmark] = None
            else:
                # Fallback: try to collect all task IDs seen in raw submissions
                inferred_ids = set()
                for rid in run_ids:
                    if args.traces_dir:
                        base_dir = Path(args.traces_dir)
                        raw_dir = base_dir / "raw_submission"
                        full_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
                        raw_path = raw_dir / f"{full_name}_{rid}_RAW_SUBMISSIONS.jsonl"
                        if not raw_path.exists():
                            raw_path = raw_dir / f"{benchmark}_{rid}_RAW_SUBMISSIONS.jsonl"
                    else:
                        r_dir = resolve_run_dir(run_root, repo_root, benchmark, rid)
                        if not r_dir:
                            continue
                        raw_path = r_dir / f"{rid}_RAW_SUBMISSIONS.jsonl"

                    if raw_path.exists():
                        with raw_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    obj = json.loads(line)
                                    if obj and isinstance(obj, dict):
                                        inferred_ids.add(str(next(iter(obj.keys()))))
                                except Exception:
                                    pass

                # Special fallback for colbench: if no tasks inferred from RAW_SUBMISSIONS,
                # check one UPLOAD.json to see how many results it has
                if not inferred_ids and benchmark == "colbench" and run_ids:
                    rid = run_ids[0]
                    if args.traces_dir:
                        upload_path = Path(args.traces_dir) / "traces" / f"colbench_{rid}_UPLOAD.json"
                        if not upload_path.exists():
                            upload_path = Path(args.traces_dir) / "traces" / f"colbench_backend_programming_{rid}_UPLOAD.json"
                    else:
                        r_dir = resolve_run_dir(run_root, repo_root, benchmark, rid)
                        upload_path = r_dir / f"{rid}_UPLOAD.json" if r_dir else None

                    if upload_path and upload_path.exists():
                        try:
                            data = json.loads(upload_path.read_text(encoding="utf-8"))
                            raw_results = data.get("raw_eval_results")
                            if isinstance(raw_results, list):
                                inferred_ids = {str(i) for i in range(len(raw_results))}
                        except Exception:
                            pass

                if inferred_ids:
                    # Sort numerically if possible, else alphabetically
                    benchmark_task_ids[benchmark] = sorted(list(inferred_ids), key=lambda x: int(x) if x.isdigit() else x)
                    
                    if benchmark == "scicode":
                        if "scicode" in BENCHMARK_TASK_META_CACHE:
                            benchmark_task_meta[benchmark] = BENCHMARK_TASK_META_CACHE["scicode"]
                        else:
                            try:
                                print(f"  [{actual_prefix}] Loading SciCode metadata (may take a moment)...")
                                # Try to find a local scicode dataset first to avoid HF hang
                                local_ds = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "scicode" / "scicode.json"
                                # Search results directory for ANY local metadata for this prefix
                                if not local_ds.exists():
                                    results_dir = run_root / "results" / "scicode"
                                    if results_dir.exists():
                                        found = [f for f in results_dir.glob(f"**/*{actual_prefix}*.json") if "_modified_" in f.name]
                                        if found:
                                            local_ds = found[0]

                                if not local_ds.exists():
                                    # Try common naming patterns in repo
                                    candidates = list((REPO_ROOT / "hal-harness").glob("**/scicode*.json"))
                                    if candidates:
                                        local_ds = candidates[0]

                                from hal.benchmarks.scicode import SciCodeBenchmark
                                # Temporarily point to local if found
                                if local_ds.exists():
                                    old_env = os.environ.get("SCICODE_DATASET_PATH")
                                    os.environ["SCICODE_DATASET_PATH"] = str(local_ds)
                                    bench = SciCodeBenchmark(agent_dir=".", config={})
                                    if old_env:
                                        os.environ["SCICODE_DATASET_PATH"] = old_env
                                    else:
                                        os.environ.pop("SCICODE_DATASET_PATH", None)
                                else:
                                    # Fallback to default (may still trigger HF if local not found)
                                    bench = SciCodeBenchmark(agent_dir=".", config={})

                                # Use the default benchmark meta to get sub_steps
                                all_meta = {}
                                for task_id in benchmark_task_ids[benchmark]:
                                    task = bench.benchmark.get(task_id)
                                    if task:
                                        sub_steps = task.get("sub_steps", [])
                                        all_meta[task_id] = [f"{task_id}.{idx + 1}" for idx in range(len(sub_steps))]
                                benchmark_task_meta[benchmark] = all_meta
                                BENCHMARK_TASK_META_CACHE["scicode"] = all_meta
                                print(f"  [{actual_prefix}] SciCode metadata loaded and cached.")
                            except Exception as e:
                                print(f"Failed to load SciCode metadata for {actual_prefix}: {e}")
                                benchmark_task_meta[benchmark] = None
                    else:
                        benchmark_task_meta[benchmark] = None
                else:
                    continue

        # Precise mapping of subscores to their respective benchmarks
        BENCHMARK_TO_SUBSCORES = {
            "scicode": ["subtask_score", "rubric_score"],
            "scienceagentbench": ["success_rate", "valid_program", "codebert_score", "rubric_score"],
            "corebench": ["written_score", "vision_score", "rubric_score"],
            "colbench": ["raw_score", "rubric_score"],
            "assistantbench": ["rubric_score"],
            "swebench_verified_mini": ["rubric_score"],
            "usaco": ["rubric_score"]
        }
        # Flatten for initialization
        ALL_SUBSCORE_NAMES = sorted(list(set(s for scores in BENCHMARK_TO_SUBSCORES.values() for s in scores)))

        # Load Judge Verdicts and Rubrics for this prefix (to export separately later and fallback task IDs)
        judge_dir = REPO_ROOT / "eval_traces" / "judge_output"
        rubric_dir = REPO_ROOT / "eval_traces" / "rubrics_output"
        benchmark_to_verdicts: Dict[str, Dict[str, int]] = {}
        
        for benchmark in active_benchmarks:
            # If --original, we want to look for [benchmark]_verdict.csv directly
            # because judge.py saves them that way when run with --original
            pfx_to_load = actual_prefix
            if args.original:
                pfx_to_load = "" # This will trigger the fallback to [benchmark]_verdict.csv
            
            v_map = load_judge_verdicts(judge_dir, benchmark, pfx_to_load)
            benchmark_to_verdicts[benchmark] = v_map

            # FALLBACK: If standard discovery failed to find task IDs, pull them from verdicts and rubrics
            if not benchmark_task_ids.get(benchmark):
                found_ids = set(v_map.keys())
                
                # Also check rubrics for any other task IDs
                run_ids = benchmark_run_ids.get(benchmark, [])
                for rid in run_ids:
                    r_scores = load_rubric_scores(rubric_dir, benchmark, rid)
                    found_ids.update(r_scores.keys())
                
                if found_ids:
                    benchmark_task_ids[benchmark] = sorted(list(found_ids), key=lambda x: int(x) if x.isdigit() else x)

        columns: List[str] = []
        for benchmark in active_benchmarks:
            # Use full benchmark names for columns
            col_bench_name = benchmark
            if benchmark == "colbench":
                col_bench_name = "colbench_backend_programming"
            elif benchmark == "corebench":
                col_bench_name = "corebench_hard"

            task_ids = benchmark_task_ids.get(benchmark, [])
            for task_id in task_ids:
                columns.append(f"{col_bench_name}.{task_id}")

        # Build rows
        rows: List[List[str]] = []
        # Map from subscore_name -> list of rows (one per agent/run)
        # Each row has the same length as columns
        subscore_type_to_rows: Dict[str, List[List[str]]] = {}

        if args.extract_subscores:
            for name in ALL_SUBSCORE_NAMES:
                subscore_type_to_rows[name] = []

        row_labels: List[str] = []

        for benchmark in active_benchmarks:
            # Use actual runs for this benchmark and prefix
            run_ids = benchmark_run_ids[benchmark]
            text = ""
            if current_log_dir:
                log_file = current_log_dir / f"{benchmark}.log"
                if log_file.exists():
                    text = log_file.read_text(errors="ignore")

            # Supplement run IDs from log if available
            if text:
                log_run_ids = [rid for rid in parse_run_ids(text) if actual_prefix in rid]
                run_ids = sorted(set(run_ids + log_run_ids))

            # If we found run IDs in results but didn't have log_dir, try to find one run's log for re-evaluation dataset path discovery
            if args.reeval and not text and run_ids:
                for rid in run_ids:
                    r_dir = resolve_run_dir(run_root, repo_root, benchmark, rid)
                    if r_dir:
                        for log_name in [f"{rid}.log", "run.log", f"{benchmark}.log"]:
                            log_path = r_dir / log_name
                            if log_path.exists():
                                text = log_path.read_text(errors="ignore")
                                break
                    if text:
                        break

            # Deduplicate runs: keep the one with the most completed tasks (least NaNs)
            best_runs: Dict[str, Tuple[str, int]] = {}
            task_meta = benchmark_task_meta.get(benchmark)
            for rid in run_ids:
                c_key = derive_config_key(rid, benchmark, actual_prefix)
                # Check how many tasks are completed
                r_dir = resolve_run_dir(run_root, repo_root, benchmark, rid, traces_dir=args.traces_dir)
                count = 0
                if r_dir:
                    full_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
                    raw_path = r_dir / f"{full_name}_{rid}_RAW_SUBMISSIONS.jsonl"
                    if not raw_path.exists():
                        raw_path = r_dir / f"{benchmark}_{rid}_RAW_SUBMISSIONS.jsonl"
                    if not raw_path.exists():
                        raw_path = r_dir / f"{rid}_RAW_SUBMISSIONS.jsonl"
                    
                    if raw_path.exists():
                        ok = load_raw_ok_tasks(raw_path, benchmark, task_meta)
                        count = len(ok)
                    else:
                        # Fallback: if UPLOAD exists, treat as valid run with unknown count for now
                        upload_path = r_dir / f"{full_name}_{rid}_UPLOAD.json"
                        if not upload_path.exists():
                            upload_path = r_dir / f"{benchmark}_{rid}_UPLOAD.json"
                        if not upload_path.exists():
                            upload_path = r_dir / f"{rid}_UPLOAD.json"
                        if upload_path.exists():
                            count = 1 # Mark as having something
                
                current_best = best_runs.get(c_key)
                if current_best is None:
                    best_runs[c_key] = (rid, count)
                else:
                    best_id, best_count = current_best
                    # Prefer more completed tasks; break ties with newer run_id
                    if count > best_count or (count == best_count and rid > best_id):
                        best_runs[c_key] = (rid, count)

            latest_runs = {k: v[0] for k, v in best_runs.items()}
            for config_key, run_id in sorted(latest_runs.items()):
                row_label = f"{benchmark}.{config_key}"
                if args.traces_dir:
                    base_dir = Path(args.traces_dir)
                    run_dir = base_dir / "traces"
                    if not run_dir.exists():
                        run_dir = base_dir
                    raw_dir = base_dir / "raw_submission"
                    if not raw_dir.exists():
                        raw_dir = base_dir
                    full_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
                    # Try full name first, then fallback to short name
                    upload_path = run_dir / f"{full_name}_{run_id}_UPLOAD.json"
                    if not upload_path.exists():
                        upload_path = run_dir / f"{benchmark}_{run_id}_UPLOAD.json"
                    if not upload_path.exists():
                        upload_path = run_dir / f"{run_id}_UPLOAD.json"
                        
                    raw_path = raw_dir / f"{full_name}_{run_id}_RAW_SUBMISSIONS.jsonl"
                    if not raw_path.exists():
                        raw_path = raw_dir / f"{benchmark}_{run_id}_RAW_SUBMISSIONS.jsonl"
                    if not raw_path.exists():
                        raw_path = raw_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
                        
                    eval_path = raw_dir / f"{full_name}_{run_id}_eval.jsonl"
                    if not eval_path.exists():
                        eval_path = raw_dir / f"{benchmark}_{run_id}_eval.jsonl"
                    if not eval_path.exists():
                        eval_path = raw_dir / f"{run_id}_eval.jsonl"
                else:
                    run_dir = resolve_run_dir(run_root, repo_root, benchmark, run_id)
                    if not run_dir:
                        continue
                    upload_path = run_dir / f"{run_id}_UPLOAD.json"
                    raw_path = run_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
                    eval_path = run_dir / f"{run_id}_eval.jsonl"

                if not upload_path.exists():
                    continue

                task_meta = benchmark_task_meta.get(benchmark)
                ok_tasks = load_raw_ok_tasks(raw_path, benchmark, task_meta)

                success_map: Dict[str, int] = {}
                if args.reeval:
                    dataset_path = resolve_dataset_path(benchmark, text, run_root, args, prefix=actual_prefix)
                    if not dataset_path or not dataset_path.exists():
                        print(f"Missing dataset path for {benchmark}; cannot re-evaluate")
                        sys.exit(1)
                    if benchmark == "scicode":
                        success_map = reeval_scicode(dataset_path, ok_tasks)
                    elif benchmark == "corebench":
                        success_map = reeval_corebench(dataset_path, ok_tasks)
                    elif benchmark == "colbench":
                        success_map = reeval_colbench(dataset_path, ok_tasks)
                    elif benchmark == "scienceagentbench":
                        success_map = reeval_scienceagentbench(dataset_path, ok_tasks)
                else:
                    success_set = None
                    colbench_scores = None
                    sab_scores: Dict[str, float] = {}
                    if benchmark == "scicode":
                        success_set = load_upload_success(upload_path)
                    elif benchmark == "corebench":
                        success_set = load_upload_success(upload_path)
                    elif benchmark == "colbench":
                        colbench_scores = load_colbench_correctness_from_file(upload_path)
                    elif benchmark == "scienceagentbench":
                        sab_scores = load_scienceagentbench_eval_from_file(eval_path, benchmark_task_ids[benchmark])

              # --- Gather Data ---
    columns = []
    # Identify which parts of the `columns` list belong to which benchmark
    # Use mapping: b -> (start_idx, end_idx)
    col_bounds = {}
    current_idx = 0
    
    for b in active_benchmarks:
        start = current_idx
        for task_id in benchmark_task_ids.get(b, []):
            columns.append(f"{b}.{task_id}")
            current_idx += 1
        col_bounds[b] = (start, current_idx)

    # Note: `rows` contains aligned arrays of length `len(columns)`
    # The first element of `header` is "task_id", followed by exactly `columns`f b != benchmark:
                            row.append("")
                            continue
                        if task_id not in ok_tasks:
                            row.append("")
                            continue
                        if args.reeval:
                            if task_id in success_map:
                                row.append("1" if success_map[task_id] == 1 else "0")
                            else:
                                row.append("")
                        else:
                            if benchmark in ("scicode", "corebench"):
                                if success_set is None:
                                    row.append("")
                                else:
                                    row.append("1" if task_id in success_set else "0")
                            elif benchmark == "colbench":
                                if colbench_scores is None:
                                    row.append("")
                                else:
                                    idx = int(task_id)
                                    if idx < len(colbench_scores):
                                        row.append("1" if colbench_scores[idx] >= 0.999 else "0")
                                    else:
                                        row.append("")
                            else:
                                if task_id in sab_scores:
                                    row.append("1" if sab_scores[task_id] >= 1.0 else "0")
                                else:
                                    row.append("")
                rows.append(row)
                row_labels.append(row_label)

                if args.extract_subscores:
                    sub_success = {}
                    if benchmark == "scicode":
                        meta = benchmark_task_meta.get(benchmark, {})
                        scicode_avg = load_scicode_subtask_scores_from_file(upload_path, meta)
                        for tid, s in scicode_avg.items():
                            sub_success[tid] = {"subtask_score": s}
                    elif benchmark == "corebench":
                        sub_success = load_corebench_subtask_scores_from_file(upload_path)
                    elif benchmark == "colbench":
                        scores = load_colbench_correctness_from_file(upload_path)
                        if scores:
                            for idx, s in enumerate(scores):
                                sub_success[str(idx)] = {"raw_score": s}
                    elif benchmark == "scienceagentbench":
                        sub_success = load_scienceagentbench_subtask_scores_from_file(eval_path, benchmark_task_ids[benchmark])

                    # Load rubric scores for ANY benchmark
                    rubric_dir = REPO_ROOT / "result" / ".hal_data" / "rubrics_output"
                    rubric_grades = load_rubric_scores(rubric_dir, benchmark, run_id)
                    if rubric_grades:
                        for tid, grade in rubric_grades.items():
                            if tid not in sub_success: sub_success[tid] = {}
                            sub_success[tid]["rubric_score"] = grade

                    # For each subscore type, we build a row that matches the 'columns' structure
                    for s_name in ALL_SUBSCORE_NAMES:
                        current_sub_row = []
                        for b in active_benchmarks:
                            tids = benchmark_task_ids.get(b, [])
                            # If this score name doesn't belong to this benchmark, fill with empty
                            if s_name not in BENCHMARK_TO_SUBSCORES.get(b, []):
                                current_sub_row.extend([""] * len(tids))
                                continue
                            # If this row belongs to a different benchmark than the one we just processed
                            if b != benchmark:
                                current_sub_row.extend([""] * len(tids))
                                continue
                            # Extract data strictly based on the mapping
                            for tid in tids:
                                task_scores = sub_success.get(tid, {})
                                if s_name == "subtask_score" and b == "scicode":
                                    val = task_scores.get("subtask_score")
                                    current_sub_row.append(str(val) if val is not None else "")
                                elif s_name == "raw_score" and b == "colbench":
                                    val = task_scores.get("raw_score")
                                    current_sub_row.append(str(val) if val is not None else "")
                                elif b == "scienceagentbench" and s_name in ("success_rate", "valid_program", "codebert_score"):
                                    val = task_scores.get(s_name)
                                    current_sub_row.append(str(val) if val is not None else "")
                                elif b == "corebench" and s_name in ("written_score", "vision_score"):
                                    val = task_scores.get(s_name)
                                    current_sub_row.append(str(val) if val is not None else "")
                                elif s_name == "rubric_score":
                                    val = task_scores.get("rubric_score")
                                    current_sub_row.append(str(val) if val is not None else "")
                                else:
                                    current_sub_row.append("")
                        subscore_type_to_rows[s_name].append(current_sub_row)

        # Output generation
        header = ["agent"] + columns
        # Remove trailing underscore from actual_prefix for filename
        pfx_for_file = actual_prefix.rstrip("_")
        if args.original:
            pfx_for_file = "original"
            
        # Determine subfolder based on benchmarks found for this prefix
        contributing_benchmarks = [b for b, rids in benchmark_run_ids.items() if rids]
        # --- Write Data Per Benchmark ---
        for b in active_benchmarks:
            if b == "colbench":
                subfolder = "colbench_backend_programming"
            elif b == "corebench":
                subfolder = "corebench_hard"
            else:
                subfolder = b
                
            if args.output:
                b_output_dir = Path(args.output) / subfolder / "resmat"
            else:
                if args.original:
                    b_output_dir = REPO_ROOT / "eval_response_matrix" / "pre-revision" / subfolder / "resmat"
                else:
                    b_output_dir = repo_root / "result" / ".hal_data" / subfolder / "resmat"
            
            b_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Slice header and rows using pre-computed bounds
            start, end = col_bounds[b]
            if start == end: continue # Nothing to write
            
            b_header = ["task_id"] + columns[start:end]
            # Strip prefix internally explicitly for original datasets if we want cleanly scoped headers
            if args.original:
                b_header_stripped = ["task_id"] + [c.split('.', 1)[1] if '.' in c else c for c in b_header[1:]]
            else:
                b_header_stripped = b_header
                
            b_rows = []
            for i in range(len(rows)):
                # Each row in `rows` aligns perfectly with `columns`. Slice it.
                row_slice = rows[i][start:end]
                # Only write row if the agent actually participated in this benchmark
                # (has at least one valid entry)
                if any(val != "" for val in row_slice):
                    b_rows.append([row_labels[i]] + row_slice)
            
            if not b_rows and columns:
                if args.original:
                    # Original requires placeholder
                    b_rows.append(["original_agent"] + [""] * (end - start))
                else:
                    continue # Empty matrix, ignore.
                    
            out_path = b_output_dir / f"resmat_{pfx_for_file}.csv"
            write_csv(out_path, b_header_stripped, b_rows)
            print(f"Wrote CSV: {out_path} with {len(b_rows)} agent rows")
            
            # Export verdicts
            has_b_verdicts = benchmark_to_verdicts.get(b)
            if has_b_verdicts:
                verdicts_dir = b_output_dir.parent / "verdicts"
                verdicts_dir.mkdir(parents=True, exist_ok=True)
                v_path = verdicts_dir / f"verdict_{pfx_for_file}.csv"
                
                judge_row = ["judge_verdict"]
                for task_id in benchmark_task_ids.get(b, []):
                    val = has_b_verdicts.get(task_id)
                    judge_row.append(str(val) if val is not None else "")
                
                write_csv(v_path, b_header_stripped, [judge_row])
                print(f"Wrote Verdict CSV: {v_path}")
                
            # Export subscores
            if args.extract_subscores:
                for s_name, s_rows in subscore_type_to_rows.items():
                    # Check if this benchmark has meaningful subscores
                    b_sub_rows = []
                    for idx in range(len(row_labels)):
                        sliced_s_row = s_rows[idx][start:end]
                        if any(val != "" for val in sliced_s_row):
                            b_sub_rows.append([row_labels[idx]] + sliced_s_row)
                            
                    if not b_sub_rows: continue
                    
                    sub_out_dir = b_output_dir.parent / ("rubrics" if s_name == "rubric_score" else "scores")
                    sub_out_dir.mkdir(parents=True, exist_ok=True)
                    s_path = sub_out_dir / f"{s_name}_{pfx_for_file}.csv"
                    write_csv(s_path, b_header_stripped, b_sub_rows)
                    print(f"Wrote Subscore CSV: {s_path}")

if __name__ == "__main__":
    main()
