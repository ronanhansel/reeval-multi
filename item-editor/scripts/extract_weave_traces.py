#!/usr/bin/env python3
"""Export and merge Weave traces by run_id prefix."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import weave  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "traces"

VERBOSE = True
LOG_EVERY = 1
HEARTBEAT_SECONDS = 5.0


def log(message: str) -> None:
    if VERBOSE:
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"[{timestamp}] {message}", flush=True)


class Heartbeat:
    def __init__(self, label: str, interval: float) -> None:
        self._label = label
        self._interval = max(interval, 0.1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = None

    def __enter__(self) -> "Heartbeat":
        if VERBOSE and self._interval > 0:
            self._started = time.monotonic()
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if VERBOSE and self._interval > 0:
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            if not VERBOSE:
                continue
            elapsed = time.monotonic() - self._started if self._started else 0.0
            log(f"{self._label}... ({elapsed:.1f}s elapsed)")


def _patch_wandb_login() -> None:
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    modules = []
    try:
        import wandb  # type: ignore

        modules.append(wandb)
    except Exception:
        log("wandb import failed; relying on weave compat module.")

    try:
        from weave.compat import wandb as weave_wandb  # type: ignore

        modules.append(weave_wandb)
    except Exception:
        log("weave.compat.wandb import failed.")

    if not modules:
        log("No wandb modules available to patch.")
        return

    for module in modules:
        if "referrer" in inspect.signature(module.login).parameters:
            log(f"wandb login signature already supports referrer ({module.__name__}).")
            continue

        original_login = module.login

        def login(*args: Any, _original=original_login, **kwargs: Any):
            kwargs.pop("referrer", None)
            if "key" not in kwargs:
                api_key = os.environ.get("WANDB_API_KEY")
                if api_key:
                    kwargs["key"] = api_key
                    kwargs.setdefault("relogin", True)
            return _original(*args, **kwargs)

        module.login = login  # type: ignore[assignment]
        log(f"Patched wandb login to ignore referrer ({module.__name__}).")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    uri = getattr(value, "uri", None)
    if callable(uri):
        try:
            return uri()
        except Exception:
            pass
    return str(value)


def _make_json_safe(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=_json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Weave traces by run_id prefix.")
    parser.add_argument("--project", required=True, help="Weave project name")
    parser.add_argument("--prefix", action="append", required=True, help="run_id prefix to merge")
    parser.add_argument(
        "--merge-input",
        action="append",
        default=[],
        help="Additional trace files to merge (repeatable).",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge local/merge-input files; skip fetching from Weave.",
    )
    parser.add_argument(
        "--include-costs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fetch and aggregate online cost data when pulling from Weave (default: True).",
    )
    parser.add_argument(
        "--output-suffix",
        default=".json",
        help="suffix for output trace files (default: .json)",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable verbose logging (default: True)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="log every N calls during remote fetch (default: 1)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=5.0,
        help="emit heartbeat logs while waiting (default: 5)",
    )
    return parser.parse_args()


def load_trace(path: Path, allow_missing: bool = False) -> Dict[str, Any] | None:
    """Load a trace file from disk.

    Args:
        path: Path to the trace file
        allow_missing: If True, return None instead of raising FileNotFoundError

    Returns:
        The loaded trace dict, or None if allow_missing=True and file doesn't exist
    """
    log(f"Loading local trace: {path}")
    if not path.exists():
        if allow_missing:
            log(f"  Warning: File not found (will extract task IDs from other sources): {path}")
            return None
        raise FileNotFoundError(f"Trace file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_task_id_from_path(path: Path) -> str | None:
    """Extract task ID from a trace file path like *_<task_id>_*_UPLOAD.json."""
    name = path.stem  # Remove .json
    # Pattern: ..._{task_id}_scienceagentbench_... or ..._{task_id}_scicode_...
    parts = name.split("_")
    for i, part in enumerate(parts):
        if part in ("scienceagentbench", "scicode", "corebench", "usaco", "assistantbench", "colbench"):
            # Task ID should be the part just before the benchmark name
            if i > 0 and parts[i - 1].isdigit():
                return parts[i - 1]
    return None


def merge_local_traces(
    paths: List[Path],
    run_id: str,
    missing_merge_inputs: List[Path] | None = None,
    individual_trace_paths: List[Path] | None = None,
) -> Dict[str, Any]:
    """Merge multiple trace files into a single trace.

    Args:
        paths: List of trace files to merge (existing files only)
        run_id: The run identifier for the merged trace
        missing_merge_inputs: List of merge-input files that were not found.
            Task IDs will be extracted from individual traces and marked as failed.
        individual_trace_paths: List of individual trace files (e.g., *_UPLOAD.json)
            used to extract task IDs when merge inputs are missing.
    """
    missing_merge_inputs = missing_merge_inputs or []
    individual_trace_paths = individual_trace_paths or []

    # Extract task IDs from individual traces when merge inputs are missing
    task_ids_from_individual_traces: set[str] = set()
    if missing_merge_inputs and individual_trace_paths:
        log(f"Missing merge-input files: {[p.name for p in missing_merge_inputs]}")
        log(f"Extracting task IDs from {len(individual_trace_paths)} individual trace files...")
        for trace_path in individual_trace_paths:
            task_id = _extract_task_id_from_path(trace_path)
            if task_id:
                task_ids_from_individual_traces.add(task_id)
        log(f"Found {len(task_ids_from_individual_traces)} task IDs from individual traces: {sorted(task_ids_from_individual_traces, key=lambda x: int(x) if x.isdigit() else x)}")

    if not paths and not task_ids_from_individual_traces:
        log(f"No local traces found for run_id={run_id}.")
        return {
            "config": {
                "run_id": run_id,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source_traces": [],
            },
            "results": {
                "accuracy": 0.0,
                "written_accuracy": 0.0,
                "vision_accuracy": 0.0,
                "successful_tasks": [],
                "failed_tasks": [],
                "total_cost": 0.0,
                "latencies": {},
            },
            "raw_eval_results": {},
            "raw_logging_results": [],
            "total_usage": {},
            "total_cost": 0.0,
            "git_info": None,
        }

    # Initialize config from first available trace or create minimal config
    config: Dict[str, Any] = {}
    git_info = None
    if paths:
        first_trace = load_trace(paths[0])
        if first_trace:
            config = json.loads(json.dumps(first_trace.get("config", {})))
            git_info = first_trace.get("git_info")
    if not config:
        config = {
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    config["run_id"] = run_id
    config["source_traces"] = [p.name for p in paths]
    if missing_merge_inputs:
        config["missing_merge_inputs"] = [p.name for p in missing_merge_inputs]

    merged_results: Dict[str, Any] = {
        "successful_tasks": [],
        "failed_tasks": [],
        "latencies": {},
    }
    total_cost = 0.0
    total_usage: Dict[str, Dict[str, float]] = {}
    raw_eval_results: Dict[str, Any] = {}
    raw_logging_results: List[Dict[str, Any]] = []

    written_correct = vision_correct = 0
    written_total = vision_total = 0

    # Track tasks that have been classified
    successful_tasks_set: set[str] = set()
    failed_tasks_set: set[str] = set()

    # Track tasks that have eval results (to detect missing results)
    tasks_with_eval_results: set[str] = set()

    for path in paths:
        data = load_trace(path)
        if data is None:
            continue

        results = data.get("results", {})

        # Extract successful/failed tasks from results lists
        for task_id in (results.get("successful_tasks") or []):
            successful_tasks_set.add(str(task_id))
        for task_id in (results.get("failed_tasks") or []):
            failed_tasks_set.add(str(task_id))

        # Also extract from raw_eval_results if available
        raw_eval = data.get("raw_eval_results", {})
        if isinstance(raw_eval, dict):
            eval_results = raw_eval.get("eval_result", {})
        else:
            eval_results = {}
        for task_id, eval_data in eval_results.items():
            tasks_with_eval_results.add(str(task_id))
            if isinstance(eval_data, dict):
                success_rate = eval_data.get("success_rate", 0)
                if success_rate > 0:
                    successful_tasks_set.add(str(task_id))
                else:
                    failed_tasks_set.add(str(task_id))

        log(
            f"Merging local trace {path.name} "
            f"(success={len(results.get('successful_tasks') or [])}, "
            f"fail={len(results.get('failed_tasks') or [])})"
        )

        if results.get("latencies"):
            for key, value in results["latencies"].items():
                if isinstance(value, dict):
                    existing = merged_results["latencies"].setdefault(
                        key,
                        {
                            "first_call_timestamp": value.get("first_call_timestamp"),
                            "last_call_timestamp": value.get("last_call_timestamp"),
                        },
                    )
                    if value.get("first_call_timestamp"):
                        existing["first_call_timestamp"] = min(
                            existing["first_call_timestamp"],
                            value["first_call_timestamp"],
                        )
                    if value.get("last_call_timestamp"):
                        existing["last_call_timestamp"] = max(
                            existing["last_call_timestamp"],
                            value["last_call_timestamp"],
                        )
                else:
                    merged_results["latencies"][key] = (
                        merged_results["latencies"].get(key, 0) + value
                    )

        total_cost += results.get("total_cost") or data.get("total_cost") or 0.0

        usage = data.get("total_usage") or {}
        for model_name, usage_stats in usage.items():
            bucket = total_usage.setdefault(
                model_name, {"input_tokens": 0, "output_tokens": 0}
            )
            bucket["input_tokens"] += usage_stats.get("input_tokens", 0)
            bucket["output_tokens"] += usage_stats.get("output_tokens", 0)

        capsule_eval = data.get("raw_eval_results") or {}
        if not isinstance(capsule_eval, dict):
            capsule_eval = {}
        for task_id, stats in capsule_eval.items():
            raw_eval_results[task_id] = stats
            if isinstance(stats, dict):
                written_correct += stats.get("correct_written_answers", 0)
                vision_correct += stats.get("correct_vision_answers", 0)
                written_total += stats.get("total_written_questions", 0)
                vision_total += stats.get("total_vision_questions", 0)

        raw_logging_results.extend(data.get("raw_logging_results") or [])

    # Handle tasks from missing merge-inputs: mark as failed if they have traces but no eval results
    if task_ids_from_individual_traces:
        for task_id in task_ids_from_individual_traces:
            if task_id not in tasks_with_eval_results and task_id not in successful_tasks_set:
                log(f"Task {task_id} has traces but no eval results in merge-input -> marking as failed")
                failed_tasks_set.add(task_id)

    # Convert sets to sorted lists
    merged_results["successful_tasks"] = sorted(successful_tasks_set, key=lambda x: int(x) if x.isdigit() else x)
    merged_results["failed_tasks"] = sorted(failed_tasks_set, key=lambda x: int(x) if x.isdigit() else x)

    total_tasks = len(merged_results["successful_tasks"]) + len(merged_results["failed_tasks"])
    merged_results["accuracy"] = (
        len(merged_results["successful_tasks"]) / total_tasks if total_tasks else 0.0
    )
    merged_results["written_accuracy"] = (
        written_correct / written_total if written_total else 0.0
    )
    merged_results["vision_accuracy"] = (
        vision_correct / vision_total if vision_total else 0.0
    )
    merged_results["total_cost"] = total_cost

    return {
        "config": config,
        "results": merged_results,
        "raw_eval_results": raw_eval_results,
        "raw_logging_results": raw_logging_results,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "git_info": git_info,
    }


# ---------------------------
# NEW: Weave object-based trace collection
# ---------------------------

def flatten_trace(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any]):
        calls.append(node)
        for child in node.get("children", []) or []:
            visit(child)

    visit(trace)
    return calls


def process_call(call: Dict[str, Any]) -> Dict[str, Any]:
    attrs = call.get("attributes") or {}
    return {
        **call,
        "weave_task_id": attrs.get("weave_task_id"),
        "created_timestamp": call.get("started_at"),
    }


def build_latency(processed_calls: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latency: Dict[str, Dict[str, Any]] = {}

    for call in processed_calls:
        task_id = call.get("weave_task_id")
        started_at = call.get("started_at")
        if not task_id or not started_at:
            continue

        bucket = latency.setdefault(
            task_id,
            {
                "first_call_timestamp": started_at,
                "last_call_timestamp": started_at,
            },
        )

        bucket["first_call_timestamp"] = min(
            bucket["first_call_timestamp"], started_at
        )
        bucket["last_call_timestamp"] = max(
            bucket["last_call_timestamp"], started_at
        )

    for task_id, entry in latency.items():
        try:
            entry["total_time"] = (
                datetime.fromisoformat(entry["last_call_timestamp"])
                - datetime.fromisoformat(entry["first_call_timestamp"])
            ).total_seconds()
        except Exception:
            entry["total_time"] = None

    return latency


def _build_prefix_query(prefixes: List[str]) -> Dict[str, Any] | None:
    if not prefixes:
        return None

    exprs = [
        {
            "$contains": {
                "input": {"$getField": "attributes.run_id"},
                "substr": {"$literal": prefix},
                "case_insensitive": False,
            }
        }
        for prefix in prefixes
    ]

    if len(exprs) == 1:
        expr = exprs[0]
    else:
        expr = {"$or": exprs}

    return {"$expr": expr}


def _serialize_call(call: Any) -> Dict[str, Any]:
    call_dict = call.to_dict()
    call_dict["wb_run_id"] = getattr(call, "wb_run_id", None)
    call_dict["wb_run_step"] = getattr(call, "wb_run_step", None)
    call_dict["wb_run_step_end"] = getattr(call, "wb_run_step_end", None)
    call_dict["storage_size_bytes"] = getattr(call, "storage_size_bytes", None)
    call_dict["total_storage_size_bytes"] = getattr(call, "total_storage_size_bytes", None)
    return _make_json_safe(call_dict)


def _extract_cost(entry: Dict[str, Any]) -> float:
    summary = entry.get("summary") or {}
    weave_summary = summary.get("weave") or {}

    for key in ("total_cost", "cost", "cost_usd", "total_cost_usd"):
        value = weave_summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    costs = (
        weave_summary.get("costs")
        or weave_summary.get("cost_by_model")
        or summary.get("costs")
    )
    if isinstance(costs, dict):
        total = 0.0
        found = False
        if isinstance(costs.get("total_cost"), (int, float)):
            return float(costs["total_cost"])
        for value in costs.values():
            if isinstance(value, (int, float)):
                total += float(value)
                found = True
            elif isinstance(value, dict):
                for key in ("total_cost", "cost", "cost_usd", "total_cost_usd"):
                    subvalue = value.get(key)
                    if isinstance(subvalue, (int, float)):
                        total += float(subvalue)
                        found = True
                        break
        if found:
            return total

    return 0.0


def _aggregate_usage_and_cost(calls: List[Dict[str, Any]]) -> tuple[Dict[str, Dict[str, float]], float]:
    usage_totals: Dict[str, Dict[str, float]] = {}
    total_cost = 0.0

    for entry in calls:
        summary = entry.get("summary") or {}
        usage = summary.get("usage") or {}
        if isinstance(usage, dict):
            for model_name, stats in usage.items():
                if not isinstance(stats, dict):
                    continue
                bucket = usage_totals.setdefault(
                    model_name, {"input_tokens": 0.0, "output_tokens": 0.0}
                )
                bucket["input_tokens"] += float(
                    stats.get("input_tokens", stats.get("prompt_tokens", 0)) or 0.0
                )
                bucket["output_tokens"] += float(
                    stats.get("output_tokens", stats.get("completion_tokens", 0)) or 0.0
                )

        total_cost += _extract_cost(entry)

    return usage_totals, total_cost


def _prefix_variants(prefix: str) -> List[str]:
    variants = [prefix]
    for suffix in ("_high", "_medium", "_low"):
        if prefix.endswith(suffix):
            variants.append(prefix[: -len(suffix)])
            break
    return variants


def _matches_prefix(path: Path, prefix: str) -> bool:
    name = path.name
    for suffix in ("_high", "_medium", "_low"):
        if prefix.endswith(suffix):
            base = prefix[: -len(suffix)]
            token = suffix.lstrip("_")
            return base in name and token in name
    return any(variant in name for variant in _prefix_variants(prefix))


def fetch_calls(
    project: str,
    prefixes: List[str],
    client: Any,
    include_costs: bool,
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    list_fn = getattr(getattr(weave, "object", None), "list", None)
    if callable(list_fn):
        log("Using weave.object.list API for trace collection.")
        for obj in list_fn(project=project, kind="trace"):
            log(f"Inspecting trace object: {obj.ref}")
            ref = weave.ref(obj.ref)
            trace = ref.get()

            attrs = trace.get("attributes") or {}
            run_id = attrs.get("run_id")

            if not run_id or not any(str(run_id).startswith(p) for p in prefixes):
                log(f"Skipping trace (run_id={run_id}).")
                continue

            for call in flatten_trace(trace):
                calls.append(process_call(call))
        return calls

    query = _build_prefix_query(prefixes)
    log(f"Using client.get_calls API for trace collection (query={query}, include_costs={include_costs}).")
    log_every = max(LOG_EVERY, 1)
    seen = 0
    skipped = 0
    with Heartbeat("Fetching calls", HEARTBEAT_SECONDS):
        call_iterator = iter(client.get_calls(query=query, page_size=1000, include_costs=include_costs))
        while True:
            try:
                # Get next call from iterator (this is where deserialization errors occur)
                call = next(call_iterator)

                # Serialize and process the call
                try:
                    serialized = _serialize_call(call)
                    attrs = serialized.get("attributes") or {}
                    run_id = attrs.get("run_id")
                    if not run_id or not any(str(run_id).startswith(p) for p in prefixes):
                        continue
                    calls.append(process_call(serialized))
                    seen += 1
                    if seen % log_every == 0:
                        log(
                            f"Fetched calls={seen} "
                            f"(last id={serialized.get('id')}, op={serialized.get('op_name')}, "
                            f"run_id={run_id})"
                        )
                except Exception as e:
                    skipped += 1
                    log(f"Warning: Failed to serialize/process call: {e}")
                    if skipped <= 5:
                        import traceback
                        log(f"  Traceback: {traceback.format_exc()}")

            except StopIteration:
                # Normal end of iteration
                break
            except (ValueError, TypeError, KeyError) as e:
                # Deserialization error from Weave SDK - skip this call and continue
                skipped += 1
                if "No serializer found" in str(e) or "Content.content" in str(e):
                    if skipped % 10 == 1:  # Log first and every 10th
                        log(f"Skipped {skipped} call(s) with deserialization issues (Content type not supported)")
                else:
                    log(f"Warning: Deserialization error: {e}")
                    if skipped <= 5:
                        import traceback
                        log(f"  Traceback: {traceback.format_exc()}")
            except Exception as e:
                # Unexpected error - log but continue
                skipped += 1
                log(f"Error fetching call: {e}")
                import traceback
                log(f"  Traceback: {traceback.format_exc()}")
                # Try to continue with next call
                continue

    log(f"Finished fetching calls: {len(calls)} total (skipped {skipped} problematic calls).")

    return calls



def main() -> None:
    args = parse_args()
    global VERBOSE, LOG_EVERY, HEARTBEAT_SECONDS
    VERBOSE = args.verbose
    LOG_EVERY = args.log_every
    HEARTBEAT_SECONDS = args.heartbeat_seconds

    log("Starting trace extraction.")
    log(f"Project: {args.project}")
    log(f"Prefixes: {args.prefix}")
    log(f"Output suffix: {args.output_suffix}")
    log(f"WANDB_API_KEY present: {'yes' if os.environ.get('WANDB_API_KEY') else 'no'}")
    log(f"Include costs: {args.include_costs}")
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Traces dir: {TRACES_DIR}")

    extra_paths = [Path(p).expanduser() for p in (args.merge_input or [])]
    for path in extra_paths:
        log(f"Merge input: {path}")

    all_calls: List[Dict[str, Any]] = []
    if args.merge_only:
        log("merge-only enabled; skipping Weave fetch.")
    else:
        _patch_wandb_login()
        with Heartbeat("Initializing Weave", HEARTBEAT_SECONDS):
            client = weave.init(args.project)
        log("Weave init complete.")

        all_calls = fetch_calls(args.project, args.prefix, client, include_costs=args.include_costs)

    for i, prefix in enumerate(args.prefix):
        log(f"Processing prefix: {prefix}")

        # Look for individual UPLOAD files first (e.g., sab_cow__sab_cow_openai_gpt-4_1_*_UPLOAD.json)
        # These contain the evaluation results per task
        individual_pattern = f"*{prefix}*_UPLOAD.json"
        local_paths = sorted(TRACES_DIR.glob(individual_pattern))

        # Filter to only include files that match the prefix pattern more precisely
        # (avoid matching partial prefixes)
        local_paths = [p for p in local_paths if prefix in p.name]

        # Keep a copy of individual trace paths before adding merge-inputs
        # These are used to extract task IDs if merge-input files are missing
        individual_trace_paths = list(local_paths)

        # Positional matching: i-th prefix gets i-th merge-input file
        extra_for_prefix = [extra_paths[i]] if i < len(extra_paths) else []

        # Check which merge-input files exist vs missing
        existing_merge_inputs: List[Path] = []
        missing_merge_inputs: List[Path] = []
        for path in extra_for_prefix:
            if path.exists():
                existing_merge_inputs.append(path)
            else:
                missing_merge_inputs.append(path)

        if existing_merge_inputs:
            log(f"Merge-input file for {prefix}: {[p.name for p in existing_merge_inputs]}")
        if missing_merge_inputs:
            log(f"WARNING: Missing merge-input files for {prefix}: {[p.name for p in missing_merge_inputs]}")
            log(f"  Will extract task IDs from individual traces and mark as failed")

        merged_by_name = {p.name: p for p in local_paths}
        for path in existing_merge_inputs:
            merged_by_name[path.name] = path
        local_paths = sorted(p.resolve() for p in merged_by_name.values())
        log(f"Local traces found: {len(local_paths)}")

        merged = merge_local_traces(
            local_paths,
            run_id=prefix,
            missing_merge_inputs=missing_merge_inputs,
            individual_trace_paths=individual_trace_paths,
        )

        filtered_calls = [
            c for c in all_calls
            if c.get("attributes", {}).get("run_id", "").startswith(prefix)
        ]
        log(f"Remote calls matched prefix {prefix}: {len(filtered_calls)}")

        if filtered_calls:
            merged["raw_logging_results"] = filtered_calls
            merged["results"]["latencies"] = build_latency(filtered_calls)
            usage_totals, total_cost = _aggregate_usage_and_cost(filtered_calls)
            if usage_totals:
                merged["total_usage"] = usage_totals
            if total_cost:
                merged["total_cost"] = total_cost
                merged["results"]["total_cost"] = total_cost
            log(f"Latencies computed for {len(merged['results']['latencies'])} tasks.")

            # If merge-inputs were missing, also extract task IDs from fetched weave calls
            # and mark tasks with traces but no eval results as failed
            if missing_merge_inputs:
                task_ids_from_weave: set[str] = set()
                for call in filtered_calls:
                    task_id = call.get("weave_task_id")
                    if task_id:
                        task_ids_from_weave.add(str(task_id))

                if task_ids_from_weave:
                    log(f"Found {len(task_ids_from_weave)} task IDs from Weave calls: {sorted(task_ids_from_weave, key=lambda x: int(x) if x.isdigit() else x)}")

                    # Check which tasks have traces but no eval results
                    existing_eval_tasks = set(merged["raw_eval_results"].keys()) if merged.get("raw_eval_results") else set()
                    existing_success_tasks = set(str(t) for t in (merged.get("results", {}).get("successful_tasks") or []))
                    existing_failed_tasks = set(str(t) for t in (merged.get("results", {}).get("failed_tasks") or []))

                    newly_failed = []
                    for task_id in task_ids_from_weave:
                        if (task_id not in existing_eval_tasks and
                            task_id not in existing_success_tasks and
                            task_id not in existing_failed_tasks):
                            newly_failed.append(task_id)

                    if newly_failed:
                        log(f"Marking {len(newly_failed)} tasks from Weave traces as failed (no eval results in missing merge-input)")
                        # Add to failed_tasks list
                        current_failed = set(str(t) for t in (merged.get("results", {}).get("failed_tasks") or []))
                        current_failed.update(newly_failed)
                        merged["results"]["failed_tasks"] = sorted(current_failed, key=lambda x: int(x) if x.isdigit() else x)

                        # Recalculate accuracy
                        total_tasks = len(merged["results"].get("successful_tasks") or []) + len(merged["results"]["failed_tasks"])
                        merged["results"]["accuracy"] = (
                            len(merged["results"].get("successful_tasks") or []) / total_tasks if total_tasks else 0.0
                        )
        else:
            log("No remote calls; keeping merged local raw_logging_results.")

        output_path = TRACES_DIR / f"{prefix}{args.output_suffix}"
        output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        log(
            f"Wrote {output_path} "
            f"(calls={len(filtered_calls)}, local_traces={len(local_paths)})"
        )


if __name__ == "__main__":
    main()
