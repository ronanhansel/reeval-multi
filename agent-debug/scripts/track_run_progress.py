#!/usr/bin/env python3
"""
Track aggregate progress for each benchmark run.

Shows completed tasks (from RAW_SUBMISSIONS.jsonl) and evaluated tasks
(from *_eval.jsonl or *_UPLOAD.json when available). Use --per-agent
to see individual agent rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BENCHMARKS = ("scicode", "scienceagentbench", "corebench", "colbench")
HAL_BENCHMARK_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
}

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
TOKEN_RE = re.compile(r"Input tokens:\s*([\d,]+).*Output tokens:\s*([\d,]+)")
PREFIX_RE = re.compile(r"^(.*?)(\d+)([^0-9]*)$")
RUN_DIR_RE = re.compile(r"benchmark_run_(\d{8}_\d{6})$")


@dataclass
class VerboseMetrics:
    start: Optional[datetime]
    end: Optional[datetime]
    tokens: int
    token_start: Optional[datetime]
    token_end: Optional[datetime]


@dataclass
class AggregateMetrics:
    benchmark: str
    runs: int
    tasks_done: Optional[int]
    tasks_total: Optional[int]
    tokens_total: int


@dataclass
class WatchState:
    run_dir: Path
    current_prefix: Optional[str]
    next_prefix: Optional[str]
    auto_command: Optional[str]
    auto_tracker: "AutoRelaunch"
    rate_tracker: Optional[RateTracker]


class RateTracker:
    def __init__(self, window_seconds: int) -> None:
        self.window_seconds = max(1, window_seconds)
        self.history: Dict[str, List[Tuple[float, Optional[int], int]]] = {}

    def update(self, metrics: List[AggregateMetrics]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        now = time.time()
        rates: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        cutoff = now - self.window_seconds
        for metric in metrics:
            hist = self.history.setdefault(metric.benchmark, [])
            hist.append((now, metric.tasks_done, metric.tokens_total))
            while hist and hist[0][0] < cutoff:
                hist.pop(0)

            tasks_rate = self._compute_rate(hist, value_index=1, cutoff=cutoff)
            tokens_rate = self._compute_rate(hist, value_index=2, cutoff=cutoff)
            rates[metric.benchmark] = (tasks_rate, tokens_rate)
        return rates

    @staticmethod
    def _compute_rate(
        hist: List[Tuple[float, Optional[int], int]],
        value_index: int,
        cutoff: float,
    ) -> Optional[float]:
        if not hist:
            return None
        latest = hist[-1]
        latest_value = latest[value_index]
        if latest_value is None:
            return None
        start_entry = None
        for entry in hist:
            if entry[0] >= cutoff and entry[value_index] is not None:
                start_entry = entry
                break
        if start_entry is None:
            return None
        delta = latest_value - start_entry[value_index]  # type: ignore[operator]
        if delta < 0:
            return None
        dt = latest[0] - start_entry[0]
        if dt <= 0:
            return None
        return delta / (dt / 60.0)


def detect_run_root(script_dir: Path) -> Path:
    local_hal = script_dir / ".hal_data"
    if local_hal.is_dir() and os.access(local_hal, os.W_OK):
        return local_hal
    data_path = os.environ.get("DATA_PATH")
    if data_path and Path(data_path).is_dir():
        namespace = os.environ.get("HAL_DATA_NAMESPACE") or os.environ.get("USER") or "user"
        return Path(data_path) / "hal_runs" / namespace / script_dir.name
    data_root = os.environ.get("HAL_DATA_ROOT")
    if data_root and Path(data_root).is_dir():
        namespace = os.environ.get("HAL_DATA_NAMESPACE") or os.environ.get("USER") or "user"
        return Path(data_root) / "hal_runs" / namespace / script_dir.name
    return script_dir


def latest_run_dir(logs_dir: Path) -> Optional[Path]:
    runs = list(logs_dir.glob("benchmark_run_*"))
    if not runs:
        return None
    def run_key(path: Path) -> str:
        match = RUN_DIR_RE.fullmatch(path.name)
        if match:
            return match.group(1)
        return path.name
    return max(runs, key=run_key)


def collect_logs_roots(script_dir: Path, run_root: Path) -> List[Path]:
    roots: List[Path] = []

    def add_root(path: Path) -> None:
        if path.is_dir() and path not in roots:
            roots.append(path)

    add_root(script_dir / ".hal_data" / "logs")
    add_root(run_root / "logs")
    add_root(script_dir / "logs")
    add_root(script_dir / ".logs")
    return roots


def latest_run_dir_from_roots(logs_roots: List[Path], prefix: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for root in logs_roots:
        candidates.extend(root.glob("benchmark_run_*"))
    
    if prefix:
        filtered = []
        for cand in candidates:
            if not cand.is_dir():
                continue
            cfg = cand / "config.json"
            if cfg.exists():
                try:
                    data = json.loads(cfg.read_text())
                    if data.get("prefix") == prefix:
                        filtered.append(cand)
                except Exception:
                    pass
        candidates = filtered

    if not candidates:
        return None
    def run_key(path: Path) -> str:
        match = RUN_DIR_RE.fullmatch(path.name)
        if match:
            return match.group(1)
        return path.name
    return max(candidates, key=run_key)


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def parse_total_tasks(text: str) -> Optional[int]:
    totals = [int(m.group(1)) for m in re.finditer(r"\((\d+) tasks\)", text)]
    return max(totals) if totals else None


def parse_run_root(text: str) -> Optional[str]:
    match = re.search(r"\[Storage\] Run root: (.+)", text)
    if match:
        return match.group(1).strip()
    return None


def parse_run_ids(text: str) -> Iterable[str]:
    return sorted(set(re.findall(r"Run ID: (\S+)", text)))


def filter_latest_run_ids(run_ids: Iterable[str]) -> List[str]:
    run_ids = list(run_ids)
    if not run_ids:
        return []
    stamp_re = re.compile(r"_(\d{8}_\d{6})$")
    stamps = [(rid, stamp_re.search(rid)) for rid in run_ids]
    stamped = [(rid, match.group(1)) for rid, match in stamps if match]
    if not stamped:
        return run_ids
    latest_stamp = max(stamp for _, stamp in stamped)
    return [rid for rid, stamp in stamped if stamp == latest_stamp]


def load_prefix(run_dir: Path) -> Optional[str]:
    cfg = run_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text()).get("prefix")
    except Exception:
        return None


def derive_config_key(run_id: str, prefix: Optional[str]) -> str:
    if prefix and run_id.startswith(prefix):
        run_id = run_id[len(prefix):]
    return re.sub(r"_[0-9]{8}_[0-9]{6}$", "", run_id)


def resolve_run_dir(run_root: Path, repo_root: Path, benchmark: str, run_id: str) -> Optional[Path]:
    hal_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)
    env_results = os.environ.get("HAL_RESULTS_DIR")
    extra_roots = []
    if env_results:
        extra_roots.append(Path(env_results))
    else:
        local_hal = repo_root / ".hal_data" / "results"
        if local_hal.exists():
            extra_roots.append(local_hal)
    candidates = [
        run_root / "results" / hal_name / run_id,
        run_root / "results" / benchmark / run_id,
        repo_root / "results" / hal_name / run_id,
        repo_root / "results" / benchmark / run_id,
    ]
    for root in extra_roots:
        candidates.append(root / hal_name / run_id)
        candidates.append(root / benchmark / run_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def count_raw_submissions(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not path.exists():
        return None, None, None
    entries: Dict[str, object] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj:
                    task_id = next(iter(obj.keys()))
                    entries[task_id] = obj[task_id]
    except Exception:
        return None, None, None
    errors = sum(1 for v in entries.values() if isinstance(v, str) and v.startswith("ERROR"))
    completed = sum(1 for v in entries.values() if not (isinstance(v, str) and v.startswith("ERROR")))
    return completed, errors, len(entries)


def count_eval_scienceagentbench(run_dir: Path, run_id: str) -> Optional[int]:
    eval_path = run_dir / f"{run_id}_eval.jsonl"
    if not eval_path.exists():
        return None
    count = 0
    try:
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return None
    return count


def count_eval_from_upload(run_dir: Path, run_id: str) -> Optional[int]:
    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    if not upload_path.exists():
        return None
    try:
        data = json.loads(upload_path.read_text())
    except Exception:
        return None
    results = data.get("results", {})
    successful = results.get("successful_tasks")
    failed = results.get("failed_tasks")
    if isinstance(successful, list) and isinstance(failed, list):
        return len(successful) + len(failed)
    return None


def status_label(completed_ok: Optional[int], completed_err: Optional[int], total_tasks: Optional[int], eval_count: Optional[int]) -> str:
    if completed_ok is None and eval_count is None:
        return "UNKNOWN"
    if completed_err and completed_err > 0:
        return "ERRORS"
    if total_tasks is not None and completed_ok is not None and completed_ok < total_tasks:
        return "INCOMPLETE"
    if total_tasks is not None and eval_count is not None and eval_count < total_tasks:
        return "EVAL_INCOMPLETE"
    return "DONE"


def build_table(rows: Iterable[Tuple[str, ...]]) -> List[str]:
    rows = list(rows)
    if not rows:
        return []
    col_count = max(len(row) for row in rows)
    col_widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    lines = []
    for row in rows:
        padded = list(row) + [""] * (col_count - len(row))
        lines.append("  ".join(padded[i].ljust(col_widths[i]) for i in range(col_count)))
    return lines


def print_table(rows: Iterable[Tuple[str, ...]]) -> None:
    for line in build_table(rows):
        print(line)


def parse_timestamp(line: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def parse_verbose_metrics(path: Path) -> VerboseMetrics:
    if not path.exists():
        return VerboseMetrics(None, None, 0, None, None)

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    token_start: Optional[datetime] = None
    token_end: Optional[datetime] = None
    total_tokens = 0
    last_total: Optional[int] = None
    last_ts: Optional[datetime] = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                ts = parse_timestamp(line)
                if ts:
                    last_ts = ts
                    if start is None or ts < start:
                        start = ts
                    if end is None or ts > end:
                        end = ts

                token_match = TOKEN_RE.search(line)
                if token_match:
                    input_tokens = int(token_match.group(1).replace(",", ""))
                    output_tokens = int(token_match.group(2).replace(",", ""))
                    current_total = input_tokens + output_tokens
                    if last_total is None:
                        delta = current_total
                    else:
                        delta = current_total - last_total
                        if delta < 0:
                            delta = current_total
                    if delta > 0:
                        total_tokens += delta
                    last_total = current_total

                    token_ts = ts or last_ts
                    if token_ts:
                        if token_start is None or token_ts < token_start:
                            token_start = token_ts
                        if token_end is None or token_ts > token_end:
                            token_end = token_ts
    except Exception:
        return VerboseMetrics(None, None, 0, None, None)

    return VerboseMetrics(start, end, total_tokens, token_start, token_end)


def parse_local_trace_tokens(run_dir: Path) -> int:
    total = 0
    try:
        for entry in run_dir.iterdir():
            if not entry.is_dir():
                continue
            trace_path = entry / "local_trace.jsonl"
            if not trace_path.exists():
                continue
            try:
                with trace_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        output = payload.get("output")
                        if not isinstance(output, dict):
                            continue
                        usage = output.get("usage")
                        if not isinstance(usage, dict):
                            continue
                        total_tokens = usage.get("total_tokens")
                        if isinstance(total_tokens, (int, float)):
                            total += int(total_tokens)
                            continue
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        input_tokens = usage.get("input_tokens")
                        output_tokens = usage.get("output_tokens")
                        if isinstance(prompt_tokens, (int, float)) or isinstance(completion_tokens, (int, float)):
                            total += int(prompt_tokens or 0) + int(completion_tokens or 0)
                            continue
                        if isinstance(input_tokens, (int, float)) or isinstance(output_tokens, (int, float)):
                            total += int(input_tokens or 0) + int(output_tokens or 0)
            except Exception:
                continue
    except Exception:
        return 0
    return total


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return f"{value:,.1f}"


def format_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "?"


def format_window(window_seconds: int) -> str:
    if window_seconds % 60 == 0:
        return f"{window_seconds // 60}m"
    return f"{window_seconds}s"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{mins:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_dt(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def display_path(path: Optional[Path], repo_root: Path) -> str:
    if path is None:
        return "?"
    try:
        return str(path.relative_to(repo_root))
    except Exception:
        return str(path)


def increment_prefix(prefix: Optional[str]) -> Optional[str]:
    if not prefix:
        return None
    match = PREFIX_RE.match(prefix)
    if not match:
        return f"{prefix}2"
    base, digits, suffix = match.groups()
    width = len(digits)
    next_num = str(int(digits) + 1).zfill(width)
    return f"{base}{next_num}{suffix}"


def build_auto_command(next_prefix: Optional[str], parallel_tasks: int) -> Optional[str]:
    if not next_prefix:
        return None
    return (
        "./run_all_benchmarks.sh --benchmarks colbench "
        f"--prefix {next_prefix} --parallel-models 10 --parallel-tasks {parallel_tasks} --trace-mode local"
    )


class AutoRelaunch:
    def __init__(
        self,
        enabled: bool,
        command: Optional[str],
        repo_root: Path,
        logs_dir: Path,
        stall_seconds: int = 7 * 60,
        delay_seconds: int = 0,
        complete_delay_seconds: int = 5 * 60,
    ) -> None:
        self.enabled = enabled
        self.command = command
        self.repo_root = repo_root
        self.logs_dir = logs_dir
        self.stall_seconds = stall_seconds
        self.delay_seconds = delay_seconds
        self.complete_delay_seconds = complete_delay_seconds
        self.last_done: Optional[int] = None
        self.last_change: Optional[float] = None
        self.stall_pending_since: Optional[float] = None
        self.complete_pending_since: Optional[float] = None
        self.triggered_at: Optional[float] = None
        self.trigger_reason: Optional[str] = None
        self.trigger_log: Optional[Path] = None
        self.trigger_pid: Optional[int] = None

    def update(self, tasks_done: Optional[int], tasks_total: Optional[int]) -> None:
        now = time.time()
        if not self.enabled or self.triggered_at is not None:
            return

        if tasks_done is None:
            self.last_done = None
            self.last_change = None
            self.stall_pending_since = None
            self.complete_pending_since = None
            return

        if self.last_done is None or tasks_done != self.last_done:
            self.last_done = tasks_done
            self.last_change = now
            self.stall_pending_since = None
            self.complete_pending_since = None
        elif self.last_change is None:
            self.last_change = now

        if self.last_change and (now - self.last_change) >= self.stall_seconds:
            if self.stall_pending_since is None:
                self.stall_pending_since = now
        else:
            self.stall_pending_since = None

        if tasks_total is not None and tasks_done >= tasks_total:
            if self.complete_pending_since is None:
                self.complete_pending_since = now
        else:
            self.complete_pending_since = None

        if self.stall_pending_since and (now - self.stall_pending_since) >= self.delay_seconds:
            self._trigger("stall")
        elif self.complete_pending_since and (now - self.complete_pending_since) >= self.complete_delay_seconds:
            self._trigger("complete")

    def status_line(self, tasks_done: Optional[int], tasks_total: Optional[int], repo_root: Path) -> str:
        if not self.enabled:
            return "Auto-relaunch: disabled (use --batch-mode to enable)"
        if not self.command:
            return "Auto-relaunch: enabled (set --prefix to compute next command)"
        if self.triggered_at is not None:
            log_path = display_path(self.trigger_log, repo_root)
            return (
                "Auto-relaunch: triggered "
                f"({self.trigger_reason}) pid={self.trigger_pid} at {format_dt(self.triggered_at)} "
                f"log={log_path}"
            )

        now = time.time()
        idle = None if self.last_change is None else now - self.last_change
        if self.complete_pending_since is not None:
            wait_left = self.complete_delay_seconds - (now - self.complete_pending_since)
            total_display = tasks_total if tasks_total is not None else "?"
            return (
                "Auto-relaunch: pending complete "
                f"({tasks_done}/{total_display}), launching in {format_duration(wait_left)}"
            )
        if self.stall_pending_since is not None:
            wait_left = self.delay_seconds - (now - self.stall_pending_since)
            return (
                "Auto-relaunch: pending stall "
                f"(idle {format_duration(idle)}), launching in {format_duration(wait_left)}"
            )
        if idle is None:
            return "Auto-relaunch: waiting for progress signal"
        return f"Auto-relaunch: enabled (idle {format_duration(idle)}; stall=7m, complete=5m)"

    def _trigger(self, reason: str) -> None:
        if not self.command:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"auto_relaunch_{timestamp}.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", self.command],
                cwd=self.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        self.triggered_at = time.time()
        self.trigger_reason = reason
        self.trigger_log = log_path
        self.trigger_pid = proc.pid


def build_watch_state(
    run_dir: Path,
    prefix_override: Optional[str],
    batch_mode: bool,
    repo_root: Path,
    logs_dir: Path,
    window_seconds: int,
    watch: bool,
    parallel_tasks: int,
) -> WatchState:
    current_prefix = load_prefix(run_dir) or prefix_override
    next_prefix = increment_prefix(current_prefix)
    auto_command = build_auto_command(next_prefix, parallel_tasks)
    auto_tracker = AutoRelaunch(
        enabled=batch_mode,
        command=auto_command,
        repo_root=repo_root,
        logs_dir=logs_dir,
    )
    rate_tracker = RateTracker(window_seconds) if watch else None
    return WatchState(
        run_dir=run_dir,
        current_prefix=current_prefix,
        next_prefix=next_prefix,
        auto_command=auto_command,
        auto_tracker=auto_tracker,
        rate_tracker=rate_tracker,
    )


def maybe_refresh_state(
    state: WatchState,
    follow_latest: bool,
    prefix_override: Optional[str],
    batch_mode: bool,
    repo_root: Path,
    logs_roots: List[Path],
    window_seconds: int,
    watch: bool,
    parallel_tasks: int,
) -> WatchState:
    if not follow_latest:
        return state
    latest = latest_run_dir_from_roots(logs_roots, prefix_override)
    if latest and latest != state.run_dir:
        return build_watch_state(
            latest,
            prefix_override,
            batch_mode,
            repo_root,
            logs_roots[0] if logs_roots else repo_root / "logs",
            window_seconds,
            watch,
            parallel_tasks,
        )
    return state


def token_rate_label(unit: str) -> str:
    return "tokens/sec" if unit == "sec" else "tokens/min"


def format_token_rate(value_per_min: Optional[float], unit: str) -> str:
    if value_per_min is None:
        return "?"
    if unit == "sec":
        return f"{value_per_min / 60.0:,.1f}"
    return f"{value_per_min:,.1f}"


def choose_tasks_done(
    completed_total: Optional[int],
    eval_count: Optional[int],
    completed_ok: Optional[int],
    total_tasks: Optional[int],
) -> Optional[int]:
    for candidate in (completed_total, eval_count, completed_ok, total_tasks):
        if candidate is not None:
            return candidate
    return None


def select_progress(metrics: List[AggregateMetrics]) -> Tuple[Optional[int], Optional[int]]:
    for item in metrics:
        if item.benchmark == "colbench":
            return item.tasks_done, item.tasks_total
    total_done = 0
    total_expected = 0
    done_known = False
    total_known = False
    for item in metrics:
        if item.tasks_done is not None:
            total_done += item.tasks_done
            done_known = True
        if item.tasks_total is not None:
            total_expected += item.tasks_total
            total_known = True
    return (total_done if done_known else None, total_expected if total_known else None)


def get_per_agent_rows(run_dir: Path, run_root: Path, repo_root: Path) -> List[Tuple[str, ...]]:
    prefix = load_prefix(run_dir)
    header = (
        "benchmark",
        "agent",
        "run_id",
        "completed",
        "evaluated",
        "total",
        "status",
    )
    rows = [header]

    for benchmark in BENCHMARKS:
        log_file = run_dir / f"{benchmark}.log"
        text = read_text(log_file)
        if not text:
            continue
        total_tasks = parse_total_tasks(text)
        run_root_override = parse_run_root(text)
        run_root_for_bench = Path(run_root_override) if run_root_override else run_root
        run_ids = filter_latest_run_ids(parse_run_ids(text))
        for run_id in run_ids:
            config_key = derive_config_key(run_id, prefix)
            run_path = resolve_run_dir(run_root_for_bench, repo_root, benchmark, run_id)
            raw_path = run_path / f"{run_id}_RAW_SUBMISSIONS.jsonl" if run_path else Path("")
            completed_ok, completed_err, completed_total = count_raw_submissions(raw_path)
            if benchmark == "scienceagentbench" and run_path:
                eval_count = count_eval_scienceagentbench(run_path, run_id)
            elif benchmark in ("scicode", "corebench") and run_path:
                eval_count = count_eval_from_upload(run_path, run_id)
            else:
                eval_count = completed_ok

            completed_str = "?"
            if completed_ok is not None:
                err_part = f", err={completed_err}" if completed_err is not None else ""
                entries_part = f", entries={completed_total}" if completed_total is not None else ""
                total_part = f"/{total_tasks}" if total_tasks is not None else ""
                completed_str = f"{completed_ok}{err_part}{entries_part}{total_part}"

            evaluated_str = "?"
            if eval_count is not None:
                total_part = f"/{total_tasks}" if total_tasks is not None else ""
                evaluated_str = f"{eval_count}{total_part}"

            total_str = str(total_tasks) if total_tasks is not None else "?"
            status = status_label(completed_ok, completed_err, total_tasks, eval_count)
            short_id = run_id if len(run_id) <= 32 else f"{run_id[:14]}..{run_id[-14:]}"
            rows.append((
                benchmark,
                config_key,
                short_id,
                completed_str,
                evaluated_str,
                total_str,
                status,
            ))

    return rows


def load_benchmark_config_count(repo_root: Path, benchmark: str) -> int:
    """Count number of models in model_to_baseline_<benchmark>.json."""
    config_path = repo_root / f"model_to_baseline_{benchmark}.json"
    if not config_path.exists():
        return 0
    try:
        data = json.loads(config_path.read_text())
        # Count keys that don't start with underscore (metadata)
        return sum(1 for k in data.keys() if not k.startswith("_"))
    except Exception:
        return 0


def find_latest_dataset(tmp_dir: Path, pattern: str) -> Optional[Path]:
    if not tmp_dir.exists():
        return None
    candidates = list(tmp_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_dataset_path_for_tasks(
    benchmark: str,
    repo_root: Path,
    run_root: Path,
    log_text: Optional[str] = None
) -> Optional[Path]:
    """Resolve dataset path to count tasks, handling custom/temp datasets."""
    
    # 1. Check env vars (standard HAL variables)
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

    # 2. Check log text for "Custom dataset: VAR=PATH"
    if log_text:
        env_var = env_map.get(benchmark)
        if env_var:
            pattern = re.compile(rf"{env_var}=([^\s]+)")
            matches = pattern.findall(log_text)
            if matches:
                path = Path(matches[-1])
                if path.exists():
                    return path

    # 3. Check temp directories for modified datasets (most common for fixes)
    # Search in both repo-local .hal_data/tmp and run-specific tmp
    tmp_dirs = [
        repo_root / ".hal_data" / "tmp",
        repo_root / ".tmp",
        run_root / "tmp"
    ]
    
    filename_patterns = {
        "scicode": "scicode_modified_*.json",
        "scienceagentbench": "scienceagentbench_modified_*.json",
        "corebench": "corebench_modified_*.json",
        "colbench": "colbench_modified_*.jsonl",
    }
    
    pattern = filename_patterns.get(benchmark)
    if pattern:
        for tmp_dir in tmp_dirs:
            path = find_latest_dataset(tmp_dir, pattern)
            if path:
                return path

    # 4. Fallback to default repo paths
    if benchmark == "corebench":
        default_path = repo_root / "hal-harness" / "hal" / "benchmarks" / "corebench" / "core_test.json"
        if default_path.exists():
            return default_path
    elif benchmark == "colbench":
        default_path = repo_root / "hal-harness" / "hal" / "benchmarks" / "colbench" / "data" / "backend_test.jsonl"
        if default_path.exists():
            return default_path
            
    return None


def count_tasks_in_dataset(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            return len(data)
    except Exception:
        return None


def collect_aggregate_metrics(run_dir: Path, run_root: Path, repo_root: Path) -> List[AggregateMetrics]:
    metrics: List[AggregateMetrics] = []

    for benchmark in BENCHMARKS:
        log_file = run_dir / f"{benchmark}.log"
        text = read_text(log_file)
        
        # Get expected runs from config file
        config_runs = load_benchmark_config_count(repo_root, benchmark)
        
        # 1. Try to find tasks per run from dataset file (most accurate for custom runs)
        tasks_per_run = None
        dataset_path = resolve_dataset_path_for_tasks(benchmark, repo_root, run_root, text)
        if dataset_path:
            tasks_per_run = count_tasks_in_dataset(dataset_path)

        # 2. Fallback: Parse from log text
        if tasks_per_run is None:
            tasks_per_run = parse_total_tasks(text)
        
        # 3. Fallback: Infer from raw submissions
        if tasks_per_run is None:
            run_ids = filter_latest_run_ids(parse_run_ids(text))
            for run_id in run_ids:
                run_root_override = parse_run_root(text)
                run_root_for_bench = Path(run_root_override) if run_root_override else run_root
                run_path = resolve_run_dir(run_root_for_bench, repo_root, benchmark, run_id)
                if run_path:
                    raw_path = run_path / f"{run_id}_RAW_SUBMISSIONS.jsonl"
                    _, _, completed_total = count_raw_submissions(raw_path)
                    if completed_total:
                        tasks_per_run = completed_total
                        break

        # Calculate total expected tasks based on config count
        if config_runs > 0 and tasks_per_run is not None:
            total_expected_tasks = config_runs * tasks_per_run
        else:
            total_expected_tasks = None

        if not text:
            # If log doesn't exist but we have config, report that
            if config_runs > 0:
                 metrics.append(AggregateMetrics(benchmark, config_runs, 0, total_expected_tasks, 0))
            continue

        run_ids = filter_latest_run_ids(parse_run_ids(text))
        
        tasks_done_sum = 0
        tasks_done_found = False
        tokens_total = 0

        # If we have run IDs in the log, we can count actual progress
        for run_id in run_ids:
            run_root_override = parse_run_root(text)
            run_root_for_bench = Path(run_root_override) if run_root_override else run_root
            run_path = resolve_run_dir(run_root_for_bench, repo_root, benchmark, run_id)
            if not run_path:
                continue

            raw_path = run_path / f"{run_id}_RAW_SUBMISSIONS.jsonl"
            completed_ok, completed_err, completed_total = count_raw_submissions(raw_path)
            if benchmark == "scienceagentbench":
                eval_count = count_eval_scienceagentbench(run_path, run_id)
            elif benchmark in ("scicode", "corebench"):
                eval_count = count_eval_from_upload(run_path, run_id)
            else:
                eval_count = completed_ok

            tasks_done = choose_tasks_done(completed_total, eval_count, completed_ok, tasks_per_run)
            if tasks_done is not None:
                tasks_done_sum += tasks_done
                tasks_done_found = True

            verbose_log = run_path / f"{run_id}_verbose.log"
            metrics_item = parse_verbose_metrics(verbose_log)
            tokens = metrics_item.tokens
            if tokens == 0:
                tokens = parse_local_trace_tokens(run_path)
            tokens_total += tokens

        # Use config_runs if available, otherwise fall back to observed runs
        final_runs = config_runs if config_runs > 0 else len(run_ids)
        
        # If we calculated a total based on config, use it. 
        # Otherwise sum from observed runs (fallback behavior).
        final_tasks_total = total_expected_tasks
        if final_tasks_total is None and len(run_ids) > 0:
             # Fallback: sum of tasks for observed runs
             final_tasks_total = (tasks_per_run * len(run_ids)) if tasks_per_run else None

        metrics.append(
            AggregateMetrics(
                benchmark=benchmark,
                runs=final_runs,
                tasks_done=tasks_done_sum if tasks_done_found else None,
                tasks_total=final_tasks_total,
                tokens_total=tokens_total,
            )
        )

    return metrics


def format_etc(tasks_done: Optional[int], tasks_total: Optional[int], tasks_per_min: Optional[float]) -> str:
    if tasks_done is None or tasks_total is None or tasks_per_min is None or tasks_per_min <= 0:
        return "?"
    remaining = tasks_total - tasks_done
    if remaining <= 0:
        return "0:00"
    total_seconds = (remaining / tasks_per_min) * 60
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    return f"{mins}:{secs:02d}"


def build_aggregate_rows(
    metrics: List[AggregateMetrics],
    rates: Dict[str, Tuple[Optional[float], Optional[float]]],
    token_rate_unit: str,
) -> List[Tuple[str, ...]]:
    header = (
        "benchmark",
        "runs",
        "tasks_done",
        "tasks_total",
        "tasks/min",
        "ETC",
        token_rate_label(token_rate_unit),
    )
    rows: List[Tuple[str, ...]] = [header]
    for item in metrics:
        tasks_rate, tokens_rate = rates.get(item.benchmark, (None, None))
        rows.append((
            item.benchmark,
            str(item.runs),
            format_int(item.tasks_done),
            format_int(item.tasks_total),
            format_rate(tasks_rate),
            format_etc(item.tasks_done, item.tasks_total, tasks_rate),
            format_token_rate(tokens_rate, token_rate_unit),
        ))
    return rows


def get_aggregate_rows(run_dir: Path, run_root: Path, repo_root: Path) -> List[Tuple[str, ...]]:
    metrics = collect_aggregate_metrics(run_dir, run_root, repo_root)
    return build_aggregate_rows(metrics, {}, "min")


def render_per_agent(run_dir: Path, run_root: Path, repo_root: Path) -> None:
    print_table(get_per_agent_rows(run_dir, run_root, repo_root))


def render_aggregate(run_dir: Path, run_root: Path, repo_root: Path) -> None:
    print_table(get_aggregate_rows(run_dir, run_root, repo_root))


def build_output_lines(
    run_dir: Path,
    run_root: Path,
    repo_root: Path,
    per_agent: bool,
    rate_tracker: Optional[RateTracker],
    window_seconds: int,
    token_rate_unit: str,
    state: WatchState,
) -> List[str]:
    rows: List[Tuple[str, ...]]
    metrics: List[AggregateMetrics] = []
    if per_agent:
        rows = get_per_agent_rows(run_dir, run_root, repo_root)
        metrics = collect_aggregate_metrics(run_dir, run_root, repo_root)
    else:
        metrics = collect_aggregate_metrics(run_dir, run_root, repo_root)
        rates = rate_tracker.update(metrics) if rate_tracker else {}
        rows = build_aggregate_rows(metrics, rates, token_rate_unit)
    header = f"Run: {run_dir}"
    if not per_agent:
        header = f"{header} | window={format_window(window_seconds)}"
    lines = [header]
    if state.current_prefix or state.next_prefix:
        lines.append(f"Prefix: {state.current_prefix or '?'} -> {state.next_prefix or '?'}")
    else:
        lines.append("Prefix: ? (pass --prefix to set the current prefix)")
    if state.auto_command:
        lines.append(f"Next command: {state.auto_command}")
    else:
        lines.append("Next command: (set --prefix to compute next run)")
    tasks_done, tasks_total = select_progress(metrics)
    state.auto_tracker.update(tasks_done, tasks_total)
    lines.append(state.auto_tracker.status_line(tasks_done, tasks_total, repo_root))
    lines.extend(build_table(rows))
    return lines


def run_tui(
    run_dir: Path,
    run_root: Path,
    repo_root: Path,
    per_agent: bool,
    interval: int,
    window_seconds: int,
    token_rate_unit: str,
    state: WatchState,
    follow_latest: bool,
    prefix_override: Optional[str],
    batch_mode: bool,
    logs_roots: List[Path],
) -> None:
    import curses

    def _loop(screen: "curses._CursesWindow") -> None:
        nonlocal state
        curses.curs_set(0)
        screen.nodelay(True)
        screen.timeout(max(100, interval * 1000))
        while True:
            state = maybe_refresh_state(
                state,
                follow_latest,
                prefix_override,
                batch_mode,
                repo_root,
                logs_roots,
                window_seconds,
                watch=True,
            )
            lines = build_output_lines(
                state.run_dir,
                run_root,
                repo_root,
                per_agent,
                state.rate_tracker,
                window_seconds,
                token_rate_unit,
                state,
            )
            height, width = screen.getmaxyx()
            screen.erase()
            max_lines = max(0, height - 1)
            for idx, line in enumerate(lines[:max_lines]):
                screen.addnstr(idx, 0, line, max(0, width - 1))
            hint = "Press q to quit"
            if height > 0:
                screen.addnstr(height - 1, 0, hint, max(0, width - 1))
            screen.refresh()
            ch = screen.getch()
            if ch in (ord("q"), ord("Q")):
                break

    curses.wrapper(_loop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track aggregate progress for each benchmark run.")
    parser.add_argument("--run-dir", help="Path to logs/benchmark_run_* directory.")
    parser.add_argument("--per-agent", action="store_true", help="Show per-agent rows instead of aggregate benchmark totals.")
    parser.add_argument("--watch", action="store_true", help="Refresh every few seconds.")
    parser.add_argument("--tui", action="store_true", help="Force curses TUI for smoother updates.")
    parser.add_argument("--no-tui", action="store_true", help="Disable curses TUI even if available.")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds.")
    parser.add_argument("--window-seconds", type=int, default=300, help="Rate window in seconds (default: 300).")
    parser.add_argument(
        "--token-rate-unit",
        choices=("min", "sec"),
        default="min",
        help="Display token rate per minute or per second (default: min).",
    )
    parser.add_argument("--batch-mode", action="store_true", help="Auto-start the next colbench run if progress stalls or completes.")
    parser.add_argument("--prefix", help="Current run prefix (e.g., sun12_). Used to compute the next prefix.")
    parser.add_argument("--parallel-tasks", type=int, default=50, help="Number of parallel tasks for auto-relaunch (default: 50).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    run_root = detect_run_root(script_dir)
    repo_root = script_dir

    logs_dir = run_root / "logs"
    logs_roots = collect_logs_roots(script_dir, run_root)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir_from_roots(logs_roots, args.prefix)
    if not run_dir or not run_dir.exists():
        print("No benchmark_run_* directory found.")
        return

    follow_latest = args.run_dir is None
    state = build_watch_state(
        run_dir,
        args.prefix,
        args.batch_mode,
        repo_root,
        logs_dir,
        args.window_seconds,
        args.watch,
        args.parallel_tasks,
    )

    use_tui = False
    if args.watch and sys.stdout.isatty() and not args.no_tui:
        use_tui = True
    if args.tui:
        use_tui = True

    if use_tui:
        try:
            run_tui(
                state.run_dir,
                run_root,
                repo_root,
                args.per_agent,
                args.interval,
                args.window_seconds,
                args.token_rate_unit,
                state,
                follow_latest,
                args.prefix,
                args.batch_mode,
                logs_roots,
            )
            return
        except Exception:
            use_tui = False

    while True:
        state = maybe_refresh_state(
            state,
            follow_latest,
            args.prefix,
            args.batch_mode,
            repo_root,
            logs_roots,
            args.window_seconds,
            args.watch,
            args.parallel_tasks,
        )
        lines = build_output_lines(
            state.run_dir,
            run_root,
            repo_root,
            args.per_agent,
            state.rate_tracker,
            args.window_seconds,
            args.token_rate_unit,
            state,
        )
        
        # Clear screen and print immediately to minimize flicker
        sys.stdout.write("\033[H\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        
        if not args.watch:
            break
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    main()
