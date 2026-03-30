#!/usr/bin/env python3
"""
Shared helpers for resolving the canonical result directories.
"""

from __future__ import annotations

import os
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
RESULT_ROOT = MODEL_DIR / "result"
DEFAULT_MAIN_RESULT_DIR = RESULT_ROOT / "main"


def result_root_dir() -> Path:
    return RESULT_ROOT


def configured_main_result_dir() -> Path:
    override = os.environ.get("AGENT_EVAL_MAIN_RESULT_DIR")
    return Path(override).expanduser() if override else DEFAULT_MAIN_RESULT_DIR


def has_main_results(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.glob("amortized_irt_*.csv"))


def main_result_dir() -> Path:
    preferred = configured_main_result_dir()
    if has_main_results(preferred):
        return preferred

    legacy = RESULT_ROOT
    if has_main_results(legacy):
        return legacy

    return preferred


def ensure_main_result_dir() -> Path:
    path = configured_main_result_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def main_baseline_dir() -> Path:
    return main_result_dir() / "baselines"
