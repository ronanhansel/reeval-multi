#!/usr/bin/env python3
"""
Docent-based rubric evaluation script (primary method).

Evaluates agent traces using benchmark-specific rubrics with:
- SQLite LLM response caching (no repeat API calls)
- Dynamic batch processing (by message count)
- Turn-by-turn conversation deduplication
- Support for multiple benchmarks via rubric templates
- Direct Azure/TRAPI access by default (no proxy needed)

Usage:
    # Default: Uses Azure/TRAPI directly (recommended)
    python scripts/eval_rubric.py \
        --trace-file traces/colbench_*_binary_UPLOAD.json \
        --rubric rubric_templates/colbench.txt \
        --rubric-model openai:gpt-5.2 \
        --failed-only -y

    # With proxy/custom endpoint (overrides Azure default)
    python scripts/eval_rubric.py \
        --trace-file traces/*.json \
        --rubric rubric_templates/scicode.txt \
        --rubric-model openai:gpt-4o \
        --openai-base-url "http://localhost:4000/v1,http://localhost:4001/v1" \
        --failed-only -y

    # Preview mode (stdout, limited tasks)
    python scripts/eval_rubric.py \
        --trace-file traces/scicode_*.json \
        --rubric rubric_templates/scicode.txt \
        --rubric-model openai:gpt-4o \
        --output-mode stdout \
        --max-tasks 3 -y

Output:
    CSV files go to rubrics_output/<rubric_name>/<trace_name>.csv
    Example: rubrics_output/colbench/colbench_backend_gpt41_binary.csv

See PIPELINE_README.md for full documentation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Sequence

try:
    import dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dotenv = None

# TRAPI deployment name mapping (from litellm.trapi.yaml)
TRAPI_DEPLOYMENT_MAP = {
    # GPT-5 series (NOTE: gpt-5 uses max_completion_tokens like o-series)
    'gpt-5': 'gpt-5_2025-08-07',
    'gpt-5-mini': 'gpt-5-mini_2025-08-07',
    'gpt-5-nano': 'gpt-5-nano_2025-08-07',
    'gpt-5-pro': 'gpt-5-pro_2025-10-06',
    'gpt-5.2': 'gpt-5.2_2025-12-11',
    'gpt-5.2-chat': 'gpt-5.2-chat_2025-12-11',

    # GPT-4 series
    'gpt-4o': 'gpt-4o_2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini_2024-07-18',
    'gpt-4.1': 'gpt-4.1_2025-04-14',
    'gpt-4.1-mini': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano_2025-04-14',
    'gpt-4-turbo': 'gpt-4_turbo-2024-04-09',
    'gpt-4-32k': 'gpt-4-32k_0613',
    'gpt-4': 'gpt-4_turbo-2024-04-09',

    # O-series (reasoning models)
    'o1': 'o1_2024-12-17',
    'o1-mini': 'o1-mini_2024-09-12',
    'o3': 'o3_2025-04-16',
    'o3-mini': 'o3-mini_2025-01-31',
    'o4-mini': 'o4-mini_2025-04-16',

    # GPT-5.1 series
    'gpt-5.1': 'gpt-5.1_2025-11-13',
    'gpt-5.1-chat': 'gpt-5.1-chat_2025-11-13',
    'gpt-5.1-codex': 'gpt-5.1-codex_2025-11-13',
    'gpt-5.1-codex-mini': 'gpt-5.1-codex-mini_2025-11-13',

    # Other models
    'grok-3.1': 'grok-3_1',
    'llama-3.3': 'gcr-llama-33-70b-shared',
    'llama-3.1-70b': 'gcr-llama-31-70b-shared',
    'llama-3.1-8b': 'gcr-llama-31-8b-instruct',
    'qwen3-8b': 'gcr-qwen3-8b',
    'phi4': 'gcr-phi-4-shared',
    'mistral': 'gcr-mistralai-8x7b-shared',
    'deepseek-r1': 'deepseek-r1_1',
    'deepseek': 'deepseek-r1_1',
}

# Azure CLI's public client ID (used for MSAL token refresh)
AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'


def resolve_trapi_deployment(model: str) -> str:
    """Resolve friendly model name to TRAPI deployment name."""
    model = model.replace('azure/', '').replace('openai/', '').replace('openai:', '')
    if model in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model]
    model_lower = model.lower()
    if model_lower in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model_lower]
    for key, value in TRAPI_DEPLOYMENT_MAP.items():
        if key in model_lower or model_lower in key:
            return value
    return model  # Return as-is if no mapping found


def get_azure_token(scope: str = 'api://trapi/.default') -> str | None:
    """Get Azure AD token using MSAL or azure-identity."""
    # Try MSAL first (works without az CLI installed)
    try:
        import msal
        cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        if os.path.exists(cache_path):
            cache = msal.SerializableTokenCache()
            with open(cache_path, 'r') as f:
                cache.deserialize(f.read())
            app = msal.PublicClientApplication(
                AZURE_CLI_CLIENT_ID,
                authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
                token_cache=cache
            )
            accounts = app.get_accounts()
            if accounts:
                result = app.acquire_token_silent([scope], account=accounts[0])
                if result and 'access_token' in result:
                    print("[Azure] Using MSAL token (dynamic refresh)")
                    return result['access_token']
    except ImportError:
        pass
    except Exception as e:
        print(f"[Azure] MSAL token refresh failed: {e}")

    # Try azure-identity as fallback
    try:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, scope)
        token = token_provider()
        print("[Azure] Using azure-identity token")
        return token
    except ImportError:
        pass
    except Exception as e:
        print(f"[Azure] azure-identity failed: {e}")

    return None


def setup_azure_environment(rubric_model: str | None = None) -> bool:
    """Set up environment for direct Azure/TRAPI access. Returns True if successful."""
    endpoint = os.environ.get('TRAPI_ENDPOINT', 'https://trapi.research.microsoft.com/gcr/shared')
    # Use 2025-03-01-preview for gpt-5.2 and newer models compatibility
    api_version = os.environ.get('TRAPI_API_VERSION', '2025-03-01-preview')
    scope = os.environ.get('TRAPI_SCOPE', 'api://trapi/.default')

    token = get_azure_token(scope)
    if not token:
        print("[Azure] Could not obtain Azure AD token. Falling back to proxy.")
        return False

    # Set OpenAI environment variables for direct Azure access
    # The base URL format for Azure OpenAI compatible endpoint
    os.environ["OPENAI_BASE_URL"] = f"{endpoint}/openai"
    os.environ["OPENAI_API_KEY"] = token
    os.environ["OPENAI_API_VERSION"] = api_version

    # Also set Azure-specific vars for azure_openai provider compatibility
    os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
    os.environ["AZURE_OPENAI_API_KEY"] = token
    os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    print(f"[Azure] Direct TRAPI access configured: {endpoint}")
    return True


# Define REPO_ROOT before using it for config loading
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script"))
sys.path.insert(0, str(REPO_ROOT / "script" / "utils"))

# Pre-parse --openai-base-url BEFORE rubric evaluation setup
# (some modules may read OPENAI_BASE_URL at import time)
# If not provided, use Azure/TRAPI directly
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--openai-base-url", type=str, default=None)
_pre_parser.add_argument("--rubric-model", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

_using_azure_direct = False
_resolved_model = None

# Load model rubrics config
try:
    with open(REPO_ROOT / "models" / "model_rubrics.json") as f:
        _rubric_config = json.load(f)
except Exception as e:
    print(f"Warning: Could not load models/model_rubrics.json: {e}")
    _rubric_config = {}

# Determine model to use (default to gpt-5.2 if not specified)
_target_model_key = "gpt-5.2"
if _pre_args.rubric_model:
    # If user specified a model, try to find it in config, otherwise use as-is
    if _pre_args.rubric_model in _rubric_config:
        _target_model_key = _pre_args.rubric_model

if _pre_args.openai_base_url is None:
    # No proxy URL provided - use Azure/TRAPI directly via config if available
    _model_info = _rubric_config.get(_target_model_key)
    
    if _model_info:
        # Use config-based setup
        _using_azure_direct = setup_azure_environment(_target_model_key)
        if _using_azure_direct:
            base_urls = _model_info.get("available_base_urls", [])
            if base_urls:
                # Set primary URL
                os.environ["OPENAI_BASE_URL"] = f"{base_urls[0]}/openai"
                os.environ["AZURE_OPENAI_ENDPOINT"] = base_urls[0]
                
                # Set fallback URLs for rotation/retry
                if len(base_urls) > 1:
                    fallback_formatted = [f"{url}/openai" for url in base_urls]
                    os.environ["OPENAI_FALLBACK_URLS"] = ",".join(fallback_formatted)
                    print(f"[Azure] Configured {len(base_urls)} URLs for rotation/fallback")
            
            # Resolve model ID from config (e.g. "openai/gpt-5.2..." -> "azure_openai:gpt-5.2...")
            raw_id = _model_info.get("model_id", _target_model_key)
            if ":" in raw_id:
                _resolved_model = raw_id
            else:
                # Map standard ID to azure_openai provider
                # e.g. openai/gpt-5.2_2025-12-11 -> azure_openai:gpt-5.2_2025-12-11
                clean_id = raw_id.replace("openai/", "").replace("azure/", "")
                _resolved_model = f"azure_openai:{clean_id}"
            
            print(f"[Azure] Model configured from json: {_target_model_key} -> {_resolved_model}")
            
    # Fallback to old logic if config not found or setup failed
    if not _resolved_model:
        _using_azure_direct = setup_azure_environment(_pre_args.rubric_model)
        if _using_azure_direct and _pre_args.rubric_model:
            # Resolve model name to TRAPI deployment name AND switch to azure_openai provider
            # The azure_openai provider uses AsyncAzureOpenAI which formats URLs correctly
            if ':' in _pre_args.rubric_model:
                provider, model_name = _pre_args.rubric_model.split(':', 1)
                deployment_name = resolve_trapi_deployment(model_name)
                # CRITICAL: Use azure_openai provider instead of openai
                # openai provider uses wrong URL format for TRAPI
                _resolved_model = f"azure_openai:{deployment_name}"
                print(f"[Azure] Model resolved: {_pre_args.rubric_model} -> {_resolved_model}")
            else:
                deployment_name = resolve_trapi_deployment(_pre_args.rubric_model)
                _resolved_model = f"azure_openai:{deployment_name}"
                print(f"[Azure] Model resolved: {_pre_args.rubric_model} -> {_resolved_model}")
    
    if not _using_azure_direct:
        # Fallback to localhost proxy
        os.environ["OPENAI_BASE_URL"] = "http://localhost:4000/v1"
        os.environ["OPENAI_FALLBACK_URLS"] = "http://localhost:4000/v1"
else:
    # Proxy URL provided - use it (keep original provider)
    _all_urls = [u.strip() for u in _pre_args.openai_base_url.split(",")]
    os.environ["OPENAI_BASE_URL"] = _all_urls[0]
    os.environ["OPENAI_FALLBACK_URLS"] = ",".join(_all_urls)
    print(f"[Proxy] Using custom endpoint: {_all_urls[0]}")

if dotenv:
    dotenv.load_dotenv()

SCAFFOLD_TAGS = {
    "<start_code>",
    "<end_code>",
    "<start_plan>",
    "<end_plan>",
    "<start_thought>",
    "<end_thought>",
    "<start_solution>",
    "<end_solution>",
}
ENVIRONMENTAL_BARRIER_DESCRIPTION = (
    "An Environmental Barrier in SWE-bench describes a failure mode where an agent is prevented from solving a task "
    "due to impassable infrastructure faults rather than a lack of coding capability. These barriers arise from defects "
    "in the evaluation setup itself—such as crashing Docker containers, broken shell environments, or pre-existing "
    "dependency conflicts—that render the codebase unrunnable regardless of the agent's actions. Unlike a capability "
    "failure where an agent writes incorrect code, an environmental barrier effectively blocks the agent from even "
    "attempting the task, often due to missing files or system-level restrictions that are outside the agent's control."
)


def _ensure_llm_cache_path() -> None:
    """Docent expects LLM_CACHE_PATH; default to result/.hal_data/.llm_cache if not provided."""
    if os.getenv("LLM_CACHE_PATH"):
        return
    default_cache = Path("result/.hal_data/.llm_cache")
    default_cache.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_CACHE_PATH"] = str(default_cache.resolve())


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


os.environ.setdefault("ENV_RESOLUTION_STRATEGY", "os_environ")
_ensure_llm_cache_path()

DOCENT_REPO_PATH = Path(__file__).resolve().parent.parent / "docent"
if DOCENT_REPO_PATH.exists():
    sys.path.insert(0, str(DOCENT_REPO_PATH))
    # Also add nested docent/docent path for the docent package itself
    nested_docent = DOCENT_REPO_PATH / "docent"
    if nested_docent.exists():
        sys.path.insert(0, str(nested_docent))

DOCENT_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - optional dependency
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message
    from docent_core.docent.ai_tools.rubric.rubric import Rubric, evaluate_rubric
    from docent_core._llm_util.providers.preferences import ModelOption
except Exception as exc:  # pragma: no cover - optional dependency
    DOCENT_IMPORT_ERROR = exc
    AgentRun = None  # type: ignore[assignment]
    Transcript = None  # type: ignore[assignment]
    parse_chat_message = None  # type: ignore[assignment]
    Rubric = None  # type: ignore[assignment]
    ModelOption = None  # type: ignore[assignment]
    evaluate_rubric = None  # type: ignore[assignment]

GLOBAL_JSON_REQUIREMENTS = dedent(
    """
    JSON response requirements:
    - Respond with only the JSON object that matches the schema (no prose, code fences, or leading text).
    - Escape newline characters as \\n; do not emit raw control characters inside JSON strings.
    """
).strip()

DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "enum": [0, 1],
            "description": "Binary score: use 1 only when the rubric is satisfied, else 0.",
        },
        "explanation": {
            "type": "string",
            "citations": True,
            "description": "Concise justification referencing transcript blocks.",
        },
    },
    "required": ["score", "explanation"],
}

DEFAULT_RUBRIC_TEXT = dedent(
    f"""
    Rubric: Environmental Barrier

    Detection target:
    {ENVIRONMENTAL_BARRIER_DESCRIPTION}

    Label the rubric as "present" when the transcript contains clear evidence that the agent was blocked by these
    infrastructural faults. Label it as "absent" when failures stem from the agent's own reasoning, planning, or code,
    or when no convincing evidence of an environmental issue appears.

    Evidence guidelines:
    - Presence indicators (label as present):
      • Sandbox or container crashes, unrecoverable kernel panics, or persistent harness errors unrelated to the agent's code.
      • Missing files, permissions, or dependencies that existed before the agent took any action.
      • System-level restrictions (e.g., read-only filesystems, broken shell environments) that halt progress for any agent.
    - Absence indicators (label as absent):
      • Errors caused by malformed patches, wrong file edits, or logical mistakes.
      • Tool or command failures that follow from the agent's own incorrect inputs.
      • Speculative or insufficient evidence; if uncertain, default to absent.

    Explanation requirements:
    - Reference the specific transcript blocks or tool outputs that justify the classification.
    - Highlight both the failure symptoms and why they originate from the environment (or why they do not).
    - Keep explanations concise and cite block IDs directly.

    {GLOBAL_JSON_REQUIREMENTS}
    """
).strip()

DEFAULT_RUBRIC_PROVIDER = os.getenv("DOCENT_RUBRIC_PROVIDER", "azure_openai")
DEFAULT_RUBRIC_BATCH_SIZE = int(os.getenv("DOCENT_RUBRIC_BATCH_SIZE", "4"))


@dataclass
class TraceMessage:
    """Normalized representation of a single message in a task trace."""

    role: str
    content: str
    entry_id: str | None
    timestamp: str | None


@dataclass
class TaskConversation:
    """Aggregated conversation for one SWE-bench task."""

    task_id: str
    entries: list[dict[str, Any]]
    messages: list[TraceMessage]

    @property
    def entry_count(self) -> int:
        return len(self.entries)


@dataclass
class LocalRubricEvaluation:
    """Container for rubric results produced by Docent."""

    task_id: str
    rubric_id: str
    rubric_version: int
    output: dict[str, Any] | None
    error: str | None = None

    @property
    def score(self) -> float | None:
        if not self.output:
            return None
        score = self.output.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    @property
    def explanation(self) -> str:
        if not self.output:
            return ""
        # Try standard explanation field first
        explanation = self.output.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            return explanation
        # Build explanation from custom schema fields (scicode rubric)
        parts = []
        for field in ("existence_reasoning", "causation_reasoning", "evidence"):
            val = self.output.get(field)
            if isinstance(val, str) and val.strip():
                parts.append(f"{field}: {val.strip()}")
        if parts:
            return " | ".join(parts)
        # Fallback: serialize entire output as explanation
        return str(self.output)


@dataclass
class RubricDefinition:
    rubric_id: str
    rubric_text: str
    output_schema: dict[str, Any]


def ensure_default_rubric_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    txt_files = list(directory.glob("*.txt"))
    if txt_files:
        return

    default_path = directory / "environmentalbarrier.txt"
    if not default_path.exists():
        default_path.write_text(DEFAULT_RUBRIC_TEXT, encoding="utf-8")


def _load_schema(file_path: Path, rubrics_dir: Path | None = None) -> dict[str, Any]:
    """Load schema for a rubric, checking benchmark-specific then unified schema."""
    # First try benchmark-specific schema (e.g., swebench.schema.json)
    schema_path = file_path.with_suffix(".schema.json")
    if schema_path.exists():
        try:
            return json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"⚠️  Failed to parse schema for {file_path.name}: {exc}.")

    # Then try unified schema in the same directory
    unified_schema_path = file_path.parent / "rubric.schema.json"
    if unified_schema_path.exists():
        try:
            return json.loads(unified_schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"⚠️  Failed to parse unified schema: {exc}.")

    # Then try unified schema in rubrics_dir if different
    if rubrics_dir and rubrics_dir != file_path.parent:
        unified_schema_path = rubrics_dir / "rubric.schema.json"
        if unified_schema_path.exists():
            try:
                return json.loads(unified_schema_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"⚠️  Failed to parse unified schema: {exc}.")

    # Fall back to default
    return DEFAULT_OUTPUT_SCHEMA


def load_single_rubric(file_path: Path, rubrics_dir: Path | None = None) -> RubricDefinition | None:
    """Load a single rubric definition from a .txt file."""
    if not file_path.exists():
        return None
    rubric_text = file_path.read_text(encoding="utf-8").strip()
    if not rubric_text:
        return None
    if GLOBAL_JSON_REQUIREMENTS not in rubric_text:
        rubric_text = f"{rubric_text.rstrip()}\n\n{GLOBAL_JSON_REQUIREMENTS}"
    output_schema = _load_schema(file_path, rubrics_dir)
    rubric_id = _slugify(file_path.stem.lower())
    return RubricDefinition(
        rubric_id=rubric_id,
        rubric_text=rubric_text,
        output_schema=output_schema,
    )


def load_rubric_definitions(directory: Path) -> list[RubricDefinition]:
    ensure_default_rubric_files(directory)
    definitions: list[RubricDefinition] = []
    for txt_path in sorted(directory.glob("*.txt")):
        rubric_text = txt_path.read_text(encoding="utf-8").strip()
        if not rubric_text:
            continue
        if GLOBAL_JSON_REQUIREMENTS not in rubric_text:
            rubric_text = f"{rubric_text.rstrip()}\n\n{GLOBAL_JSON_REQUIREMENTS}"
        output_schema = _load_schema(txt_path, directory)
        rubric_id = _slugify(txt_path.stem.lower())
        definitions.append(
            RubricDefinition(
                rubric_id=rubric_id,
                rubric_text=rubric_text,
                output_schema=output_schema,
            )
        )
    return definitions


def resolve_trace_path(trace_file: str | None, trace_dir: Path) -> Path:
    if trace_file:
        candidate = Path(trace_file).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Trace file not found: {candidate}")
        return candidate

    trace_files = sorted(trace_dir.expanduser().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trace_files:
        raise FileNotFoundError(f"No *.json trace files found in {trace_dir}")
    return trace_files[0]


def load_trace_file(trace_path: Path) -> dict[str, Any]:
    """Load the selected JSON trace file."""
    print(f"📂 Loading trace data from {trace_path} ...")
    with trace_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_overview(data: dict[str, Any]) -> None:
    """Print a concise overview of the dataset/config metadata."""
    config = data.get("config", {})
    results = data.get("results", {})

    agent_name = config.get("agent_name", "unknown-agent")
    model_name = config.get("agent_args", {}).get("model_name", "unknown-model")
    benchmark = config.get("benchmark_name", "unknown-benchmark")

    print("\n📊 Dataset Information:")
    print(f"   Agent: {agent_name}")
    print(f"   Model: {model_name}")
    print(f"   Benchmark: {benchmark}")

    accuracy = results.get("accuracy")
    total_cost = results.get("total_cost")
    failed_tasks = results.get("failed_tasks", [])
    successful_tasks = results.get("successful_tasks", [])

    print("\n📈 Results Summary:")
    if accuracy is not None:
        print(f"   Accuracy: {accuracy:.1%}")
    if total_cost is not None:
        print(f"   Total Cost: ${total_cost:.2f}")
    print(f"   Successful Tasks: {len(successful_tasks)}")
    print(f"   Failed Tasks: {len(failed_tasks)}")


def group_entries_by_task(raw_entries: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group raw logging entries by their weave_task_id or task_id (ColBench)."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    for entry in raw_entries:
        task_id = (
            entry.get("task_id")  # ColBench format: {"task_id": "1", "dialogue_history": [...]}
            or entry.get("attributes", {}).get("weave_task_id")
            or entry.get("weave_task_id")
            or entry.get("inputs", {}).get("task_id")
            or "unknown"
        )
        tasks.setdefault(task_id, []).append(entry)
    return tasks


def sanitize_text(text: str) -> str:
    """Strip agent scaffolding tags and trim whitespace."""
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if line.strip() in SCAFFOLD_TAGS:
            continue
        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines).strip()


def normalize_content(content: Any) -> str:
    """Convert OpenAI-style message content into a simple string."""
    if isinstance(content, str):
        return sanitize_text(content.strip())

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "reasoning":
                # reasoned content may include summaries we can surface
                summary = block.get("summary")
                if summary:
                    parts.append(summary)
                elif block.get("redacted") is False:
                    parts.append(block.get("reasoning", ""))
        raw_text = "\n".join(part for part in (p.strip() for p in parts) if part)
        return sanitize_text(raw_text)

    if content is None:
        return ""

    return sanitize_text(str(content).strip())


def entry_timestamp(entry: dict[str, Any]) -> str | None:
    """Best-effort timestamp for ordering entries."""
    return entry.get("created_timestamp") or entry.get("started_at") or entry.get("ended_at")


def content_fingerprint(text: str) -> str:
    """Normalize content for deduplication comparison."""
    # Collapse all whitespace to single spaces and strip
    normalized = " ".join(text.split()).strip()
    # Use first 500 chars for fingerprint to handle slight trailing differences
    return normalized[:500]


def is_duplicate_message(existing: "TraceMessage", new: "TraceMessage") -> bool:
    """Check if new message is a duplicate of existing (same role, similar content)."""
    if existing.role != new.role:
        return False
    # Exact match
    if existing.content == new.content:
        return True
    # Normalized match (handles whitespace differences)
    return content_fingerprint(existing.content) == content_fingerprint(new.content)


def sort_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort entries chronologically using available timestamps."""
    return sorted(entries, key=lambda e: (entry_timestamp(e) or "", e.get("id", "")))


def build_trace_message(raw_message: dict[str, Any] | list | Any, entry: dict[str, Any]) -> TraceMessage | None:
    """Normalize a raw OpenAI message dict into TraceMessage.

    Handles multiple formats:
    - Standard OpenAI format: {"role": "user", "content": "..."}
    - LangChain serialization format: [{"lc": 1, "type": "constructor", "id": [...], "kwargs": {"content": "..."}}]
    """
    # Handle LangChain serialization format (list containing constructor dict)
    if isinstance(raw_message, list):
        if raw_message and isinstance(raw_message[0], dict):
            lc_msg = raw_message[0]
            if lc_msg.get("type") == "constructor" and "kwargs" in lc_msg:
                # Extract role from id path (e.g., ["langchain", "schema", "messages", "SystemMessage"])
                msg_id = lc_msg.get("id", [])
                role = None
                if msg_id and len(msg_id) >= 4:
                    msg_type = msg_id[-1].lower()
                    if "system" in msg_type:
                        role = "system"
                    elif "human" in msg_type or "user" in msg_type:
                        role = "user"
                    elif "ai" in msg_type or "assistant" in msg_type:
                        role = "assistant"
                    elif "tool" in msg_type or "function" in msg_type:
                        role = "tool"

                if role:
                    content = normalize_content(lc_msg.get("kwargs", {}).get("content"))
                    if content:
                        return TraceMessage(role=role, content=content, entry_id=entry.get("id"), timestamp=entry_timestamp(entry))
        return None

    # Must be a dict for standard format
    if not isinstance(raw_message, dict):
        return None

    role = raw_message.get("role")
    if not role:
        return None

    content = normalize_content(raw_message.get("content"))
    if not content:
        return None

    return TraceMessage(role=role, content=content, entry_id=entry.get("id"), timestamp=entry_timestamp(entry))


def build_assistant_message(entry: dict[str, Any]) -> TraceMessage | None:
    """Extract the assistant response from the entry output."""
    output = entry.get("output") or {}
    choices = output.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message")
    if not message:
        return None

    role = message.get("role", "assistant")
    content = normalize_content(message.get("content"))
    if not content:
        return None

    return TraceMessage(role=role, content=content, entry_id=entry.get("id"), timestamp=entry_timestamp(entry))


def extract_task_messages(task_id: str, entries: list[dict[str, Any]]) -> TaskConversation:
    """Convert task-specific entries into an ordered conversation with deduplication."""
    # Special handling for ColBench format: entries have dialogue_history directly
    if len(entries) == 1 and "dialogue_history" in entries[0]:
        colbench_entry = entries[0]
        dialogue = colbench_entry.get("dialogue_history", [])
        conversation: list[TraceMessage] = []
        for idx, msg in enumerate(dialogue):
            role = msg.get("role", "user")
            content = normalize_content(msg.get("content", ""))
            if content:
                conversation.append(TraceMessage(
                    role=role,
                    content=content,
                    entry_id=f"colbench_{task_id}_{idx}",
                    timestamp=None
                ))
        return TaskConversation(task_id=task_id, entries=entries, messages=conversation)

    # Standard Weave format processing
    ordered_entries = sort_entries(entries)
    conversation: list[TraceMessage] = []
    previous_message_count = 0
    # Track recent content fingerprints to catch duplicates even if not consecutive
    seen_fingerprints: set[tuple[str, str]] = set()  # (role, fingerprint)

    def is_seen_or_duplicate(msg: TraceMessage) -> bool:
        """Check if message is duplicate of any recent message."""
        fp = (msg.role, content_fingerprint(msg.content))
        if fp in seen_fingerprints:
            return True
        # Also check against last message for same-role consecutive duplicates
        if conversation and is_duplicate_message(conversation[-1], msg):
            return True
        return False

    def add_message(msg: TraceMessage) -> None:
        """Add message if not duplicate."""
        if is_seen_or_duplicate(msg):
            return
        fp = (msg.role, content_fingerprint(msg.content))
        seen_fingerprints.add(fp)
        conversation.append(msg)

    for entry in ordered_entries:
        raw_messages = entry.get("inputs", {}).get("messages") or []

        if len(raw_messages) < previous_message_count:
            previous_message_count = 0

        if len(raw_messages) > previous_message_count:
            new_messages = raw_messages[previous_message_count:]
            for raw in new_messages:
                normalized = build_trace_message(raw, entry)
                if normalized:
                    add_message(normalized)

        previous_message_count = len(raw_messages)

        # Add assistant output from entry
        assistant_message = build_assistant_message(entry)
        if assistant_message:
            add_message(assistant_message)

    return TaskConversation(task_id=task_id, entries=ordered_entries, messages=conversation)


def resolve_rubric_model_option(
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> "ModelOption":
    """Derive the Docent ModelOption to use for rubric evaluation."""
    if ModelOption is None:
        raise RuntimeError("Docent ModelOption class is unavailable. Ensure docent is installed.")

    raw_model = model_override or os.getenv("DOCENT_RUBRIC_MODEL")
    provider = DEFAULT_RUBRIC_PROVIDER
    model_name = None

    if raw_model:
        if ":" in raw_model:
            provider, model_name = raw_model.split(":", 1)
        else:
            model_name = raw_model

    if not model_name:
        model_name = os.getenv("DOCENT_RUBRIC_MODEL_NAME")

    if not model_name and provider == "azure_openai":
        for candidate in (
            "AZURE_OPENAI_RUBRIC_MODEL",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
        ):
            value = os.getenv(candidate)
            if value:
                model_name = value
                break

    if not model_name:
        raise ValueError(
            "Unable to determine an Azure OpenAI deployment for rubric evaluation. "
            "Set DOCENT_RUBRIC_MODEL (provider:model) or DOCENT_RUBRIC_MODEL_NAME / "
            "AZURE_OPENAI_DEPLOYMENT_NAME in your environment."
        )

    reasoning_effort = os.getenv("DOCENT_RUBRIC_REASONING_EFFORT")
    if reasoning_effort:
        reasoning_effort = reasoning_effort.lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "DOCENT_RUBRIC_REASONING_EFFORT must be one of: low, medium, high."
            )

    return ModelOption(
        provider=provider,
        model_name=model_name,
        reasoning_effort=(reasoning_effort_override or reasoning_effort),  # type: ignore[arg-type]
    )


def build_rubric_from_definition(
    definition: RubricDefinition,
    model_option: "ModelOption",
) -> "Rubric":
    if Rubric is None:
        raise RuntimeError("Docent Rubric class is unavailable. Ensure docent is installed.")

    return Rubric(
        id=definition.rubric_id,
        version=1,
        rubric_text=definition.rubric_text.strip(),
        judge_model=model_option,
        output_schema=definition.output_schema,
    )


def validate_provider_environment(model_option: "ModelOption") -> None:
    """Ensure required environment variables exist for the selected provider."""
    if model_option.provider != "azure_openai":
        return

    missing: list[str] = []
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        missing.append("AZURE_OPENAI_ENDPOINT")

    # If API key is missing, check if we have a token provider as fallback
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        has_token_refresh = False
        # Check for MSAL cache
        if os.path.exists(os.path.expanduser('~/.azure/msal_token_cache.json')):
            has_token_refresh = True
        # Check for azure-identity (requires az CLI login)
        elif os.environ.get("USE_AZURE_IDENTITY") == "true":
            has_token_refresh = True
            
        if not has_token_refresh:
            missing.append("AZURE_OPENAI_API_KEY (or MSAL cache for auto-refresh)")

    api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
    if not api_version:
        missing.append("OPENAI_API_VERSION (or AZURE_OPENAI_API_VERSION)")
    else:
        os.environ.setdefault("OPENAI_API_VERSION", api_version)

    if missing:
        raise EnvironmentError(
            "Azure OpenAI environment configuration is incomplete. Missing: "
            + ", ".join(missing)
        )


def build_docent_agent_runs(
    conversations: Sequence[TaskConversation],
    failed_tasks: set[str],
    agent_name: str,
) -> list["AgentRun"]:
    if AgentRun is None or Transcript is None or parse_chat_message is None:
        raise RuntimeError(
            "Docent data models are unavailable. Ensure the docent package is installed and importable."
        )

    # Map unsupported roles to supported ones
    ROLE_MAPPING = {
        "developer": "system",  # OpenAI's developer role is similar to system
    }

    agent_runs: list[AgentRun] = []
    for conversation in conversations:
        parsed_messages = []
        for message in conversation.messages:
            # Map unsupported roles to supported equivalents
            role = ROLE_MAPPING.get(message.role, message.role)
            payload = {"role": role, "content": message.content or ""}
            try:
                parsed_messages.append(parse_chat_message(payload))
            except Exception as exc:  # pragma: no cover - best effort logging
                print(
                    f"⚠️  Skipping malformed message in task {conversation.task_id}: {exc}"
                )
        if not parsed_messages:
            continue

        transcript_metadata = {
            "task_id": conversation.task_id,
            "entry_count": conversation.entry_count,
        }
        transcript = Transcript(messages=parsed_messages, metadata=transcript_metadata)

        run_metadata = {
            "task_id": conversation.task_id,
            "failed": conversation.task_id in failed_tasks,
            "agent": agent_name,
            "entry_count": conversation.entry_count,
            "message_count": len(parsed_messages),
        }
        try:
            agent_run = AgentRun(
                id=str(conversation.task_id),
                transcripts=[transcript],
                metadata=run_metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️  Failed to build AgentRun for {conversation.task_id}: {exc}")
            continue
        agent_runs.append(agent_run)

    return agent_runs


def _create_dynamic_batches(
    agent_runs: Sequence["AgentRun"],
    max_batch_messages: int,
) -> list[list["AgentRun"]]:
    """Create batches where total message count doesn't exceed max_batch_messages."""
    batches: list[list["AgentRun"]] = []
    current_batch: list["AgentRun"] = []
    current_count = 0

    for run in agent_runs:
        msg_count = run.metadata.get("message_count", 0)
        # If single run exceeds limit, it gets its own batch
        if msg_count >= max_batch_messages:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_count = 0
            batches.append([run])
        elif current_count + msg_count > max_batch_messages:
            # Start new batch
            batches.append(current_batch)
            current_batch = [run]
            current_count = msg_count
        else:
            current_batch.append(run)
            current_count += msg_count

    if current_batch:
        batches.append(current_batch)

    return batches


def _get_fallback_urls() -> list[str]:
    """Get list of fallback URLs from environment variable."""
    fallback_env = os.getenv("OPENAI_FALLBACK_URLS", "")
    if fallback_env:
        return [u.strip() for u in fallback_env.split(",") if u.strip()]
    # Default to current URL only
    current = os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
    return [current]


def _switch_to_url(url: str, token: str | None = None) -> None:
    """Switch OPENAI_BASE_URL to a new URL and reinitialize client if needed."""
    os.environ["OPENAI_BASE_URL"] = url
    os.environ["AZURE_OPENAI_ENDPOINT"] = url.replace("/openai", "")
    
    if token:
        os.environ["OPENAI_API_KEY"] = token
        os.environ["AZURE_OPENAI_API_KEY"] = token
        
    # Try to reinitialize the OpenAI client in docent if possible
    try:
        from docent_core._llm_util.providers import openai as openai_provider
        # Clear cached clients to force recreation with new URL
        if hasattr(openai_provider, '_client'):
            openai_provider._client = None
        if hasattr(openai_provider, 'get_client'):
            openai_provider._cached_client = None
            
        from docent_core._llm_util.providers import azure_openai as azure_provider
        if hasattr(azure_provider, '_client'):
            azure_provider._client = None
    except Exception:
        pass  # Best effort - env var change should be picked up on next client init


def _is_connection_error(error_str: str) -> bool:
    """Check if error is a connection/timeout/blocked error that warrants URL fallback."""
    connection_indicators = (
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection error",
        "connect error",
        "unreachable",
        "no route to host",
        "name resolution",
        "dns",
        "eof",
        "broken pipe",
        "connection aborted",
        "ssl",
        "certificate",
        "handshake",
        "502",
        "503",
        "504",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "permission denied",
        "forbidden",
        "blocked",
        "403",
        "access denied"
    )
    return any(indicator in error_str for indicator in connection_indicators)


async def evaluate_environmental_barrier(
    agent_runs: Sequence["AgentRun"],
    rubric: "Rubric",
    batch_size: int = DEFAULT_RUBRIC_BATCH_SIZE,
    max_batch_messages: int = 0,
    inter_batch_delay: float = 0,
    retries: int = 3,
    json_mode: bool = False,
    use_cache: bool = True,
    rate_limit_delay: int = 65,
    max_concurrency: int = 10,
) -> list[LocalRubricEvaluation]:
    if evaluate_rubric is None:
        raise RuntimeError("Docent rubric evaluator is unavailable. Ensure docent is installed.")

    if batch_size <= 0:
        batch_size = 1

    evaluations: list[LocalRubricEvaluation] = []

    # Get fallback URLs for connection error recovery
    fallback_urls = _get_fallback_urls()
    current_url_idx = 0
    if len(fallback_urls) > 1:
        print(f"  📡 Using {len(fallback_urls)} fallback URLs for rotation")

    # Use dynamic batching if max_batch_messages is set
    if max_batch_messages > 0:
        batches = _create_dynamic_batches(agent_runs, max_batch_messages)
    else:
        # Fixed-size batching
        batches = [
            list(agent_runs[start : start + batch_size])
            for start in range(0, len(agent_runs), batch_size)
        ]

    for batch_idx, batch in enumerate(batches):
        batch_msg_count = sum(r.metadata.get("message_count", 0) for r in batch)
        print(f"  Processing batch {batch_idx + 1}/{len(batches)}: {len(batch)} tasks ({batch_msg_count} messages)...")
        response_format = {"type": "json_object"} if json_mode else None

        # Retry logic with exponential backoff, rate limit detection, and URL fallback
        outputs = None
        last_error = None
        urls_tried = 0  # Track how many URLs we've tried for this batch

        for attempt in range(retries):
            try:
                outputs = await evaluate_rubric(batch, rubric, response_format=response_format, use_cache=use_cache, max_concurrency=max_concurrency)
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limit errors (429)
                is_rate_limit = (
                    "429" in error_str or
                    "rate" in error_str or
                    "too many requests" in error_str or
                    "token limit" in error_str
                )
                
                # Check for auth errors (401)
                is_auth_error = "401" in error_str or "unauthorized" in error_str

                # Check for connection errors that warrant URL fallback
                is_connection = _is_connection_error(error_str)

                if is_auth_error:
                    print(f"    ⚠️  Auth error: {e}")
                    print(f"    🔄 Refreshing Azure token...")
                    new_token = get_azure_token()
                    if new_token:
                        # Update env vars with new token (keeping current URL)
                        _switch_to_url(os.environ["OPENAI_BASE_URL"], new_token)
                        # Retry immediately
                        continue
                    else:
                        print("    ❌ Failed to refresh token.")

                if is_connection and len(fallback_urls) > 1 and urls_tried < len(fallback_urls):
                    # Try next fallback URL
                    current_url_idx = (current_url_idx + 1) % len(fallback_urls)
                    next_url = fallback_urls[current_url_idx]
                    urls_tried += 1
                    print(f"    ⚠️  Connection error: {e}")
                    print(f"    🔄 Switching to fallback URL: {next_url}")
                    # Refresh token too while we're at it, to be safe
                    new_token = get_azure_token()
                    _switch_to_url(next_url, new_token)
                    # Retry immediately with new URL (no wait)
                    continue

                if is_rate_limit:
                    # Try to extract wait time from error message (e.g., "Try again in 44 seconds")
                    wait_match = re.search(r'try again in (\d+)', error_str)
                    if wait_match:
                        wait_time = int(wait_match.group(1)) + 5  # Add buffer
                    else:
                        wait_time = rate_limit_delay  # Use configured rate limit delay
                    print(f"    ⚠️  Rate limit hit (attempt {attempt + 1}/{retries}): {e}")
                else:
                    # Regular exponential backoff for other errors
                    wait_time = min(2 ** attempt * 3, 60)  # 3, 6, 12, 24, 48, 60 max
                    print(f"    ⚠️  Attempt {attempt + 1}/{retries} failed: {e}")

                if attempt < retries - 1:
                    print(f"    Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    # Reset URL tried counter for next retry round
                    urls_tried = 0

        if outputs is None:
            print(f"    ❌ Batch failed after {retries} attempts: {last_error}")
            # Mark all tasks in batch as failed
            for agent_run in batch:
                task_id = str(agent_run.metadata.get("task_id") or agent_run.id)
                evaluations.append(
                    LocalRubricEvaluation(
                        task_id=task_id,
                        rubric_id=rubric.id,
                        rubric_version=rubric.version,
                        output=None,
                        error=f"Batch failed after {retries} attempts: {last_error}",
                    )
                )
        else:
            for agent_run, output in zip(batch, outputs):
                error = None
                if output is None:
                    error = "Rubric evaluation returned no valid output."
                task_id = str(agent_run.metadata.get("task_id") or agent_run.id)
                evaluations.append(
                    LocalRubricEvaluation(
                        task_id=task_id,
                        rubric_id=rubric.id,
                        rubric_version=rubric.version,
                        output=output,
                        error=error,
                    )
                )

        # Delay between batches
        if inter_batch_delay > 0 and batch_idx < len(batches) - 1:
            print(f"    Waiting {inter_batch_delay}s before next batch...")
            time.sleep(inter_batch_delay)

    return evaluations


def preview_task(conversation: TaskConversation, limit: int = 8) -> None:
    """Print a human-readable preview for one task."""
    print(f"\n🔍 Previewing task {conversation.task_id}")
    print(f"   Entries: {conversation.entry_count}")
    print(f"   Messages extracted: {len(conversation.messages)}")

    if not conversation.messages:
        print("   (No messages extracted)")
        return

    for message in conversation.messages[:limit]:
        snippet = message.content.replace("\n", " ")[:160]
        print(f"   [{message.role}] {snippet}")

    remaining = len(conversation.messages) - limit
    if remaining > 0:
        print(f"   ... ({remaining} more messages)")


def confirm(prompt: str) -> bool:
    """Prompt the user before uploading."""
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def print_grading_results(
    rubric_id: str,
    results: Sequence[LocalRubricEvaluation],
) -> None:
    """Pretty-print grading results returned by Docent rubric evaluation."""
    if not results:
        print("⚠️  Rubric evaluation returned no results.")
        return

    print(f"\n🧪 {rubric_id} Grades:")
    for item in results:
        score = f"{item.score:.2f}" if item.score is not None else "N/A"
        print(f"   • Task {item.task_id}: {item.rubric_id} = {score}")
        if item.error:
            print(f"      Error: {item.error}")
        elif item.explanation:
            print(f"      Explanation: {item.explanation.strip()}")


def write_cloud_grading_csv(
    grading_results: Sequence[LocalRubricEvaluation],
    model_run: str,
    output_path: Path | None = None,
    default_criteria: str = "environmentalbarrier",
    task_success_map: dict[str, bool | None] | None = None,
) -> Path | None:
    """Persist cloud grading results to CSV."""
    if not grading_results:
        return None

    final_path = output_path or (Path("result/.hal_data/rubrics_output") / f"{default_criteria}.csv")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "criteria", "grade", "correct", "explanation", "model_run"])
        for item in grading_results:
            if item.score is None:
                grade_value = ""
            else:
                grade_value = f"{item.score:.2f}"

            success = ""
            if task_success_map is not None:
                success_flag = task_success_map.get(item.task_id)
                if success_flag is True:
                    success = "1"
                elif success_flag is False:
                    success = "0"

            writer.writerow(
                [
                    item.task_id,
                    item.rubric_id or default_criteria,
                    grade_value,
                    success,
                    item.explanation.strip(),
                    model_run,
                ]
            )
    return final_path


def default_rubric_output_path(
    rubric_id: str,
    output_dir: Path,
    trace_label: str,
) -> Path:
    rubric_slug = _slugify(rubric_id)
    trace_slug = _slugify(trace_label)
    return output_dir / rubric_slug / f"{trace_slug}.csv"


def run_rubric_evaluation(args: argparse.Namespace) -> None:
    try:
        trace_path = resolve_trace_path(args.trace_file, Path(args.trace_dir))
        data = load_trace_file(trace_path)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return

    trace_label = trace_path.stem

    dataset_overview(data)

    raw_entries = data.get("raw_logging_results") or []
    if not raw_entries:
        print("❌ No raw logging results found in trace file.")
        return

    tasks = group_entries_by_task(raw_entries)
    print(f"\n🧵 Found {len(tasks)} unique tasks")

    conversations: list[TaskConversation] = []
    for task_id, entries in tasks.items():
        conversation = extract_task_messages(task_id, entries)
        # Include all tasks, even those without messages (e.g. immediate failure)
        # This ensures the rubric CSV has a line for every task in the logs.
        conversations.append(conversation)

    conversations.sort(key=lambda conv: conv.task_id)

    results_block = data.get("results", {}) or {}
    failed_tasks_list = results_block.get("failed_tasks", [])
    successful_tasks_list = results_block.get("successful_tasks", [])

    # Fallback: Infer task status from raw_eval_results if summaries are missing
    if not failed_tasks_list and not successful_tasks_list:
        raw_eval = data.get("raw_eval_results", {})
        if isinstance(raw_eval, dict):
            eval_result = raw_eval.get("eval_result")
            if isinstance(eval_result, dict):
                for tid, res in eval_result.items():
                    is_success = False
                    if isinstance(res, dict):
                        success_rate = res.get("success_rate", 0)
                        is_success = success_rate >= 1.0 or res.get("success", False)
                    elif isinstance(res, (int, float)):
                        is_success = res >= 1.0
                    
                    if is_success:
                        successful_tasks_list.append(str(tid))
                    else:
                        failed_tasks_list.append(str(tid))
                print(f"   [Info] Inferred {len(successful_tasks_list)} successful and {len(failed_tasks_list)} failed tasks from raw results.")
        elif isinstance(raw_eval, list):
            # For ColBench, raw_eval_results is a list of scores corresponding to tasks in log order
            for idx, res in enumerate(raw_eval):
                try:
                    score = float(res)
                    if score >= 0.999:
                        successful_tasks_list.append(str(idx))
                    else:
                        failed_tasks_list.append(str(idx))
                except (ValueError, TypeError):
                    failed_tasks_list.append(str(idx))
            print(f"   [Info] Inferred {len(successful_tasks_list)} successful and {len(failed_tasks_list)} failed tasks from list results.")

    failed_tasks = set(failed_tasks_list)
    successful_tasks = set(successful_tasks_list)
    task_success_map: dict[str, bool | None] = {}
    for task_id in successful_tasks:
        task_success_map[task_id] = True
    for task_id in failed_tasks:
        task_success_map[task_id] = False

    if args.failed_only:
        original = len(conversations)
        conversations = [conv for conv in conversations if conv.task_id in failed_tasks]
        if not conversations:
            print("❌ No failed tasks available for rubric evaluation (--failed-only).")
            return
        print(f"🎯 Filtering to {len(conversations)} failed task(s) out of {original} (--failed-only).")

    if hasattr(args, "task_ids") and args.task_ids:
        original = len(conversations)
        # Ensure task_ids are strings for comparison
        target_ids = set(str(t) for t in args.task_ids)
        conversations = [conv for conv in conversations if conv.task_id in target_ids]
        if not conversations:
            print("❌ No tasks matched the provided task_ids filter.")
            return
        print(f"🎯 Filtering to {len(conversations)} specific task(s) out of {original} (via task_ids).")

    if not conversations:
        print("❌ Failed to extract any conversations.")
        return

    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            print("❌ --max-tasks must be positive when provided.")
            return
        original_count = len(conversations)
        conversations = conversations[: args.max_tasks]
        print(f"🔬 Limiting evaluation to {len(conversations)} of {original_count} tasks (--max-tasks).")

    preview_task(conversations[0])

    agent_name = data.get("config", {}).get("agent_name", "unknown-agent")
    if not (args.yes or confirm("\nProceed with Docent rubric evaluation using the configured LLM provider?")):
        print("ℹ️  Evaluation canceled. Inspect the preview above and rerun when ready.")
        return

    if DOCENT_IMPORT_ERROR is not None:
        print(f"❌ Unable to import Docent modules: {DOCENT_IMPORT_ERROR}")
        print("   Ensure the docent repo is installed (e.g., `pip install -e docent`).")
        return

    try:
        agent_runs = build_docent_agent_runs(conversations, failed_tasks, agent_name)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    if not agent_runs:
        print("❌ No agent runs could be constructed for rubric evaluation.")
        return

    # Sort by message count if requested
    if getattr(args, 'sort_by_messages', False):
        agent_runs = sorted(agent_runs, key=lambda r: r.metadata.get("message_count", 0))
        print(f"📊 Sorted {len(agent_runs)} tasks by message count (least to most)")

    effort_override = args.reasoning_effort.lower() if args.reasoning_effort else None

    try:
        model_option = resolve_rubric_model_option(args.rubric_model, reasoning_effort_override=effort_override)
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    try:
        validate_provider_environment(model_option)
    except EnvironmentError as exc:
        print(f"❌ {exc}")
        print("   Set the missing environment variables (see docs.transluce.org self-hosting env vars).")
        return

    # Auto-enable JSON mode for supported providers (OpenAI/Azure)
    use_json_mode = args.json_mode
    if model_option.provider in {"openai", "azure_openai"}:
        if not use_json_mode:
            print("📋 Auto-enabling JSON mode for structured output (OpenAI/Azure).")
        use_json_mode = True
    elif args.json_mode:
        print("⚠️  JSON mode requested but not supported for this provider. Using prompt-based JSON.")
        use_json_mode = False

    # Load rubric(s) - either single file or directory
    rubrics_dir = Path(args.rubrics_dir).expanduser()
    if hasattr(args, 'rubric') and args.rubric:
        rubric_path = Path(args.rubric).expanduser()
        if not rubric_path.is_absolute():
            rubric_path = Path.cwd() / rubric_path
        rubric_def = load_single_rubric(rubric_path, rubrics_dir=rubric_path.parent)
        if not rubric_def:
            print(f"❌ Could not load rubric from {rubric_path}")
            return
        rubric_definitions = [rubric_def]
    else:
        rubric_definitions = load_rubric_definitions(rubrics_dir)
        if not rubric_definitions:
            print(f"❌ No rubric definitions found in {rubrics_dir}. Add *.txt files and retry.")
            return

    # Use --parallel arg, fall back to env var, then default
    batch_size = getattr(args, 'parallel', None) or DEFAULT_RUBRIC_BATCH_SIZE
    env_batch = os.getenv("DOCENT_RUBRIC_BATCH_SIZE")
    if env_batch and not getattr(args, 'parallel', None):
        try:
            batch_size = max(1, int(env_batch))
        except ValueError:
            pass
    batch_size = max(1, batch_size)

    # Get max_batch_messages for dynamic batching
    max_batch_messages = getattr(args, 'max_batch_messages', 0) or 0
    inter_batch_delay = getattr(args, 'inter_batch_delay', 0) or 0
    retries = getattr(args, 'retries', 3) or 3
    rate_limit_delay = getattr(args, 'rate_limit_delay', 65) or 65
    max_concurrency = getattr(args, 'max_concurrency', 10) or 10

    output_dir = Path(args.output_dir).expanduser()
    if args.output_mode == "csv":
        output_dir.mkdir(parents=True, exist_ok=True)

    for definition in rubric_definitions:
        rubric = build_rubric_from_definition(definition, model_option)
        output_path = default_rubric_output_path(definition.rubric_id, output_dir, trace_label)

        if max_batch_messages > 0:
            print(
                f"\n🧪 Running rubric '{definition.rubric_id}' on {len(agent_runs)} agent runs "
                f"with {model_option.provider}:{model_option.model_name} (max_batch_messages={max_batch_messages})..."
            )
        else:
            print(
                f"\n🧪 Running rubric '{definition.rubric_id}' on {len(agent_runs)} agent runs "
                f"with {model_option.provider}:{model_option.model_name} (batch_size={batch_size})..."
            )

        # Determine cache usage - default to True unless --no-cache is specified
        use_cache = not getattr(args, "no_cache", False)
        try:
            grading_results = asyncio.run(
                evaluate_environmental_barrier(
                    agent_runs,
                    rubric,
                    batch_size=batch_size,
                    max_batch_messages=max_batch_messages,
                    inter_batch_delay=inter_batch_delay,
                    retries=retries,
                    json_mode=use_json_mode,
                    use_cache=use_cache,
                    rate_limit_delay=rate_limit_delay,
                    max_concurrency=max_concurrency,
                ),
            )
        except Exception as exc:  # pragma: no cover - depends on provider availability
            print(f"❌ Rubric '{definition.rubric_id}' evaluation failed: {exc}")
            continue

        print_grading_results(definition.rubric_id, grading_results)

        if args.output_mode == "csv":
            csv_path = write_cloud_grading_csv(
                grading_results,
                trace_label,
                output_path=output_path,
                default_criteria=definition.rubric_id,
                task_success_map=task_success_map,
            )
            if csv_path:
                print(f"🗂️  Rubric CSV written to {csv_path}")
        else:
            print("🗂️  Output mode 'stdout' selected; skipping CSV export.")


def main():
    parser = argparse.ArgumentParser(
        description="Rubric evaluation using Docent. Input: trace file. Output: CSV with same name."
    )

    # Trace selection
    parser.add_argument(
        "--trace-file",
        type=str,
        action="append",
        dest="trace_files",
        help="Path to trace JSON file to evaluate (can be specified multiple times)",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="result/.hal_data",
        help="Directory to scan for trace files when using --prefix (default: result/.hal_data)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Regex prefix to group trace files by (e.g., 'sky[0-9]+_'). If provided, finds traces in 'traces/' folder.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark name (required for --fixes-only to locate correct directory)",
    )

    # Rubric configuration
    parser.add_argument(
        "--rubric",
        type=str,
        help="Path to a single rubric .txt file (overrides --rubrics-dir)",
    )
    parser.add_argument(
        "--rubrics-dir",
        type=str,
        default="config/rubric",
        help="Directory containing *.txt rubric definitions (default: config/rubric/)",
    )
    parser.add_argument(
        "--rubric-model",
        type=str,
        help="Model as provider:model. Defaults to gpt-5.2 from models/model_rubrics.json if available.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning models",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/.hal_data/rubrics_output",
        help="Directory for CSV output (default: result/.hal_data/rubrics_output/)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["csv", "stdout"],
        default="csv",
        help="Output mode: csv (write files) or stdout (print only)",
    )

    # Filtering
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Limit number of tasks to evaluate",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only evaluate tasks in failed_tasks list",
    )
    parser.add_argument(
        "--fixes-only",
        action="store_true",
        help="Only evaluate tasks that have corresponding fixes in the fixes/ directory",
    )

    # Other options
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Force JSON-mode (auto-enabled for OpenAI/Azure)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--max-batch-messages",
        type=int,
        default=1000,
        help="Max total messages per batch (default: 1000). Dynamically adjusts batch size.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per batch on failure (default: 3).",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=int,
        default=65,
        help="Seconds to wait on rate limit errors (default: 65). Set to match your API rate limit window.",
    )
    parser.add_argument(
        "--sort-by-messages",
        action="store_true",
        help="Sort tasks from least to most messages before processing.",
    )
    parser.add_argument(
        "--sort-by-file-size",
        action="store_true",
        help="Sort trace files from smallest to largest file size before processing.",
    )
    parser.add_argument(
        "--inbetween",
        type=str,
        help="Bash command to execute after each trace file (e.g., 'TMUX= ./deploy_llm.sh')",
    )
    parser.add_argument(
        "--sleep",
        type=str,
        help="Sleep duration before and after inbetween command (e.g., '5s', '2m')",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching (force re-evaluation of all tasks).",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="OpenAI API base URL(s). If not provided, uses Azure/TRAPI directly. "
             "Comma-separated for fallback on errors "
             "(e.g., 'http://localhost:4000/v1,http://localhost:4001/v1')",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Treat as original (pre-revision) data. Currently mainly for consistency with other scripts.",
    )

    args = parser.parse_args()

    # FORCE override rubric model with resolved Azure model if applicable
    # This ensures we use 'azure_openai:...' provider with correct headers/client
    if _resolved_model:
        print(f"[Override] Replacing user model '{args.rubric_model}' with resolved '{_resolved_model}'")
        args.rubric_model = _resolved_model

    # Set defaults for underlying CLI (removed from this script for simplicity)
    args.parallel = 1000  # Not used when max_batch_messages > 0
    args.max_concurrency = 1000  # High concurrency for throughput
    args.inter_batch_delay = 0  # No delay between batches

    # Parse sleep duration
    sleep_seconds = 0
    if args.sleep:
        match = re.match(r'^(\d+)(s|m)?$', args.sleep)
        if match:
            value = int(match.group(1))
            unit = match.group(2) or 's'
            sleep_seconds = value * 60 if unit == 'm' else value
        else:
            print(f"Invalid sleep format: {args.sleep}. Use e.g., '5s' or '2m'")
            sys.exit(1)

    # Set trace_dir (required by CLI but not used when trace_file is specified)
    trace_dir = Path(args.traces_dir)
    if not trace_dir.is_absolute():
        trace_dir = REPO_ROOT / trace_dir
    args.trace_dir = str(trace_dir)

    # Resolve rubric path
    if args.rubric:
        rubric_path = Path(args.rubric)
        if not rubric_path.is_absolute():
            rubric_path = REPO_ROOT / rubric_path
        args.rubric = str(rubric_path)

    # Resolve rubrics directory
    rubrics_dir = Path(args.rubrics_dir)
    if not rubrics_dir.is_absolute():
        rubrics_dir = REPO_ROOT / rubrics_dir
    args.rubrics_dir = str(rubrics_dir)

    # Resolve output directory
    # Unified output paths: full rubrics to result/.hal_data/rubrics_output/
    output_dir = Path(args.output_dir)
    if args.output_dir == "rubrics_output":
        output_dir = REPO_ROOT / "result" / ".hal_data" / "rubrics_output"
    elif not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    args.output_dir = str(output_dir)

    # Resolve all trace file paths
    prefix_to_traces = defaultdict(list)
    
    # Pre-calculate fixes if needed
    fixes_ids = set()
    if args.fixes_only:
        if not args.benchmark:
            print("Error: --fixes-only requires --benchmark to be specified")
            sys.exit(1)
            
        # Map benchmark name to fixes directory name
        # Internal key -> Fix dir name
        benchmark_map = {
            "colbench_backend_programming": "colbench",
            "colbench": "colbench",
            "corebench_hard": "corebench_hard",
            "corebench": "corebench_hard",
            "scicode": "scicode",
            "scienceagentbench": "scienceagentbench",
            "usaco": "usaco"
        }
        
        fix_dir_name = benchmark_map.get(args.benchmark, args.benchmark)
        fixes_path = REPO_ROOT / "result" / "fixes" / fix_dir_name
        
        if not fixes_path.exists():
            print(f"Error: Fixes directory not found at {fixes_path}")
            sys.exit(1)
            
        # Collect all directory names in fixes_path as task IDs
        # Filter out non-numeric/special files if necessary, but generally any dir is a task
        for item in fixes_path.iterdir():
            if item.is_dir():
                fixes_ids.add(item.name)
        
        # Special filtering for ColBench (shared fixes directory)
        if fix_dir_name == "colbench":
            filtered_ids = set()
            is_backend_bench = "backend" in args.benchmark
            is_frontend_bench = "frontend" in args.benchmark
            
            if is_backend_bench or is_frontend_bench:
                for tid in list(fixes_ids):
                    fix_path = fixes_path / tid
                    is_frontend_fix = False
                    
                    # Check content for frontend markers
                    try:
                        # Check README first
                        readme = fix_path / "README.md"
                        if readme.exists():
                            content = readme.read_text(errors="ignore").lower()
                            if "frontend" in content or "html" in content:
                                is_frontend_fix = True
                        
                        # Check json if README inconclusive
                        if not is_frontend_fix:
                            for json_file in fix_path.glob("*.json"):
                                content = json_file.read_text(errors="ignore").lower()
                                if "frontend" in content or "html" in content:
                                    is_frontend_fix = True
                                    break
                    except Exception:
                        pass
                    
                    # Filter based on benchmark type
                    if is_backend_bench:
                        if not is_frontend_fix:
                            filtered_ids.add(tid)
                    elif is_frontend_bench:
                        if is_frontend_fix:
                            filtered_ids.add(tid)
                
                fixes_ids = filtered_ids
                print(f"Filtered to {len(fixes_ids)} {args.benchmark} tasks (removed {'frontend' if is_backend_bench else 'backend'} mismatches)")
        
        print(f"Found {len(fixes_ids)} tasks with fixes in {fixes_path}")
        if not fixes_ids:
            print("Error: No fixes found. Exiting.")
            sys.exit(1)
            
        # Pass to rubric_cli via args
        args.task_ids = list(fixes_ids)

    if args.prefix:
        trace_dir = Path(args.trace_dir)
        if not trace_dir.exists():
            print(f"Error: trace directory not found at {trace_dir}")
            sys.exit(1)
        
        try:
            regex = re.compile(args.prefix)
        except re.error as e:
            print(f"Error: Invalid prefix regex '{args.prefix}': {e}")
            sys.exit(1)

        print(f"🔍 Scanning for traces in: {trace_dir}")
            
        all_traces = sorted(trace_dir.glob("*.json"))
        for f in all_traces:
            # Skip merged or temporary files if necessary
            if "_MERGED_UPLOAD.json" in f.name or "__" in f.name:
                continue
            match = regex.search(f.name)
            if match:
                actual_pfx = match.group(0)
                prefix_to_traces[actual_pfx].append(f)
        
        if not prefix_to_traces:
            print(f"Error: No trace files matching prefix regex '{args.prefix}' found in {trace_dir}")
            sys.exit(1)
    elif args.trace_files:
        for trace_file in args.trace_files:
            trace_path = Path(trace_file)
            if not trace_path.is_absolute():
                trace_path = REPO_ROOT / trace_path
            prefix_to_traces["default"].append(trace_path)
    else:
        print("Error: Either --trace-file or --prefix must be provided.")
        sys.exit(1)

    # Process each prefix group
    for actual_prefix, trace_files in sorted(prefix_to_traces.items()):
        print(f"\n{'#'*80}")
        print(f"### PROCESSING PREFIX: {actual_prefix}")
        print(f"{'#'*80}")

        # Sort by file size if requested (smallest to largest)
        if args.sort_by_file_size:
            trace_files.sort(key=lambda p: p.stat().st_size)
            print("Trace files sorted by file size (smallest to largest):")
            for tf in trace_files:
                size_mb = tf.stat().st_size / (1024 * 1024)
                print(f"  {tf.name}: {size_mb:.2f} MB")
            print()

        # Track which rubric CSVs correspond to this prefix
        generated_csvs = []

        # Process each trace file independently
        for i, trace_path in enumerate(trace_files):
            args.trace_file = str(trace_path)

            print(f"\n[{i+1}/{len(trace_files)}] Processing: {trace_path.name}")
            print(f"{'-'*60}\n")

            # Run the evaluator for this trace
            run_rubric_evaluation(args)
            
            # Determine where the output CSV went
            # eval_rubric.py usually puts it in rubrics_output/<rubric_name>/<trace_name>.csv
            rubric_name = Path(args.rubric).stem if args.rubric else "default"
            if args.rubric and "scienceagentbench" in args.rubric:
                rubric_name = "scienceagentbench"
            
            output_csv = Path(args.output_dir) / rubric_name / f"{trace_path.stem}.csv"
            if output_csv.exists():
                generated_csvs.append(output_csv)

            # Execute inbetween command after each trace file
            if args.inbetween:
                if sleep_seconds:
                    print(f"Sleeping for {sleep_seconds}s before inbetween command...")
                    time.sleep(sleep_seconds)
                print(f"\n{'='*60}")
                print(f"Running inbetween command: {args.inbetween}")
                print(f"{'='*60}\n")
                subprocess.run(args.inbetween, shell=True, check=True)
                if sleep_seconds:
                    print(f"Sleeping for {sleep_seconds}s after inbetween command...")
                    time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
