#!/usr/bin/env python3
"""
Judge script for aggregating rubric evaluations across multiple model runs.

Uses docent for LLM calls with caching and retry/backoff.
Defaults to direct Azure/TRAPI access (no proxy needed).

Usage:
    # Default: Uses Azure/TRAPI directly (recommended)
    python scripts/judge.py \
        --pattern "scicode_*" \
        --rubric-dir rubrics_output/scicode \
        --model openai:gpt-5.2 \
        --parallel 5 \
        -y

    # With proxy/custom endpoint (overrides Azure default)
    python scripts/judge.py \
        --pattern "*.csv" \
        --rubric-dir rubrics_output/scicode \
        --model openai:o3-mini \
        --openai-base-url "http://localhost:4000/v1" \
        --parallel 5 \
        -y

    # With reasoning model
    python scripts/judge.py \
        --pattern "*.csv" \
        --rubric-dir rubrics_output/scicode \
        --model openai:o3-mini \
        --reasoning-effort medium \
        --parallel 5 \
        -y
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# ============================================================================
# TRAPI/Azure Direct Access Configuration
# ============================================================================

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


def setup_azure_environment(model: str | None = None) -> bool:
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
    os.environ["OPENAI_BASE_URL"] = f"{endpoint}/openai"
    os.environ["OPENAI_API_KEY"] = token
    os.environ["OPENAI_API_VERSION"] = api_version

    # Also set Azure-specific vars for azure_openai provider compatibility
    os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
    os.environ["AZURE_OPENAI_API_KEY"] = token
    os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    print(f"[Azure] Direct TRAPI access configured: {endpoint}")
    return True


# Pre-parse --openai-base-url BEFORE importing docent
# (docent reads OPENAI_BASE_URL at import time)
# If not provided, use Azure/TRAPI directly
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--openai-base-url", type=str, default=None)
_pre_parser.add_argument("--model", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

_using_azure_direct = False
_resolved_model = None
if _pre_args.openai_base_url is None:
    # No proxy URL provided - use Azure/TRAPI directly
    _using_azure_direct = setup_azure_environment(_pre_args.model)
    if _using_azure_direct and _pre_args.model:
        # Resolve model name to TRAPI deployment name AND switch to azure_openai provider
        if ':' in _pre_args.model:
            provider, model_name = _pre_args.model.split(':', 1)
            deployment_name = resolve_trapi_deployment(model_name)
            # CRITICAL: Use azure_openai provider instead of openai
            _resolved_model = f"azure_openai:{deployment_name}"
            print(f"[Azure] Model resolved: {_pre_args.model} -> {_resolved_model}")
        else:
            deployment_name = resolve_trapi_deployment(_pre_args.model)
            _resolved_model = f"azure_openai:{deployment_name}"
            print(f"[Azure] Model resolved: {_pre_args.model} -> {_resolved_model}")
    if not _using_azure_direct:
        # Fallback to localhost proxy
        os.environ["OPENAI_BASE_URL"] = "http://localhost:4000/v1"
        os.environ["OPENAI_FALLBACK_URLS"] = "http://localhost:4000/v1"
else:
    # Proxy URL provided - use it
    _all_urls = [u.strip() for u in _pre_args.openai_base_url.split(",")]
    os.environ["OPENAI_BASE_URL"] = _all_urls[0]
    os.environ["OPENAI_FALLBACK_URLS"] = ",".join(_all_urls)
    print(f"[Proxy] Using custom endpoint: {_all_urls[0]}")

# Set environment BEFORE importing docent
os.environ.setdefault("ENV_RESOLUTION_STRATEGY", "os_environ")


def _ensure_llm_cache_path() -> None:
    """Docent expects LLM_CACHE_PATH; default to .llm_cache if not provided."""
    if os.getenv("LLM_CACHE_PATH"):
        return
    default_cache = Path(".llm_cache")
    default_cache.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_CACHE_PATH"] = str(default_cache.resolve())


_ensure_llm_cache_path()

# Docent imports (required)
try:
    from docent_core._llm_util.prod_llms import LLMManager
    from docent_core._llm_util.providers.preferences import ModelOption
except ImportError as e:
    print(f"Error: docent not available ({e}). Ensure docent is installed.")
    sys.exit(1)


@dataclass
class TaskEvaluation:
    """A single evaluation result for a task from one model run."""
    task_id: str
    criteria: str
    grade: str
    correct: str
    explanation: str
    model_run: str
    source_file: str


@dataclass
class JudgeVerdict:
    """Final verdict from the judge for a task."""
    task_id: str
    final_grade: int  # Binary: 0 or 1
    satisfies_rubric: bool
    reasoning: str
    num_evaluations: int
    model_runs: list[str]


JUDGE_SYSTEM_PROMPT = """You are an EXTREMELY SKEPTICAL expert judge analyzing rubric evaluations from multiple AI agent runs on scientific coding benchmarks.

Your task is to determine whether a specific task has "Intrinsic Formation Errors" (IFEs) based on evaluations from different model runs.

## CRITICAL: Be Highly Skeptical

YOU MUST BE SKEPTICAL OF THE EVIDENCE PROVIDED. The evaluations you receive may contain:
- Incorrect analysis or misattributed failures
- Confirmation bias from evaluators
- Superficial pattern matching without deep understanding
- Mistakes in understanding the task requirements
- Overconfident claims without sufficient proof

ALWAYS ask yourself:
1. Could this failure actually be the agent's fault disguised as a benchmark issue?
2. Is there an alternative explanation where the benchmark is correct and the agent simply failed?
3. Could a sufficiently capable agent have solved this task correctly?
4. Is the "evidence" actually proving an IFE, or just showing agent mistakes?
5. Are there ways the agent could have worked around the supposed issue?

## What are Intrinsic Formation Errors (IFEs)?
IFEs are GENUINE defects in the benchmark/evaluation setup itself - NOT failures in agent reasoning or coding ability. TRUE IFEs include:
- Contradictory or PROVABLY impossible task requirements (not just difficult ones)
- Parsing/regex issues in the evaluation harness that DEFINITIVELY reject valid solutions
- Instructions that are GENUINELY ambiguous with no reasonable interpretation
- Missing dependencies that CANNOT be installed or worked around
- Evaluation criteria that DIRECTLY contradict the stated task requirements

## What are DEFINITELY NOT IFEs (reject these claims):
- Agent making coding mistakes or logic errors (this is agent failure)
- Agent misunderstanding a clear task requirement (agent failure)
- Agent producing incorrect numerical results (agent failure)
- Agent failing to follow formatting instructions (agent failure)
- Tasks that are "difficult" or "require domain knowledge" (not an IFE)
- Tasks where the agent "almost" got it right (still agent failure)
- Vague claims of "ambiguity" without specific proof
- Speculation about what "might" be wrong with the benchmark

## Original Rubric Context (for reference)
The original rubric evaluates whether traces show evidence of Intrinsic Formation Errors:
- Grade 1.0: Clear, irrefutable evidence of benchmark defect that NO agent could overcome
- Grade 0.0: No evidence of benchmark defect, or failure is attributable to agent

## Your Task - BINARY DECISION REQUIRED
You MUST output a BINARY grade: either 0 or 1. No intermediate values.
- Grade 1: ONLY if there is OVERWHELMING, IRREFUTABLE evidence of a genuine benchmark defect
- Grade 0: If there is ANY reasonable doubt, or if the failure could be agent-related

For each task, critically analyze:
1. Is the claimed issue PROVABLY a benchmark defect, or could it be agent failure?
2. What alternative explanations exist? Could a better agent have succeeded?
3. Is the evidence concrete and specific, or vague and speculative?
4. Would you bet your reputation that this is truly a benchmark bug?

## Response Format
Respond in JSON format:
{
    "final_grade": <BINARY: 0 or 1 ONLY>,
    "satisfies_rubric": <boolean - true ONLY if grade is 1>,
    "reasoning": "<your SKEPTICAL analysis including: (1) claimed issue, (2) why you doubt/accept it, (3) alternative explanations considered, (4) final determination>"
}

Remember: When in doubt, grade 0. The burden of proof is on demonstrating a TRUE benchmark defect.
"""


def find_csv_files(rubric_dir: Path, pattern: str) -> list[Path]:
    """Find CSV files matching the pattern in the rubric directory."""
    csv_files = []
    for f in rubric_dir.glob("*.csv"):
        if fnmatch(f.name, pattern):
            csv_files.append(f)
    return sorted(csv_files)


def read_evaluations(csv_files: list[Path]) -> dict[str, list[TaskEvaluation]]:
    """Read all CSV files and group evaluations by task_id."""
    evaluations: dict[str, list[TaskEvaluation]] = defaultdict(list)

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_item = TaskEvaluation(
                    task_id=row.get("task_id", ""),
                    criteria=row.get("criteria", ""),
                    grade=row.get("grade", ""),
                    correct=row.get("correct", ""),
                    explanation=row.get("explanation", ""),
                    model_run=row.get("model_run", ""),
                    source_file=csv_file.name,
                )
                evaluations[eval_item.task_id].append(eval_item)

    return evaluations


def build_judge_prompt(task_id: str, evals: list[TaskEvaluation]) -> str:
    """Build the prompt for the judge LLM."""
    prompt_parts = [
        f"# Task ID: {task_id}\n",
        f"## Evaluations from {len(evals)} model run(s):\n",
    ]

    for i, ev in enumerate(evals, 1):
        prompt_parts.append(f"""
### Evaluation {i} (from {ev.model_run})
- **Source**: {ev.source_file}
- **Criteria**: {ev.criteria}
- **Grade**: {ev.grade}
- **Correct**: {ev.correct}
- **Explanation**: {ev.explanation}
""")

    prompt_parts.append("""
## Your Task
Based on the above evaluations, provide your final verdict on whether this task has formational errors (issues with the evaluation environment, not agent capability).

Respond in JSON format with: final_grade, satisfies_rubric, reasoning
""")

    return "\n".join(prompt_parts)


def parse_judge_response(response: str) -> dict:
    """Parse the JSON response from the judge LLM. Enforces binary grades."""
    parsed = None

    # Try to extract JSON from the response
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        pass

    if not parsed:
        # Try to find JSON block in response
        json_match = re.search(r'\{[^{}]*"final_grade"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    if not parsed:
        # Try to find JSON in code block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

    if not parsed:
        return {
            "final_grade": 0,
            "satisfies_rubric": False,
            "reasoning": f"Failed to parse response: {response[:500]}",
        }

    # ENFORCE BINARY GRADE: round to 0 or 1
    raw_grade = float(parsed.get("final_grade", 0))
    binary_grade = 1 if raw_grade >= 0.5 else 0

    return {
        "final_grade": binary_grade,
        "satisfies_rubric": binary_grade == 1,
        "reasoning": parsed.get("reasoning", ""),
    }


async def judge_tasks_with_docent(
    task_ids: list[str],
    evaluations: dict[str, list[TaskEvaluation]],
    model_option: "ModelOption",
    parallel: int = 5,
    retries: int = 3,
    use_cache: bool = True,
) -> list[JudgeVerdict]:
    """Judge tasks using docent LLMManager with caching."""
    from tqdm.auto import tqdm

    # Build all prompts
    prompts = []
    task_id_order = []
    for task_id in task_ids:
        evals = evaluations[task_id]
        prompt = build_judge_prompt(task_id, evals)
        prompts.append([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        task_id_order.append(task_id)

    # Create LLMManager
    llm_manager = LLMManager(
        model_options=[model_option],
        use_cache=use_cache,
    )

    print(f"Calling {model_option.model_name} (temperature=0) with {parallel} concurrent requests...")

    # Call LLM with parallelism
    outputs = await llm_manager.get_completions(
        inputs=prompts,
        max_new_tokens=4096,
        temperature=0.0,  # Deterministic
        max_concurrency=parallel,
        timeout=300.0,  # 5 minute timeout per request
        response_format={"type": "json_object"},
    )

    # Process results
    verdicts = []
    for i, (task_id, output) in enumerate(zip(task_id_order, outputs)):
        evals = evaluations[task_id]

        if output.did_error or not output.completions:
            error_msg = str(output.errors[0]) if output.errors else "Unknown error"
            print(f"  ERROR: Task {task_id} failed: {error_msg}")
            verdict = JudgeVerdict(
                task_id=task_id,
                final_grade=0,
                satisfies_rubric=False,
                reasoning=f"LLM call failed: {error_msg}",
                num_evaluations=len(evals),
                model_runs=[ev.model_run for ev in evals],
            )
        else:
            content = output.completions[0].text or ""
            parsed = parse_judge_response(content)

            verdict = JudgeVerdict(
                task_id=task_id,
                final_grade=int(parsed.get("final_grade", 0)),
                satisfies_rubric=bool(parsed.get("satisfies_rubric", False)),
                reasoning=str(parsed.get("reasoning", "")),
                num_evaluations=len(evals),
                model_runs=[ev.model_run for ev in evals],
            )

        # Log result (full output, no truncation)
        status = "IFE CONFIRMED" if verdict.satisfies_rubric else "NO IFE"
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(task_ids)}] Task {task_id}")
        print(f"{'='*60}")
        print(f"Grade: {verdict.final_grade} ({status})")
        if verdict.reasoning:
            print(f"Reasoning:\n{verdict.reasoning}")

        verdicts.append(verdict)

    return verdicts


def write_verdicts_csv(verdicts: list[JudgeVerdict], output_path: Path) -> None:
    """Write judge verdicts to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "final_grade",
            "satisfies_rubric",
            "num_evaluations",
            "model_runs",
            "reasoning",
        ])
        for v in verdicts:
            writer.writerow([
                v.task_id,
                v.final_grade,
                "1" if v.satisfies_rubric else "0",
                v.num_evaluations,
                ";".join(v.model_runs),
                v.reasoning,
            ])

    print(f"Wrote {len(verdicts)} verdicts to {output_path}")


def parse_model_string(model_str: str) -> tuple[str, str]:
    """Parse provider:model string. Returns (provider, model_name)."""
    if ":" in model_str:
        provider, model_name = model_str.split(":", 1)
        return provider, model_name
    return "openai", model_str


def apply_priority_override(
    evaluations: dict[str, list[TaskEvaluation]],
    high_priority_pattern: str,
    low_priority_pattern: str
) -> dict[str, list[TaskEvaluation]]:
    """
    Apply priority override: for each task, if evaluations exist from high_priority_pattern,
    remove evaluations from low_priority_pattern.

    Args:
        evaluations: Dict mapping task_id to list of evaluations
        high_priority_pattern: Glob pattern for high priority files (e.g., "sab_husky_*")
        low_priority_pattern: Glob pattern for low priority files (e.g., "sab_cow_*")

    Returns:
        Filtered evaluations dict with overrides applied
    """
    from fnmatch import fnmatch

    filtered_evaluations = {}
    override_count = 0

    for task_id, evals in evaluations.items():
        # Check if any evaluations match high priority pattern
        has_high_priority = any(fnmatch(ev.source_file, high_priority_pattern) for ev in evals)

        if has_high_priority:
            # Filter out low priority evaluations
            filtered_evals = [
                ev for ev in evals
                if not fnmatch(ev.source_file, low_priority_pattern)
            ]
            removed_count = len(evals) - len(filtered_evals)
            if removed_count > 0:
                override_count += 1
            filtered_evaluations[task_id] = filtered_evals
        else:
            # No high priority evaluations, keep all
            filtered_evaluations[task_id] = evals

    if override_count > 0:
        print(f"Applied priority override: {high_priority_pattern} > {low_priority_pattern}")
        print(f"  Overridden {override_count} tasks")

    return filtered_evaluations


def main():
    parser = argparse.ArgumentParser(
        description="Judge aggregates rubric evaluations and produces final verdicts."
    )

    parser.add_argument(
        "--pattern",
        type=str,
        action='append',
        help="Glob pattern to match CSV files (can be used multiple times, e.g., '--pattern prop_* --pattern iter1_*')",
    )
    parser.add_argument(
        "--rubric-dir",
        type=str,
        required=True,
        help="Directory containing rubric CSV outputs (e.g., rubrics_output/scicode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model as provider:model (e.g., openai:gpt-5.2, openai:o3-mini)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning models",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV path (default: judge_output/<rubric_dir_name>_verdict.csv)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Limit number of tasks to judge",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per task on failure (default: 3)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of parallel tasks to judge at once (default: 5)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without making API calls",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching (force re-evaluation, keeps parallelism)",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        help="Custom OpenAI API base URL (overrides Azure/TRAPI default)",
    )
    parser.add_argument(
        "--priority-override",
        nargs=2,
        metavar=("HIGH_PRIORITY", "LOW_PRIORITY"),
        help="Override evaluations: if a task has evaluations from HIGH_PRIORITY pattern, remove LOW_PRIORITY evaluations for that task (e.g., 'sab_husky_*' 'sab_cow_*')",
    )

    args = parser.parse_args()

    # Resolve paths
    rubric_dir = Path(args.rubric_dir)
    if not rubric_dir.is_absolute():
        rubric_dir = REPO_ROOT / rubric_dir

    if not rubric_dir.exists():
        print(f"Error: Rubric directory not found: {rubric_dir}")
        sys.exit(1)

    # Find CSV files (handle multiple patterns)
    patterns = args.pattern if args.pattern else ["*.csv"]
    csv_files = []
    for pattern in patterns:
        csv_files.extend(find_csv_files(rubric_dir, pattern))
    # Remove duplicates and sort
    csv_files = sorted(set(csv_files))

    if not csv_files:
        print(f"Error: No CSV files matching patterns {patterns} found in {rubric_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Read evaluations
    evaluations = read_evaluations(csv_files)
    print(f"\nLoaded evaluations for {len(evaluations)} unique tasks")

    # Apply priority override if specified
    if args.priority_override:
        high_priority, low_priority = args.priority_override
        evaluations = apply_priority_override(evaluations, high_priority, low_priority)

    # Limit tasks if requested
    task_ids = list(evaluations.keys())
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]
        print(f"Limiting to {len(task_ids)} tasks")

    # Parse model - use resolved model if Azure direct access is active
    if _using_azure_direct and _resolved_model:
        provider, model_name = parse_model_string(_resolved_model)
        print(f"\nUsing model (Azure): {_resolved_model}")
    else:
        provider, model_name = parse_model_string(args.model)
        print(f"\nUsing model: {provider}:{model_name}")
    print(f"Temperature: 0 (deterministic)")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")

    # Determine output path early to show user
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    else:
        # Extract prefix from first pattern: keep only letters, numbers, and underscores
        # e.g., "scicode_potato_*" -> "scicode_potato"
        first_pattern = patterns[0] if patterns else "*.csv"
        pattern_prefix = re.sub(r'[^a-zA-Z0-9_]', '', first_pattern.split('*')[0].split('?')[0])
        pattern_prefix = pattern_prefix.rstrip('_')  # Remove trailing underscore
        if pattern_prefix:
            output_path = REPO_ROOT / "judge_output" / f"{pattern_prefix}_verdict.csv"
        else:
            output_path = REPO_ROOT / "judge_output" / f"{rubric_dir.name}_verdict.csv"

    print(f"Output: {output_path}")

    # Confirm
    if not args.yes:
        response = input(f"\nJudge {len(task_ids)} tasks? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    # Dry run mode
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Previewing prompts (no API calls)")
        print("="*60)

        for i, task_id in enumerate(task_ids[:3], 1):  # Show first 3 only
            evals = evaluations[task_id]
            prompt = build_judge_prompt(task_id, evals)

            print(f"\n{'='*60}")
            print(f"[{i}/{len(task_ids)}] Task: {task_id}")
            print(f"Evaluations: {len(evals)} from models: {[e.model_run for e in evals]}")
            print("="*60)
            print("\n--- SYSTEM PROMPT ---")
            print(JUDGE_SYSTEM_PROMPT[:800] + "...")
            print("\n--- USER PROMPT ---")
            print(prompt[:1500] + "..." if len(prompt) > 1500 else prompt)

        print(f"\n{'='*60}")
        print(f"DRY RUN COMPLETE - Would judge {len(task_ids)} tasks with {args.parallel} parallel workers")
        print("="*60)
        return

    # Run judge with docent
    use_cache = not args.no_cache
    cache_status = "caching enabled" if use_cache else "caching disabled"
    print(f"\nUsing docent ({cache_status}, parallel={args.parallel})...")

    model_option = ModelOption(
        provider=provider,
        model_name=model_name,
        reasoning_effort=args.reasoning_effort,
    )
    verdicts = asyncio.run(judge_tasks_with_docent(
        task_ids=task_ids,
        evaluations=evaluations,
        model_option=model_option,
        parallel=args.parallel,
        retries=args.retries,
        use_cache=use_cache,
    ))

    # Write output
    write_verdicts_csv(verdicts, output_path)

    # Sort verdicts by task_id
    verdicts.sort(key=lambda v: v.task_id)

    # Summary
    ife_confirmed = [v for v in verdicts if v.satisfies_rubric]
    no_ife = [v for v in verdicts if not v.satisfies_rubric]

    print(f"\n{'='*60}")
    print("SUMMARY (Binary Grades)")
    print("="*60)
    print(f"IFE Confirmed (grade=1): {len(ife_confirmed)}/{len(verdicts)}")
    print(f"No IFE (grade=0):        {len(no_ife)}/{len(verdicts)}")

    if ife_confirmed:
        print(f"\nTasks with confirmed IFEs:")
        for v in ife_confirmed:
            print(f"  - {v.task_id}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
