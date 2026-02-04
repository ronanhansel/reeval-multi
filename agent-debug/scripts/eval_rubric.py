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
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

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
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Pre-parse --openai-base-url BEFORE importing rubric_evaluator
# (the module reads OPENAI_BASE_URL at import time)
# If not provided, use Azure/TRAPI directly
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--openai-base-url", type=str, default=None)
_pre_parser.add_argument("--rubric-model", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

_using_azure_direct = False
_resolved_model = None

# Load model rubrics config
try:
    with open(REPO_ROOT / "model_configs" / "model_rubrics.json") as f:
        _rubric_config = json.load(f)
except Exception as e:
    print(f"Warning: Could not load model_configs/model_rubrics.json: {e}")
    _rubric_config = {}

# Determine model to use (default to gpt-5.2_2025-12-11 if not specified)
_target_model_key = "gpt-5.2_2025-12-11"
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

from rubric_evaluator import cli as rubric_cli


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
        default="eval_traces/traces",
        help="Directory to scan for trace files when using --prefix (default: eval_traces/traces)",
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
        default="rubrics",
        help="Directory containing *.txt rubric definitions (default: rubrics/)",
    )
    parser.add_argument(
        "--rubric-model",
        type=str,
        help="Model as provider:model. Defaults to gpt-5.2_2025-12-11 from model_configs/model_rubrics.json if available.",
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
        default="rubrics_output",
        help="Directory for CSV output (default: rubrics_output/)",
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
    # Unified output paths: full rubrics to eval_traces/rubrics_output/
    output_dir = Path(args.output_dir)
    if args.output_dir == "rubrics_output":
        output_dir = REPO_ROOT / "eval_traces" / "rubrics_output"
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
        fixes_path = REPO_ROOT / "fixes" / fix_dir_name
        
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
            rubric_cli.run(args)
            
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
