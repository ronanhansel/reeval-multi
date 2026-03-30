
import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4": (2.50, 10.00),
    "o3-mini": (1.10, 4.40),
    "o3": (1.10, 4.40),
    "o4-mini": (0.15, 0.60),
    "DeepSeek-R1": (0.55, 2.19),
    "gpt-5": (2.50, 15.00),
    "grok": (2.00, 6.00),
    "default": (5.00, 15.00),
}

FOUR_MAIN_BENCHMARKS = {
    "colbench_backend_programming": "ColBench Backend",
    "corebench_hard": "CoreBench Hard",
    "scicode": "SciCode",
    "scienceagentbench": "ScienceAgentBench",
}

PRIMARY_COHORT_MODELS = {
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5",
    "gpt-4.1",
    "o3-mini",
    "o4-mini",
}

TRAIN_MINUTES_PER_RUN = 1.5
A600_HOURLY_RATE = 0.30
SECTION41_FRACTIONS = (0.3, 0.5, 0.7)
RESEARCH_EMBEDDINGS = ("RAW", "PCA", "SAE")
SECTION41_RESEARCH_REGIMES = ("post_bernoulli", "post_beta")
FULL_SWEEP_SEEDS = 50


def get_price(model_name):
    model_name = (model_name or "").lower()
    for key, value in PRICING.items():
        if key.lower() in model_name:
            return value
    return PRICING["default"]


def parse_iso_timestamp(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def get_runtime_stats(data):
    raw_results = data.get("raw_logging_results", [])
    task_durations = []
    trace_starts = []
    trace_ends = []

    if not isinstance(raw_results, list):
        return None, None, 0

    for row in raw_results:
        if not isinstance(row, dict):
            continue
        started_at = parse_iso_timestamp(row.get("started_at"))
        ended_at = parse_iso_timestamp(row.get("ended_at"))
        if started_at and ended_at and ended_at >= started_at:
            duration = (ended_at - started_at).total_seconds()
            task_durations.append(duration)
            trace_starts.append(started_at)
            trace_ends.append(ended_at)

    wall_clock_seconds = None
    if trace_starts and trace_ends:
        wall_clock_seconds = (max(trace_ends) - min(trace_starts)).total_seconds()

    avg_task_seconds = statistics.mean(task_durations) if task_durations else None
    return wall_clock_seconds, avg_task_seconds, len(task_durations)


def format_duration(seconds):
    if seconds is None:
        return "N/A"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_currency(amount):
    return f"${amount:,.2f}"


def format_break_even(value, digits=2):
    if value is None:
        return "N/A"
    if value < 1:
        return "< 1"
    return f"{value:.{digits}f}"


def percentile(values, pct):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def normalize_model_name(model_name):
    if not model_name:
        return "unknown"
    normalized = model_name.lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[1]
    normalized = re.sub(r"_20\d{2}.*$", "", normalized)
    return normalized


def is_primary_cohort_model(model_name):
    return normalize_model_name(model_name) in PRIMARY_COHORT_MODELS


def compute_usage_stats(obj):
    stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "total_cost": 0.0,
    }

    def walk(node):
        if isinstance(node, dict):
            if "usage" in node and "model" in node:
                usage = node.get("usage")
                if isinstance(usage, dict):
                    input_price, output_price = get_price(node.get("model", "unknown"))

                    prompt = usage.get("prompt_tokens", 0) or 0
                    completion = usage.get("completion_tokens", 0) or 0

                    reasoning = 0
                    completion_details = usage.get("completion_tokens_details")
                    if isinstance(completion_details, dict):
                        reasoning = completion_details.get("reasoning_tokens", 0) or 0

                    cached = 0
                    prompt_details = usage.get("prompt_tokens_details")
                    if isinstance(prompt_details, dict):
                        cached = prompt_details.get("cached_tokens", 0) or 0

                    prompt_cost = prompt * input_price / 1_000_000
                    completion_cost = completion * output_price / 1_000_000

                    stats["prompt_tokens"] += prompt
                    stats["completion_tokens"] += completion
                    stats["reasoning_tokens"] += reasoning
                    stats["cached_tokens"] += cached
                    stats["prompt_cost"] += prompt_cost
                    stats["completion_cost"] += completion_cost
                    stats["total_cost"] += prompt_cost + completion_cost

            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return stats


def get_eval_item_count(data):
    raw_eval_results = data.get("raw_eval_results")
    if isinstance(raw_eval_results, list):
        return len(raw_eval_results)
    if isinstance(raw_eval_results, dict):
        eval_result = raw_eval_results.get("eval_result")
        if isinstance(eval_result, list):
            return len(eval_result)
        return len(raw_eval_results)
    return None


def process_trace_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    config = data.get("config", {})
    benchmark = config.get("benchmark_name", "unknown")
    model_name = config.get("agent_args", {}).get("model_name", "unknown")
    usage_stats = compute_usage_stats(data)
    wall_clock_seconds, avg_task_seconds, task_count = get_runtime_stats(data)

    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "benchmark": benchmark,
        "benchmark_label": FOUR_MAIN_BENCHMARKS.get(benchmark, benchmark),
        "model_name": model_name,
        "model_slug": normalize_model_name(model_name),
        "run_id": config.get("run_id"),
        "trace_cost": usage_stats["total_cost"],
        "prompt_tokens": usage_stats["prompt_tokens"],
        "completion_tokens": usage_stats["completion_tokens"],
        "reasoning_tokens": usage_stats["reasoning_tokens"],
        "cached_tokens": usage_stats["cached_tokens"],
        "prompt_cost": usage_stats["prompt_cost"],
        "completion_cost": usage_stats["completion_cost"],
        "wall_clock_seconds": wall_clock_seconds,
        "avg_task_seconds": avg_task_seconds,
        "task_count": task_count,
        "eval_item_count": get_eval_item_count(data),
        "is_primary_cohort": is_primary_cohort_model(model_name),
    }


def load_trace_records(trace_dir, max_workers=None):
    json_files = sorted(glob.glob(os.path.join(trace_dir, "*.json")))
    if not json_files:
        return []

    if max_workers == 1:
        return [process_trace_file(path) for path in json_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_trace_file, json_files, chunksize=4))


def summarize_records(records):
    benchmark_stats = defaultdict(lambda: {
        "trace_count": 0,
        "models": set(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "total_cost": 0.0,
        "trace_runtimes": [],
        "item_count_histogram": Counter(),
    })

    for record in records:
        stats = benchmark_stats[record["benchmark"]]
        stats["trace_count"] += 1
        stats["models"].add(record["model_slug"])
        stats["prompt_tokens"] += record["prompt_tokens"]
        stats["completion_tokens"] += record["completion_tokens"]
        stats["reasoning_tokens"] += record["reasoning_tokens"]
        stats["cached_tokens"] += record["cached_tokens"]
        stats["prompt_cost"] += record["prompt_cost"]
        stats["completion_cost"] += record["completion_cost"]
        stats["total_cost"] += record["trace_cost"]
        if record["wall_clock_seconds"] is not None:
            stats["trace_runtimes"].append(record["wall_clock_seconds"])
        if record["eval_item_count"] is not None:
            stats["item_count_histogram"][record["eval_item_count"]] += 1

    summary = {}
    for benchmark, stats in benchmark_stats.items():
        runtimes = stats["trace_runtimes"]
        summary[benchmark] = {
            "benchmark_label": FOUR_MAIN_BENCHMARKS.get(benchmark, benchmark),
            "trace_count": stats["trace_count"],
            "unique_models": sorted(stats["models"]),
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "reasoning_tokens": stats["reasoning_tokens"],
            "cached_tokens": stats["cached_tokens"],
            "prompt_cost": stats["prompt_cost"],
            "completion_cost": stats["completion_cost"],
            "total_cost": stats["total_cost"],
            "avg_cost": stats["total_cost"] / stats["trace_count"] if stats["trace_count"] else None,
            "runtime_seconds_total": sum(runtimes),
            "avg_runtime_seconds": statistics.mean(runtimes) if runtimes else None,
            "median_runtime_seconds": statistics.median(runtimes) if runtimes else None,
            "p90_runtime_seconds": percentile(runtimes, 0.90),
            "min_runtime_seconds": min(runtimes) if runtimes else None,
            "max_runtime_seconds": max(runtimes) if runtimes else None,
            "item_count_histogram": dict(stats["item_count_histogram"]),
        }
    return summary


def infer_tau_grid(reproduce_script_path):
    if not reproduce_script_path or not os.path.exists(reproduce_script_path):
        return []
    with open(reproduce_script_path, "r") as f:
        text = f.read()
    match = re.search(r'SHARED_TAUS="([^"]+)"', text)
    if not match:
        return []
    return match.group(1).split()


def compute_setup_cost(seed_config_runs):
    gpu_minutes = seed_config_runs * TRAIN_MINUTES_PER_RUN
    gpu_hours = gpu_minutes / 60.0
    dollars = gpu_hours * A600_HOURLY_RATE
    return {
        "seed_config_runs": seed_config_runs,
        "gpu_minutes": gpu_minutes,
        "gpu_hours": gpu_hours,
        "dollars": dollars,
    }


def compute_section41_economics(records, reproduce_script_path):
    main_records = [r for r in records if r["benchmark"] in FOUR_MAIN_BENCHMARKS]
    primary_records = [r for r in main_records if r["is_primary_cohort"]]
    cost_records = primary_records if primary_records else main_records

    benchmark_summary = summarize_records(cost_records)

    full_eval_cost = sum(
        benchmark_summary[benchmark]["avg_cost"]
        for benchmark in FOUR_MAIN_BENCHMARKS
        if benchmark in benchmark_summary
    )
    full_eval_runtime_seconds = sum(
        benchmark_summary[benchmark]["avg_runtime_seconds"]
        for benchmark in FOUR_MAIN_BENCHMARKS
        if benchmark in benchmark_summary
    )

    tau_grid = infer_tau_grid(reproduce_script_path)
    research_seed_config_runs = (
        len(RESEARCH_EMBEDDINGS)
        * len(SECTION41_RESEARCH_REGIMES)
        * len(tau_grid)
        * FULL_SWEEP_SEEDS
    )
    deploy_seed_config_runs = 1

    setup_research = compute_setup_cost(research_seed_config_runs)
    setup_deploy = compute_setup_cost(deploy_seed_config_runs)

    # Deployment analysis treats inference as operationally negligible relative to benchmark execution.
    inference_cost = 0.0
    inference_runtime_seconds = 0.0

    marginal_costs = {}
    break_even = {"research": {}, "deploy": {}}
    for fraction in SECTION41_FRACTIONS:
        per_model_cost = (fraction * full_eval_cost) + inference_cost
        per_model_runtime_seconds = (fraction * full_eval_runtime_seconds) + inference_runtime_seconds
        marginal_costs[fraction] = {
            "cost": per_model_cost,
            "runtime_seconds": per_model_runtime_seconds,
        }

        savings = full_eval_cost - per_model_cost
        break_even["research"][fraction] = setup_research["dollars"] / savings if savings > 0 else None
        break_even["deploy"][fraction] = setup_deploy["dollars"] / savings if savings > 0 else None

    return {
        "records_used": cost_records,
        "used_primary_cohort_filter": bool(primary_records),
        "benchmark_summary": benchmark_summary,
        "full_eval_cost": full_eval_cost,
        "full_eval_runtime_seconds": full_eval_runtime_seconds,
        "tau_grid": tau_grid,
        "setup_research": setup_research,
        "setup_deploy": setup_deploy,
        "inference_cost": inference_cost,
        "inference_runtime_seconds": inference_runtime_seconds,
        "marginal_costs": marginal_costs,
        "break_even": break_even,
    }


def compute_rollout_footprint(records):
    main_records = [r for r in records if r["benchmark"] in FOUR_MAIN_BENCHMARKS]
    summary = summarize_records(main_records)
    total_cost = sum(row["total_cost"] for row in summary.values())
    total_runtime_seconds = sum(row["runtime_seconds_total"] for row in summary.values())
    total_traces = sum(row["trace_count"] for row in summary.values())
    return {
        "benchmark_summary": summary,
        "total_cost": total_cost,
        "total_runtime_seconds": total_runtime_seconds,
        "total_traces": total_traces,
    }


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":---"] + ["---:" for _ in headers[1:]]) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_pricing_table():
    rows = []
    for model_key, (input_price, output_price) in PRICING.items():
        rows.append([model_key, f"${input_price:.2f}", f"${output_price:.2f}"])
    return markdown_table(
        ["Model family", "Input price / 1M tokens", "Output price / 1M tokens"],
        rows,
    )


def build_full_stats_markdown(summary):
    ordered_benchmarks = sorted(summary)
    headers = [
        "Benchmark",
        "Traces",
        "Unique Models",
        "Input Tokens",
        "Input Cost ($)",
        "Output Tokens",
        "Output Cost ($)",
        "Reasoning Tokens",
        "Cached Tokens",
        "Total Cost ($)",
        "Cumulative Trace Runtime",
        "Avg Trace Runtime",
        "Median Trace Runtime",
        "P90 Trace Runtime",
        "Min Trace Runtime",
        "Max Trace Runtime",
    ]
    rows = []
    for benchmark in ordered_benchmarks:
        row = summary[benchmark]
        rows.append([
            row["benchmark_label"],
            str(row["trace_count"]),
            str(len(row["unique_models"])),
            f"{row['prompt_tokens']:,}",
            format_currency(row["prompt_cost"]),
            f"{row['completion_tokens']:,}",
            format_currency(row["completion_cost"]),
            f"{row['reasoning_tokens']:,}",
            f"{row['cached_tokens']:,}",
            format_currency(row["total_cost"]),
            format_duration(row["runtime_seconds_total"]),
            format_duration(row["avg_runtime_seconds"]),
            format_duration(row["median_runtime_seconds"]),
            format_duration(row["p90_runtime_seconds"]),
            format_duration(row["min_runtime_seconds"]),
            format_duration(row["max_runtime_seconds"]),
        ])
    return markdown_table(headers, rows)


def build_compact_stats_markdown(summary):
    ordered_benchmarks = sorted(summary)
    headers = [
        "Benchmark",
        "Traces",
        "Models",
        "Input Tokens",
        "Input Cost ($)",
        "Output Tokens",
        "Output Cost ($)",
        "Reasoning",
        "Total Cost ($)",
    ]
    rows = []
    for benchmark in ordered_benchmarks:
        row = summary[benchmark]
        rows.append([
            row["benchmark_label"],
            str(row["trace_count"]),
            str(len(row["unique_models"])),
            f"{row['prompt_tokens']:,}",
            format_currency(row["prompt_cost"]),
            f"{row['completion_tokens']:,}",
            format_currency(row["completion_cost"]),
            f"{row['reasoning_tokens']:,}",
            format_currency(row["total_cost"]),
        ])
    return markdown_table(headers, rows)


def build_constants_markdown(trace_dir, economics):
    tau_count = len(economics["tau_grid"])
    constants = [
        f"- Trace directory: `{trace_dir}`",
        "- Trace unit: one JSON file corresponds to one complete model-on-benchmark run over that benchmark's full item set.",
        "- Four main benchmarks: ColBench Backend, CoreBench Hard, SciCode, and ScienceAgentBench.",
        f"- Primary 8-model cohort: {', '.join(sorted(PRIMARY_COHORT_MODELS))}.",
        f"- Training time constant: 1 seed × 1 config × 1000 epochs = {TRAIN_MINUTES_PER_RUN} minutes.",
        f"- GPU type: NVIDIA A600.",
        f"- GPU rate: {format_currency(A600_HOURLY_RATE)} / hour.",
        f"- Section 4.1 observed fractions: {SECTION41_FRACTIONS}.",
        f"- Research embeddings: {', '.join(RESEARCH_EMBEDDINGS)}.",
        f"- Section 4.1 research regimes: {', '.join(SECTION41_RESEARCH_REGIMES)}.",
        f"- Full sweep seeds: {FULL_SWEEP_SEEDS}.",
        f"- Tau grid count: {tau_count} values recovered from `model/reproduce.sh`.",
        "- Token prices are taken directly from the `PRICING` table in `analyze_traces.py`.",
    ]
    return "\n".join(constants)


def build_section41_markdown(trace_dir, economics):
    lines = []
    lines.append("# Section 4.1 Economic Analysis")
    lines.append("")
    lines.append(
        "This section supports the main economic claim of Section 4.1: once ARAF has been calibrated, it can reduce the **marginal** cost of evaluating future models by observing only a subset of benchmark items and predicting the rest. The calculations below intentionally focus on the core Section 4.1 deployment story and do **not** use the full remediation and repeated-sampling pipeline as the primary economic comparator."
    )
    lines.append("")
    lines.append("## 1. Scope and costing convention")
    lines.append("")
    lines.append("- **Full evaluation cost**: trace-derived API/token cost and wall-clock runtime of running one new model across the four main benchmarks.")
    lines.append("- **ARAF setup cost**: one-time model training / selection cost on the NVIDIA A600.")
    lines.append("- **ARAF per-new-model cost**: cost of running one new model on only a fraction `p` of items, with inference treated as operationally negligible because learned parameters are fixed at deployment.")
    lines.append("- **Repeated rollout / repeated-sampling cost**: reported separately as an execution-footprint section; it is not part of the main Section 4.1 break-even calculation.")
    lines.append("- **Excluded costs in rollout footprint**: judging, grading, item-fixing, and the Claude Code 4.5 OPUS remediation-targeting agent cost are excluded unless they are present in the trace logs. They are not added here.")
    lines.append("")
    lines.append("This costing convention matches the paper's main contribution in Section 4.1: reducing the marginal cost of future model evaluation through partial observation and prediction, rather than claiming that the entire benchmark-stabilization pipeline is cheap.")
    lines.append("")
    lines.append("## 2. Constants and assumptions")
    lines.append("")
    lines.append(build_constants_markdown(trace_dir, economics))
    lines.append("")
    lines.append("### Token pricing table used by the script")
    lines.append("")
    lines.append(build_pricing_table())
    lines.append("")
    lines.append("## 3. Definition of key terms")
    lines.append("")
    lines.append("- **Full evaluation**: executing one new model on all items in the four-benchmark suite.")
    lines.append("- **Research-faithful setup cost**: reproducing the full Section 4.1 ARAF sweep across all 3 embedding families, both canonical Section 4.1 regimes, all tau values, and all 50 seeds.")
    lines.append("- **Deployment-faithful setup cost**: training one final already-chosen ARAF configuration after design choices have been fixed from prior study; this is a lower-bound deployment setup cost.")
    lines.append("- **Marginal per-new-model cost**: the cost of evaluating a future model under the deployed ARAF protocol, where only a fraction `p` of items is executed and the rest are predicted.")
    lines.append("- **Break-even point**: the number of future models required for the one-time setup cost to become cheaper than repeated full evaluation.")
    lines.append("- **Repeated rollout execution cost**: the trace-derived execution footprint of repeated benchmark runs used to build repeated-sampling/post-revision analysis artifacts.")
    lines.append("- **Execution-only cost**: benchmark-execution cost only, excluding judging, grading, and remediation-targeting overhead not present in the trace logs.")
    lines.append("")
    lines.append("Deployment-faithful is intentionally optimistic and should be interpreted as a lower bound. Research-faithful is the conservative number and is the better choice for main-paper reporting.")
    lines.append("")
    lines.append("## 4. Formulas used")
    lines.append("")
    lines.append("Full evaluation cost:")
    lines.append("")
    lines.append("`C_full_eval = \\sum_b \\operatorname{avg\\_cost}(b)` over the four main benchmarks.")
    lines.append("")
    lines.append("ARAF setup cost:")
    lines.append("")
    lines.append("`GPU_hours = seed_config_runs × 1.5 / 60`")
    lines.append("")
    lines.append("`C_setup = GPU_hours × 0.30`")
    lines.append("")
    lines.append("Per-new-model ARAF cost:")
    lines.append("")
    lines.append("`C_ARAF(p) = p × C_full_eval + C_infer`")
    lines.append("")
    lines.append("Break-even point:")
    lines.append("")
    lines.append("`A_BE(p) = C_setup / (C_full_eval - C_ARAF(p))`")
    lines.append("")
    lines.append("Repeated rollout execution footprint:")
    lines.append("")
    lines.append("`C_rollout_total = \\sum_t \\operatorname{trace\\_cost}(t)` over repeated execution traces included in the logs.")
    lines.append("")
    lines.append("`T_rollout_total = \\sum_t \\operatorname{wall\\_clock}(t)` over the same traces.")
    lines.append("")
    lines.append("In this report, `C_infer` is treated as negligible because deployment uses fixed learned parameters and inference is operationally small relative to benchmark execution cost. The main approximation is therefore:")
    lines.append("")
    lines.append("`C_ARAF(p) ≈ p × C_full_eval`")
    lines.append("")
    lines.append("## 5. Full evaluation cost derivation")
    lines.append("")
    if economics["used_primary_cohort_filter"]:
        lines.append(
            "The benchmark averages below are computed over the primary 8-model cohort when available. This keeps the Section 4.1 economics aligned with the controlled cohort used in the main paper."
        )
    else:
        lines.append(
            "Primary-cohort filtering was unavailable, so the four-benchmark trace averages below use all available traces."
        )
    lines.append("")
    headers = ["Benchmark", "Full-run traces", "Avg cost / run", "Avg runtime / run"]
    rows = []
    for benchmark in FOUR_MAIN_BENCHMARKS:
        row = economics["benchmark_summary"].get(benchmark)
        if not row:
            continue
        rows.append([
            row["benchmark_label"],
            str(row["trace_count"]),
            format_currency(row["avg_cost"]),
            format_duration(row["avg_runtime_seconds"]),
        ])
    rows.append([
        "**Total 4-benchmark suite**",
        "",
        f"**{format_currency(economics['full_eval_cost'])}**",
        f"**{format_duration(economics['full_eval_runtime_seconds'])}**",
    ])
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append("## 6. ARAF setup cost derivation")
    lines.append("")
    lines.append(
        f"Research-faithful seed-config runs = 3 embeddings × 2 regimes × {len(economics['tau_grid'])} tau values × 50 seeds = {economics['setup_research']['seed_config_runs']:,}."
    )
    lines.append("")
    lines.append(
        f"Research-faithful GPU-hours = {economics['setup_research']['seed_config_runs']:,} × {TRAIN_MINUTES_PER_RUN} / 60 = {economics['setup_research']['gpu_hours']:,.3f}."
    )
    lines.append("")
    lines.append(
        f"Research-faithful dollar cost = {economics['setup_research']['gpu_hours']:,.3f} × {A600_HOURLY_RATE:.2f} = {format_currency(economics['setup_research']['dollars'])}."
    )
    lines.append("")
    lines.append(
        "Deployment-faithful setup means one final already-chosen configuration after model-design decisions have been fixed from prior study."
    )
    lines.append("")
    lines.append(
        f"Deployment-faithful seed-config runs = {economics['setup_deploy']['seed_config_runs']}."
    )
    lines.append("")
    lines.append(
        f"Deployment-faithful GPU-hours = {economics['setup_deploy']['seed_config_runs']} × {TRAIN_MINUTES_PER_RUN} / 60 = {economics['setup_deploy']['gpu_hours']:,.3f}."
    )
    lines.append("")
    lines.append(
        f"Deployment-faithful dollar cost = {economics['setup_deploy']['gpu_hours']:,.3f} × {A600_HOURLY_RATE:.2f} = {format_currency(economics['setup_deploy']['dollars'])}."
    )
    lines.append("")
    lines.append(
        "A more conservative deployment estimate could use multiple final refits, but the one-run number is retained here as a lower-bound deployment setup cost."
    )
    lines.append("")
    lines.append("## 7. Marginal ARAF cost by observed fraction")
    lines.append("")
    lines.append(
        "Inference cost is treated as negligible. The deployed per-new-model cost therefore scales approximately with the observed item fraction."
    )
    lines.append("")
    headers = ["Observed fraction p", "Per-new-model cost", "Per-new-model runtime"]
    rows = []
    for fraction in SECTION41_FRACTIONS:
        marginal = economics["marginal_costs"][fraction]
        rows.append([
            f"{fraction:.1f}",
            format_currency(marginal["cost"]),
            format_duration(marginal["runtime_seconds"]),
        ])
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append("## 8. Break-even analysis")
    lines.append("")
    lines.append(
        "Research-faithful break-even numbers are conservative and are the best choice for the main paper. Deployment-faithful break-even numbers are lower-bound operational estimates and are better suited to the appendix or a footnote."
    )
    lines.append("")
    headers = ["Observed fraction p", "Break-even (research-faithful)", "Break-even (deployment-faithful)"]
    rows = []
    for fraction in SECTION41_FRACTIONS:
        rows.append([
            f"{fraction:.1f}",
            format_break_even(economics["break_even"]["research"][fraction], 2),
            format_break_even(economics["break_even"]["deploy"][fraction], 2),
        ])
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append("A deployment-faithful break-even value below 1 means that even a single future evaluation is enough to amortize the minimal lower-bound setup cost. This is why the research-faithful number is more informative for conservative reporting.")
    lines.append("")
    return "\n".join(lines)


def build_rollout_markdown(rollout):
    lines = []
    lines.append("# Repeated Rollout / Repeated Sampling Execution Footprint")
    lines.append("")
    lines.append(
        "This section reports the total execution footprint of the repeated benchmark runs captured in the trace logs. It is included for financial transparency and appendix reproducibility. It is **not** the same as the marginal per-new-model evaluation claim in Section 4.1."
    )
    lines.append("")
    lines.append(
        "This section is **execution-only**. It excludes judging, grading, item-remediation targeting, and the Claude Code 4.5 OPUS agent cost unless those costs are explicitly present in the trace logs. They are not added here."
    )
    lines.append("")
    lines.append(
        "If the trace logs do not allow perfect separation between single-run and repeated-run execution, the totals below should be interpreted as the aggregate four-benchmark execution footprint represented in the available logs."
    )
    lines.append("")
    headers = ["Benchmark", "Traces", "Total Cost ($)", "Total Runtime"]
    rows = []
    for benchmark in FOUR_MAIN_BENCHMARKS:
        row = rollout["benchmark_summary"].get(benchmark)
        if not row:
            continue
        rows.append([
            row["benchmark_label"],
            str(row["trace_count"]),
            format_currency(row["total_cost"]),
            format_duration(row["runtime_seconds_total"]),
        ])
    rows.append([
        "**Total footprint**",
        str(rollout["total_traces"]),
        f"**{format_currency(rollout['total_cost'])}**",
        f"**{format_duration(rollout['total_runtime_seconds'])}**",
    ])
    lines.append(markdown_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def build_interpretation_markdown(economics, rollout):
    min_be = min(
        v for v in economics["break_even"]["research"].values()
        if v is not None
    )
    max_be = max(
        v for v in economics["break_even"]["research"].values()
        if v is not None
    )
    lines = []
    lines.append("# Interpretation for the Paper")
    lines.append("")
    lines.append(
        "The main economic claim is about **marginal evaluation cost**. Once ARAF has been calibrated, a future model does not need to be executed on every benchmark item. Instead, one can evaluate only a fraction of items and predict the remainder, reducing per-model execution cost."
    )
    lines.append("")
    lines.append(
        "Section 4.1 supports that claim because ARAF retains useful predictive performance when observed item coverage is reduced. Under the trace-derived execution costs in this report, a full evaluation of one new model across the four benchmarks costs "
        f"{format_currency(economics['full_eval_cost'])} and {format_duration(economics['full_eval_runtime_seconds'])}, while deployed ARAF evaluation reduces that to "
        + ", ".join(
            f"{format_currency(economics['marginal_costs'][p]['cost'])} at p={p:.1f}"
            for p in SECTION41_FRACTIONS
        )
        + f". Using the conservative research-faithful setup cost, the break-even point falls between {min_be:.2f} and {max_be:.2f} future models."
    )
    lines.append("")
    lines.append(
        "The repeated-rollout execution footprint is reported separately because it serves a different purpose. It makes the broader evaluation pipeline financially transparent, but it should not be interpreted as the main source of cost savings. Remediation and repeated sampling are benchmark-infrastructure investments for measurement stabilization, whereas the core Section 4.1 claim is that ARAF can reduce the marginal cost of future model evaluation."
    )
    lines.append("")
    return "\n".join(lines)


def build_appendix_note(economics, rollout):
    min_be = min(
        v for v in economics["break_even"]["research"].values()
        if v is not None
    )
    max_be = max(
        v for v in economics["break_even"]["research"].values()
        if v is not None
    )
    return (
        "# Appendix-ready note\n\n"
        f"Using trace-derived benchmark execution costs, a full evaluation of one new model across the four main benchmarks costs approximately {format_currency(economics['full_eval_cost'])} and {format_duration(economics['full_eval_runtime_seconds'])}. "
        f"Under partial-observation deployment, the corresponding per-new-model ARAF cost is approximately {format_currency(economics['marginal_costs'][0.3]['cost'])}, {format_currency(economics['marginal_costs'][0.5]['cost'])}, and {format_currency(economics['marginal_costs'][0.7]['cost'])} at observed fractions p=0.3, 0.5, and 0.7, respectively, treating inference cost as negligible. "
        f"Using the conservative research-faithful setup cost, the resulting break-even point lies between {min_be:.2f} and {max_be:.2f} future models. "
        f"For financial transparency, the trace logs also report a broader repeated-rollout execution footprint of {format_currency(rollout['total_cost'])} and {format_duration(rollout['total_runtime_seconds'])} across the captured four-benchmark traces. "
        "These rollout-footprint totals are execution-only and exclude judging, grading, and Claude Code 4.5 OPUS remediation-targeting costs."
    )


def write_report(summary, economics, rollout, output_file, trace_dir):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    parts = []
    parts.append("# Consolidated Experiment Trace and Cost Statistics by Benchmark")
    parts.append("")
    parts.append(
        "The statistics below are aggregated from the raw trace logs. Cached tokens are reported separately. Cost totals are computed using the pricing logic implemented in `analyze_traces.py`; the script does not apply cached-token discounting, so cached usage is tracked for transparency rather than subtracted from the compact totals."
    )
    parts.append("")
    parts.append(build_full_stats_markdown(summary))
    parts.append("")
    parts.append("## Compact appendix-ready benchmark statistics table")
    parts.append("")
    parts.append(
        "This compact table is intended to map directly to a LaTeX appendix table titled **Consolidated Experiment Trace and Cost Statistics by Benchmark (Aggregated without counting cached tokens)**. Cached tokens are reported in the full table above but are not used to discount the compact cost totals."
    )
    parts.append("")
    parts.append(build_compact_stats_markdown(summary))
    parts.append("")
    parts.append(build_section41_markdown(trace_dir, economics))
    parts.append(build_rollout_markdown(rollout))
    parts.append(build_interpretation_markdown(economics, rollout))
    parts.append(build_appendix_note(economics, rollout))
    parts.append("")
    with open(output_file, "w") as f:
        f.write("\n".join(parts))


def analyze_traces(trace_dir, output_file):
    records = load_trace_records(trace_dir)
    if not records:
        print(f"No trace files found in {trace_dir}")
        return

    print(f"Found {len(records)} trace files. Processing...")

    summary = summarize_records(records)
    total_cost = sum(row["total_cost"] for row in summary.values())
    total_runtime = sum(row["runtime_seconds_total"] for row in summary.values())
    total_runtime_count = sum(row["trace_count"] for row in summary.values())

    print("\n" + "=" * 60)
    print("           CONSOLIDATED EXPERIMENT STATISTICS")
    print("=" * 60)
    print(f"Total Traces Processed:  {len(records)}")
    print(f"Total Benchmarks:        {len(summary)}")
    print(f"Total Estimated Cost:    {format_currency(total_cost)}")
    print(f"Cumulative Runtime:      {format_duration(total_runtime)}")
    if total_runtime_count:
        print(f"Avg Trace Runtime:       {format_duration(total_runtime / total_runtime_count)}")
    print("=" * 60)

    reproduce_script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model",
        "reproduce.sh",
    )
    economics = compute_section41_economics(records, reproduce_script_path)
    rollout = compute_rollout_footprint(records)

    print("\n" + "=" * 60)
    print("                SECTION 4.1 COST OF EVALUATION")
    print("=" * 60)
    print(f"Primary 8-model cohort filter used: {economics['used_primary_cohort_filter']}")
    for benchmark in FOUR_MAIN_BENCHMARKS:
        row = economics["benchmark_summary"].get(benchmark)
        if not row:
            continue
        print(
            f"{row['benchmark_label']:<22} "
            f"avg cost={format_currency(row['avg_cost']):>10} "
            f"avg runtime={format_duration(row['avg_runtime_seconds']):>12} "
            f"traces={row['trace_count']}"
        )
    print(
        f"Total 4-benchmark suite   avg cost={format_currency(economics['full_eval_cost']):>10} "
        f"avg runtime={format_duration(economics['full_eval_runtime_seconds']):>12}"
    )
    print(
        f"Research-faithful setup   runs={economics['setup_research']['seed_config_runs']:,} "
        f"gpu_hours={economics['setup_research']['gpu_hours']:.3f} "
        f"cost={format_currency(economics['setup_research']['dollars'])}"
    )
    print(
        f"Deployment setup          runs={economics['setup_deploy']['seed_config_runs']:,} "
        f"gpu_hours={economics['setup_deploy']['gpu_hours']:.3f} "
        f"cost={format_currency(economics['setup_deploy']['dollars'])}"
    )
    for fraction in SECTION41_FRACTIONS:
        marginal = economics["marginal_costs"][fraction]
        print(
            f"ARAF p={fraction:.1f}               cost={format_currency(marginal['cost'])} "
            f"runtime={format_duration(marginal['runtime_seconds'])} "
            f"BE_research={format_break_even(economics['break_even']['research'][fraction], 2)} "
            f"BE_deploy={format_break_even(economics['break_even']['deploy'][fraction], 2)}"
        )

    print("\n" + "=" * 60)
    print("           REPEATED ROLLOUT EXECUTION FOOTPRINT")
    print("=" * 60)
    print(f"Total rollout traces:     {rollout['total_traces']}")
    print(f"Total rollout cost:       {format_currency(rollout['total_cost'])}")
    print(f"Total rollout runtime:    {format_duration(rollout['total_runtime_seconds'])}")

    write_report(summary, economics, rollout, output_file, trace_dir)
    print(f"\nConsolidated report exported to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace_dir",
        nargs="?",
        default="/Users/ronan/Developer/agent-eval/item-editor/eval_traces/traces",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="/Users/ronan/Developer/agent-eval/model/result/statistics.md",
    )
    args = parser.parse_args()
    analyze_traces(args.trace_dir, args.output_file)
