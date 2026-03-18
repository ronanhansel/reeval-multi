import os
import json
import glob
import sys
from collections import defaultdict

# Pricing per 1M tokens (Input, Output)
PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4": (2.50, 10.00),
    "o3-mini": (1.10, 4.40),
    "o3": (1.10, 4.40),      # Maps o3 to o3-mini pricing
    "o4-mini": (0.15, 0.60),
    "DeepSeek-R1": (0.55, 2.19),
    "gpt-5": (2.50, 15.00),  # Top-tier version pricing
    "grok": (2.00, 6.00),    # Grok 4.20 Beta pricing
    "default": (5.00, 15.00)
}

def get_price(model_name):
    for key in PRICING:
        if key in model_name.lower():
            return PRICING[key]
    return PRICING["default"]

def analyze_traces(trace_dir, output_file):
    # Stats for models (tokens/cost)
    model_stats = defaultdict(lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
    })

    # Stats for benchmarks
    benchmark_stats = defaultdict(lambda: {
        "num_traces": 0,
        "models": set(),
        "total_accuracy": 0.0,
        "total_correctness": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "total_cost": 0.0
    })

    json_files = glob.glob(os.path.join(trace_dir, "*.json"))
    
    if not json_files:
        print(f"No trace files found in {trace_dir}")
        return

    print(f"Found {len(json_files)} trace files. Processing...")

    processed_count = 0
    for file_path in json_files:
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{len(json_files)} files (Latest: {os.path.basename(file_path)})...")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            config = data.get("config", {})
            results = data.get("results", {})
            
            benchmark = config.get("benchmark_name", "unknown")
            # Extract model name
            model = config.get("agent_args", {}).get("model_name", "unknown")
            
            accuracy = results.get("accuracy", results.get("success_rate", 0.0)) or 0.0
            correctness = results.get("average_correctness", 0.0) or 0.0
            
            benchmark_stats[benchmark]["num_traces"] += 1
            benchmark_stats[benchmark]["models"].add(model)
            benchmark_stats[benchmark]["total_accuracy"] += accuracy
            benchmark_stats[benchmark]["total_correctness"] += correctness
            
            def find_usage(obj):
                if isinstance(obj, dict):
                    if "usage" in obj and "model" in obj:
                        model = obj.get("model", "unknown")
                        usage = obj["usage"]
                        
                        if isinstance(usage, dict):
                            input_price, output_price = get_price(model)
                            
                            prompt = usage.get("prompt_tokens", 0) or 0
                            completion = usage.get("completion_tokens", 0) or 0
                            
                            # Handle details if available
                            reasoning = 0
                            details = usage.get("completion_tokens_details")
                            if isinstance(details, dict):
                                reasoning = details.get("reasoning_tokens", 0) or 0
                            
                            cached = 0
                            p_details = usage.get("prompt_tokens_details")
                            if isinstance(p_details, dict):
                                cached = p_details.get("cached_tokens", 0) or 0
                            
                            benchmark_stats[benchmark]["prompt_tokens"] += prompt
                            benchmark_stats[benchmark]["completion_tokens"] += completion
                            benchmark_stats[benchmark]["reasoning_tokens"] += reasoning
                            benchmark_stats[benchmark]["cached_tokens"] += cached
                            
                            # Cost calculation per 1M tokens
                            p_cost = (prompt * input_price / 1_000_000)
                            c_cost = (completion * output_price / 1_000_000)
                            
                            benchmark_stats[benchmark]["prompt_cost"] += p_cost
                            benchmark_stats[benchmark]["completion_cost"] += c_cost
                            benchmark_stats[benchmark]["total_cost"] += (p_cost + c_cost)
                        
                    for key, value in obj.items():
                        find_usage(value)
                elif isinstance(obj, list):
                    for item in obj:
                        find_usage(item)

            find_usage(data)
            
        except Exception as e:
            # Silently skip errors for now or log them properly
            pass

    # Generate Totals for Console
    total_agg = {
        "prompt": sum(s["prompt_tokens"] for s in benchmark_stats.values()),
        "completion": sum(s["completion_tokens"] for s in benchmark_stats.values()),
        "reasoning": sum(s["reasoning_tokens"] for s in benchmark_stats.values()),
        "cached": sum(s["cached_tokens"] for s in benchmark_stats.values()),
        "cost": sum(s["total_cost"] for s in benchmark_stats.values())
    }

    print("\n" + "="*60)
    print("           CONSOLIDATED EXPERIMENT STATISTICS")
    print("="*60)
    print(f"Total Traces Processed:  {processed_count}")
    print(f"Total Benchmarks:        {len(benchmark_stats)}")
    print(f"Total Estimated Cost:    ${total_agg['cost']:.4f}")
    print("="*60 + "\n")

    # Generate Markdown Output
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write("# Experiment Appendix: Consolidated Statistics Table\n\n")
            
            f.write("| Benchmark | Traces | Unique Models | Input Tokens | Input Cost ($) | Output Tokens | Output Cost ($) | Reasoning | Cached | Total Cost ($) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            for b_name, b_s in sorted(benchmark_stats.items()):
                f.write(f"| {b_name} | {b_s['num_traces']} | {len(b_s['models'])} | "
                        f"{b_s['prompt_tokens']:,} | ${b_s['prompt_cost']:.2f} | "
                        f"{b_s['completion_tokens']:,} | ${b_s['completion_cost']:.2f} | "
                        f"{b_s['reasoning_tokens']:,} | {b_s['cached_tokens']:,} | "
                        f"${b_s['total_cost']:.2f} |\n")

        print(f"Consolidated statistics table exported to {output_file}")

if __name__ == "__main__":
    # Corrected paths after moving to data-collection
    # traces remain in their original location
    TRACE_DIR = "/Users/ronan/Developer/agent-eval/item-editor/eval_traces/traces"
    OUTPUT_FILE = "/Users/ronan/Developer/agent-eval/model/result/statistics.md"
    
    if len(sys.argv) > 1:
        TRACE_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
        
    analyze_traces(TRACE_DIR, OUTPUT_FILE)
