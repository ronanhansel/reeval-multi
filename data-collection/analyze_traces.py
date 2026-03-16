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
    "o4-mini": (0.15, 0.60),
    "DeepSeek-R1": (0.55, 2.19),
    "gpt-5": (5.00, 15.00),  # Placeholder
    "grok": (2.00, 10.00),   # Placeholder
    "default": (5.00, 15.00)
}

def get_price(model_name):
    for key in PRICING:
        if key in model_name.lower():
            return PRICING[key]
    return PRICING["default"]

def analyze_traces(trace_dir, output_file):
    stats = defaultdict(lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0
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
            
            def find_usage(obj):
                if isinstance(obj, dict):
                    if "usage" in obj and "model" in obj:
                        model = obj.get("model", "unknown")
                        usage = obj["usage"]
                        
                        if isinstance(usage, dict):
                            input_price, output_price = get_price(model)
                            
                            prompt = usage.get("prompt_tokens", 0) or 0
                            completion = usage.get("completion_tokens", 0) or 0
                            total = usage.get("total_tokens", 0) or 0
                            
                            # Handle details if available
                            reasoning = 0
                            details = usage.get("completion_tokens_details")
                            if isinstance(details, dict):
                                reasoning = details.get("reasoning_tokens", 0) or 0
                            
                            cached = 0
                            p_details = usage.get("prompt_tokens_details")
                            if isinstance(p_details, dict):
                                cached = p_details.get("cached_tokens", 0) or 0
                            
                            stats[model]["prompt_tokens"] += prompt
                            stats[model]["completion_tokens"] += completion
                            stats[model]["reasoning_tokens"] += reasoning
                            stats[model]["cached_tokens"] += cached
                            stats[model]["total_tokens"] += total
                            
                            # Cost calculation per 1M tokens
                            cost = (prompt * input_price / 1_000_000) + (completion * output_price / 1_000_000)
                            stats[model]["cost"] += cost
                        
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
        "prompt": sum(s["prompt_tokens"] for s in stats.values()),
        "completion": sum(s["completion_tokens"] for s in stats.values()),
        "reasoning": sum(s["reasoning_tokens"] for s in stats.values()),
        "cached": sum(s["cached_tokens"] for s in stats.values()),
        "total": sum(s["total_tokens"] for s in stats.values()),
        "cost": sum(s["cost"] for s in stats.values())
    }

    print("\n" + "="*50)
    print("      AGGREGATED TRACE STATISTICS")
    print("="*50)
    print(f"Total Files Processed: {processed_count}")
    print(f"Total Tokens:          {total_agg['total']:,}")
    print(f"Prompt Tokens:         {total_agg['prompt']:,}")
    print(f"Completion Tokens:     {total_agg['completion']:,}")
    print(f"  (of which reasoning: {total_agg['reasoning']:,})")
    print(f"Cached Tokens:         {total_agg['cached']:,}")
    print(f"Estimated Total Cost:  ${total_agg['cost']:.4f}")
    print("="*50 + "\n")

    # Generate Markdown Output
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write("# Trace Statistics Summary\n\n")
            f.write("| Model | Total Tokens | Prompt | Completion | Reasoning | Cached | Estimated Cost ($) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            # Sort by cost descending
            sorted_models = sorted(stats.items(), key=lambda x: x[1]["cost"], reverse=True)
            
            for model, s in sorted_models:
                f.write(f"| {model} | {s['total_tokens']:,} | {s['prompt_tokens']:,} | {s['completion_tokens']:,} | {s['reasoning_tokens']:,} | {s['cached_tokens']:,} | ${s['cost']:.4f} |\n")
                
            f.write(f"\n**Total Aggregate Cost: ${total_agg['cost']:.4f}**\n")
        print(f"Detailed statistics exported to {output_file}")

if __name__ == "__main__":
    # Corrected paths after moving to data-collection
    # traces remain in their original location
    TRACE_DIR = "/Users/ronan/Developer/agent-eval/item-editor/eval_traces/traces"
    OUTPUT_FILE = "/Users/ronan/Developer/agent-eval/item-editor/eval_traces/result/statistics.md"
    
    if len(sys.argv) > 1:
        TRACE_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
        
    analyze_traces(TRACE_DIR, OUTPUT_FILE)
