import pandas as pd
import io

core_bench_csv = """Rank,Scaffold,Primary Model,Verified,Accuracy,Cost (USD),Runs,Traces
1,CORE-Agent,Claude Opus 4.1 (August 2025),Yes,51.11%,$412.42,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754492675_UPLOAD.zip?download=true
2,CORE-Agent,Claude Sonnet 4.5 High (September 2025),Yes,44.44%,$92.34,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet45high_1759330487_UPLOAD.zip?download=true
3,CORE-Agent,Claude Opus 4.5 High (November 2025),Yes,42.22%,$152.66,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_core_agent_opus_45_high_1764027725_UPLOAD.zip?download=true
4,CORE-Agent,Claude Opus 4.5 (November 2025),Yes,42.22%,$168.99,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_core_agent_opus_45_1764027531_UPLOAD.zip?download=true
5,CORE-Agent,Claude Opus 4.1 High (August 2025),Yes,42.22%,$509.95,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754539779_UPLOAD.zip?download=true
6,CORE-Agent,Gemini 3 Pro Preview High (November 2025),Yes,40.00%,$86.60,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_core_agent_1763842609_UPLOAD.zip?download=true
7,HAL Generalist Agent,Claude-3.7 Sonnet High (February 2025),Yes,37.78%,$66.15,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentsonnet_37_high_1755663010_UPLOAD.zip?download=true
8,CORE-Agent,Claude Sonnet 4.5 (September 2025),Yes,37.78%,$97.15,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet45_1759329435_UPLOAD.zip?download=true
9,HAL Generalist Agent,o4-mini High (April 2025),Yes,35.56%,$45.37,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agento4minihigh_1755580383_UPLOAD.zip?download=true
10,CORE-Agent,Claude-3.7 Sonnet (February 2025),Yes,35.56%,$73.04,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744922181_UPLOAD.zip?download=true
11,HAL Generalist Agent,Gemini 3 Pro Preview High (November 2025),Yes,35.56%,$101.27,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1763837773_UPLOAD.zip?download=true
12,HAL Generalist Agent,Claude Opus 4.1 (August 2025),Yes,35.56%,$375.11,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754443772_UPLOAD.zip?download=true
13,HAL Generalist Agent,Claude Sonnet 4.5 (September 2025),Yes,33.33%,$85.19,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_halgeneralistagentclaudesonnet45_1759433359_UPLOAD.zip?download=true
14,CORE-Agent,Claude Sonnet 4 High (May 2025),Yes,33.33%,$100.48,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet4high_1755814601_UPLOAD.zip?download=true
15,CORE-Agent,GPT-4.1 (April 2025),Yes,33.33%,$107.36,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744752123_UPLOAD.zip?download=true
16,HAL Generalist Agent,Claude Opus 4.5 (November 2025),Yes,33.33%,$127.41,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_opus_45_1764046559_UPLOAD.zip?download=true
17,HAL Generalist Agent,Claude Opus 4.1 High (August 2025),Yes,33.33%,$358.47,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754569694_UPLOAD.zip?download=true
18,HAL Generalist Agent,Claude-3.7 Sonnet (February 2025),Yes,31.11%,$56.64,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentsonnet_37_1755652380_UPLOAD.zip?download=true
19,HAL Generalist Agent,Claude Opus 4.5 High (November 2025),Yes,31.11%,$112.38,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_opus_45_high_1764046628_UPLOAD.zip?download=true
20,CORE-Agent,Claude Sonnet 4 (May 2025),Yes,28.89%,$50.27,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudesonnet4_1755796611_UPLOAD.zip?download=true
21,HAL Generalist Agent,Claude Sonnet 4.5 High (September 2025),Yes,28.89%,$87.77,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_halgeneralistagentclaudesonnet45high_1759423572_UPLOAD.zip?download=true
22,CORE-Agent,GPT-5 Medium (August 2025),Yes,26.67%,$31.76,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754599494_UPLOAD.zip?download=true
23,CORE-Agent,o4-mini High (April 2025),Yes,26.67%,$61.35,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745075792_UPLOAD.zip?download=true
24,CORE-Agent,Claude-3.7 Sonnet High (February 2025),Yes,24.44%,$72.47,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745258007_UPLOAD.zip?download=true
25,CORE-Agent,o3 Medium (April 2025),Yes,24.44%,$120.47,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745118876_UPLOAD.zip?download=true
26,HAL Generalist Agent,GPT-4.1 (April 2025),Yes,22.22%,$58.32,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentgpt41_1755644685_UPLOAD.zip?download=true
27,HAL Generalist Agent,o3 Medium (April 2025),Yes,22.22%,$88.34,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agento3medium_1755626315_UPLOAD.zip?download=true
28,CORE-Agent,Gemini 2.5 Pro Preview (March 2025),Yes,22.22%,$182.34,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744922265_UPLOAD.zip?download=true
29,CORE-Agent,DeepSeek V3.1 (August 2025),Yes,20.00%,$12.55,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentdeepseekv31_1755793007_UPLOAD.zip?download=true
30,CORE-Agent,DeepSeek V3 (March 2025),Yes,17.78%,$25.26,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744854746_UPLOAD.zip?download=true
31,CORE-Agent,o4-mini Low (April 2025),Yes,17.78%,$31.79,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1745046580_UPLOAD.zip?download=true
32,HAL Generalist Agent,o4-mini Low (April 2025),Yes,15.56%,$22.50,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agento4minilow_1755608756_UPLOAD.zip?download=true
33,CORE-Agent,GPT-OSS-120B (August 2025),Yes,11.11%,$4.21,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754492673_UPLOAD.zip?download=true
34,CORE-Agent,GPT-OSS-120B High (August 2025),Yes,11.11%,$4.21,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1754539776_UPLOAD.zip?download=true
35,CORE-Agent,Gemini 2.0 Flash (February 2025),Yes,11.11%,$12.46,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744856042_UPLOAD.zip?download=true
36,HAL Generalist Agent,GPT-5 Medium (August 2025),Yes,11.11%,$29.75,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentgpt5medium_1756137340_UPLOAD.zip?download=true
37,CORE-Agent,Claude Haiku 4.5 (October 2025),Yes,11.11%,$43.93,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagentclaudehaiku45_1760647306_UPLOAD.zip?download=true
38,HAL Generalist Agent,GPT-OSS-120B High (August 2025),Yes,8.89%,$2.05,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754569684_UPLOAD.zip?download=true
39,HAL Generalist Agent,GPT-OSS-120B (August 2025),Yes,8.89%,$2.79,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1754439280_UPLOAD.zip?download=true
40,HAL Generalist Agent,DeepSeek V3 (March 2025),Yes,8.89%,$4.69,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentdeepseekv30324_1755710007_UPLOAD.zip?download=true
41,HAL Generalist Agent,DeepSeek R1 (May 2025),Yes,8.89%,$7.77,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentdeepseekr10528_1755721934_UPLOAD.zip?download=true
42,CORE-Agent,DeepSeek R1 (January 2025),Yes,6.67% (-2.22/+2.22),$81.11 (-46.45/+46.45),2,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_coreagent_1744922373_UPLOAD.zip?download=true
43,HAL Generalist Agent,DeepSeek R1 (January 2025),Yes,4.45% (-2.22/+2.22),$24.95 (-11.07/+22.15),2,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1747247500_UPLOAD.zip?download=true
44,HAL Generalist Agent,Gemini 2.0 Flash (February 2025),Yes,4.44%,$7.06,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agentgemini20flash_1755839828_UPLOAD.zip?download=true
45,HAL Generalist Agent,Gemini 2.5 Pro Preview (March 2025),Yes,4.44%,$30.38,1,https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/corebench_hard_hal_generalist_agent_1747247518_UPLOAD.zip?download=true"""

available_models_list = [
    "gemini-3-flash-preview", "deepseek-reasoner", "gpt-5.2-2025-12-11_xhigh",
    "Qwen3-235B-A22B-Thinking-2507", "openai/gpt-oss-120b_high", "gpt-5.2-2025-12-11_high",
    "gpt-5.2-2025-12-11_medium", "gpt-5.2-2025-12-11_low", "claude-opus-4-5-20251101_16K",
    "claude-opus-4-5-20251101_32K", "claude-opus-4-5-20251101", "gemini-3-pro-preview",
    "gpt-5.1-2025-11-13_medium", "gemini-2.5-pro", "gpt-5.1-2025-11-13_high",
    "kimi-k2-thinking-turbo", "gpt-5-nano-2025-08-07_high", "gpt-5-mini-2025-08-07_high",
    "gpt-5-2025-08-07_high", "claude-sonnet-4-5-20250929_16K", "claude-sonnet-4-5-20250929_59K",
    "gpt-4-0314", "claude-haiku-4-5-20251001_32K", "claude-sonnet-4-5-20250929_32K",
    "claude-haiku-4-5-20251001", "qwen3-max-2025-09-23", "claude-sonnet-4-5-20250929",
    "gpt-5-mini-2025-08-07_medium", "gpt-5-2025-08-07_medium", "gpt-5-nano-2025-08-07_medium",
    "claude-opus-4-1-20250805_27K", "claude-opus-4-1-20250805_16K", "claude-opus-4-1-20250805",
    "magistral-small-2506", "gemini-2.5-pro-preview-06-05", "qwen3-235b-a22b",
    "gemini-2.5-pro-preview-05-06", "DeepSeek-R1-0528", "claude-3-7-sonnet-20250219_64K",
    "grok-3-mini-beta_high", "DeepSeek-R1", "claude-sonnet-4-20250514_59K", "grok-3-beta",
    "claude-sonnet-4-20250514_32K", "claude-opus-4-20250514_16K", "claude-sonnet-4-20250514_16K",
    "claude-sonnet-4-20250514", "claude-opus-4-20250514", "mistral-medium-2505",
    "o4-mini-2025-04-16_high", "o3-2025-04-16_high", "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14", "gpt-4.1-2025-04-14", "qwq-plus", "grok-3-mini-beta_low",
    "Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "qwen-turbo-2024-11-01", "qwen-plus-2025-01-25", "qwen-max-2025-01-25",
    "DeepSeek-V3-0324", "gemini-2.5-pro-exp-03-25", "mistral-small-2503", "gemma-3-27b-it",
    "claude-3-5-haiku-20241022", "gemini-1.5-flash-8b-001", "DeepSeek-R1-Distill-Qwen-14B",
    "claude-3-7-sonnet-20250219_32K", "DeepSeek-R1-Distill-Llama-70B", "gpt-4.5-preview-2025-02-27",
    "gpt-4-turbo-2024-04-09", "claude-3-7-sonnet-20250219_16K", "mistral-large-2411",
    "claude-3-7-sonnet-20250219", "o1-2024-12-17_high", "o1-mini-2024-09-12_high",
    "o3-mini-2025-01-31_high", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-001", "gpt-4o-2024-11-20", "phi-4", "Phi-3-medium-128k-instruct",
    "o3-mini-2025-01-31_medium", "mistral-small-2501", "qwen2.5-32b-instruct",
    "gemini-1.5-flash-001", "claude-2.0", "claude-3-opus-20240229",
    "o1-mini-2024-09-12_medium", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20241022",
    "gpt-4-0125-preview", "claude-2.1", "gemma-2-27b-it", "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620", "gpt-4o-2024-05-13", "gemini-1.5-flash-002",
    "gpt-4-0613", "gemma-2-9b-it", "gpt-3.5-turbo-1106", "gemini-1.5-pro-001",
    "gpt-4-1106-preview", "gpt-4o-mini-2024-07-18", "gemini-1.5-pro-002", "gpt-3.5-turbo-0125",
    "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Llama-3.1-405B-Instruct",
    "Yi-1.5-34B-Chat", "Yi-34B-Chat", "grok-2-1212", "qwen2-72b-instruct",
    "qwen1.5-72b-chat", "qwen1.5-32b-chat", "Hermes-2-Theta-Llama-3-70B",
    "Llama-2-70b-chat-hf", "Meta-Llama-3-70B-Instruct", "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3", "deepseek-llm-67b-chat", "Mixtral-8x7B-Instruct-v0.1",
    "qwen2.5-72b-instruct", "WizardLM-2-8x22B", "dbrx-instruct", "gemini-1.0-pro-001",
    "ministral-3b-2410", "mistral-large-2402", "open-mixtral-8x22b", "ministral-8b-2410",
    "mistral-large-2407", "open-mixtral-8x7b", "open-mistral-7b", "open-mistral-nemo-2407",
    "Llama-3.2-90B-Vision-Instruct", "Llama-3.1-Tulu-3-70B-DPO", "Eurus-2-7B-PRIME",
    "Llama-3.3-70B-Instruct", "o1-2024-12-17_medium", "DeepSeek-V3", "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12"
]

core_bench_df = pd.read_csv(io.StringIO(core_bench_csv))
unique_core_models = core_bench_df['Primary Model'].unique()

mapping = {}

# Heuristic matching
for core_model in unique_core_models:
    matched = ""
    # Lowercase for comparison
    core_lower = core_model.lower()
    
    # Try to find a direct or close match
    for avail in available_models_list:
        avail_lower = avail.lower()
        
        # Specific handling for High/Medium/Low in names
        is_high = "high" in core_lower
        is_medium = "medium" in core_lower
        is_low = "low" in core_lower
        
        # Check if the core model name bits are in the available model name
        # We look for keywords like 'claude', 'opus', '4.1', '2025', 'august' (08)
        
        # Example: Claude Opus 4.1 (August 2025)
        # Check bits: 'claude', 'opus', '4.1', '2025'
        
        # Claude matches
        if "claude" in core_lower and "claude" in avail_lower:
            if "opus 4.1" in core_lower and "opus-4-1" in avail_lower:
                if "20250805" in avail_lower: # August 2025
                    matched = avail
                    break
            if "opus 4.5" in core_lower and "opus-4-5" in avail_lower:
                if "20251101" in avail_lower: # November 2025
                    matched = avail
                    break
            if "sonnet 4.5" in core_lower and "sonnet-4-5" in avail_lower:
                if "20250929" in avail_lower: # September 2025
                    matched = avail
                    break
            if "sonnet 4" in core_lower and "sonnet-4" in avail_lower and "4.5" not in core_lower:
                if "20250514" in avail_lower: # May 2025
                    matched = avail
                    break
            if "haiku 4.5" in core_lower and "haiku-4-5" in avail_lower:
                if "20251001" in avail_lower: # October 2025
                    matched = avail
                    break
            if "claude-3.7 sonnet" in core_lower and "claude-3-7-sonnet" in avail_lower:
                if "20250219" in avail_lower:
                    matched = avail
                    break

        # GPT/O matches
        if "gpt-5" in core_lower and "gpt-5" in avail_lower:
            if "medium" in core_lower and "medium" in avail_lower and "2025-08-07" in avail_lower:
                matched = avail
                break
        
        if "gpt-4.1" in core_lower and "gpt-4.1" in avail_lower:
            if "2025-04-14" in avail_lower:
                matched = avail
                break
        
        if "o4-mini" in core_lower and "o4-mini" in avail_lower:
            if is_high and "_high" in avail_lower:
                matched = avail
                break
            if is_low and "_low" in avail_lower:
                matched = avail
                break
        
        if "o3" in core_lower and "o3" in avail_lower:
            # CORE bench says "o3 Medium (April 2025)". Available has "o3-2025-04-16_high".
            # Or check o3-mini-2025-01-31_medium
            if "medium" in core_lower:
                if "o3-mini" in avail_lower and "medium" in avail_lower:
                    matched = avail
                    break
                elif "o3" in avail_lower and "high" in avail_lower: # Fallback?
                    matched = avail
                    # Don't break yet, look for medium
        
        # DeepSeek
        if "deepseek v3.1" in core_lower and "v3.1" in avail_lower:
             # Check if available has v3.1? List doesn't have v3.1 explicitly, but has V3.
             pass
        if "deepseek v3" in core_lower and "deepseek-v3" in avail_lower:
             matched = avail
             break
        if "deepseek r1" in core_lower and "deepseek-r1" in avail_lower:
             if "0528" in avail_lower and "may" in core_lower:
                 matched = avail
                 break
             if "(" not in core_model: # No date, default
                 matched = "DeepSeek-R1"
        
        # Gemini
        if "gemini 3 pro preview" in core_lower and "gemini-3-pro-preview" in avail_lower:
            matched = avail
            break
        if "gemini 2.5 pro" in core_lower and "gemini-2.5-pro" in avail_lower:
            if "march 2025" in core_lower and "03-25" in avail_lower:
                matched = avail
                break
            elif "march 2025" in core_lower and "05-06" in avail_lower: # Check other previews
                pass
        if "gemini 2.0 flash" in core_lower and "gemini-2.0-flash" in avail_lower:
            matched = avail
            break
            
        if "gpt-oss-120b" in core_lower and "gpt-oss-120b" in avail_lower:
            matched = avail
            break
            
    mapping[core_model] = matched

# Manual refinements for missing ones
mapping["DeepSeek R1 (January 2025)"] = "DeepSeek-R1"
mapping["o3 Medium (April 2025)"] = "o3-mini-2025-01-31_medium"
mapping["Gemini 2.5 Pro Preview (March 2025)"] = "gemini-2.5-pro-exp-03-25"

# Results
df_mapping = pd.DataFrame(list(mapping.items()), columns=["CORE Bench Model", "Available Model"])
df_mapping.to_csv("model_mapping.csv", index=False)
print(df_mapping)