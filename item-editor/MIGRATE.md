# Migration Guide - Project Reorganization

The project structure has been reorganized into a more streamlined hierarchy under the `item-editor/` root.

## New Structure

```
item-editor/
├── docent/                     # Docent library
├── hal-harness/                # HAL evaluation harness
├── eval_traces/                # Evaluation traces (input/aggregated)
├── eval_response_matrix/       # Response matrices (input/aggregated)
├── patch/                      # Patch diff files
├── result/                     # Results and outputs
│   ├── fixes/                  # Generated fixes for IFEs
│   └── .hal_data/              # Internal data (logs, traces, results, cache)
├── config/                     # Configuration files
│   ├── model/                  # Model-specific benchmark configs
│   └── rubric/                 # Rubric templates (.txt files only)
└── script/                     # Python scripts and utilities
    ├── eval/                   # Evaluation and judging scripts
    ├── trace/                  # Trace merging and collection
    ├── fix/                    # IFE diagnosis and fixing
    └── utils/                  # Common utilities and maintenance scripts
```

## Key Changes

1.  **Renamed Config**: `config/model_config` has been renamed to `config/model`.
2.  **Organized Scripts**: Scripts are grouped by function. Major operational scripts are in `eval/`, `trace/`, and `fix/`, while utilities and support scripts are centralized in `utils/`.
3.  **Centralized Data**: All outputs (logs, traces, results, LLM cache) are written to `result/.hal_data/`.
4.  **Streamlined Rubrics**: Rubric templates are located in `config/rubric/` and contain only `.txt` files.
5.  **Path Resolution**: All scripts correctly resolve the project root and include necessary search paths.

## How to Run

Most commands should be run from the project root.

### Rubric Evaluation
```bash
python script/eval/eval_rubric.py \
    --trace-file result/.hal_data/traces/some_trace_UPLOAD.json \
    --rubric config/rubric/colbench.txt \
    --rubric-model azure_openai:gpt-5.2 \
    --failed-only -y
```

### Build Response Matrix
```bash
python script/utils/build_response_matrix.py --prefix "moon18_" --benchmark colbench
```

### Judge IFEs
```bash
python script/eval/judge.py \
    --pattern "*.csv" \
    --rubric-dir result/.hal_data/rubrics_output/scicode \
    --model azure_openai:gpt-5.2 \
    -y
```

### Fix IFEs with Claude
```bash
python script/fix/claude_fixer.py --benchmark scicode --ife-only
```

### Maintenance & Utils
```bash
# Find failed tasks
python script/utils/find_failed_tasks.py --benchmark scicode

# Cleanup local processes and containers
python script/utils/cleanup.py --aggressive

# Prebuild all required docker images
python script/utils/prebuild_all_images.py scicode colbench
```

## Internal Path Updates

The following environment variables and defaults were updated:
- `HAL_RESULTS_DIR` defaults to `result/.hal_data/results`
- `HAL_TRACES_DIR` defaults to `result/.hal_data/traces`
- `HAL_TMP_DIR` defaults to `result/.hal_data/tmp`
- `HAL_LOGS_DIR` defaults to `result/.hal_data/logs`
- `LLM_CACHE_PATH` defaults to `result/.hal_data/.llm_cache`
