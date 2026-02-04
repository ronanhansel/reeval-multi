#!/bin/bash
set -e

# Ensure we are in item-editor
if [[ ! -d "scripts" ]]; then
    if [[ -d "item-editor/scripts" ]]; then
        cd item-editor
    else
        echo "Error: Must run from item-editor directory or root containing it"
        exit 1
    fi
fi

TRACES_DIR="eval_traces/traces_hal_harness"
OUTPUT_BASE="eval_response_matrix/pre-revision"
RUBRIC_MODEL="openai:gpt-5.2"

run_fast_pipeline() {
    local BENCH=$1
    local RUBRIC_FILE=$2
    local RUBRIC_NAME=$3
    local PREFIX=$4

    echo "============================================================"
    echo "FAST PROCESSING: $BENCH"
    echo "============================================================"

    # SKIP Step 1: Eval Rubric (Assumed mostly done)
    echo "--- Skipping Step 1: Eval Rubric ---"

    # 2. Judge
    echo "--- Step 2: Judge ---"
    python3 scripts/judge.py \
        --rubric-dir "eval_traces/rubrics_output/$RUBRIC_NAME" \
        --prefix "$PREFIX" \
        --model "$RUBRIC_MODEL" \
        --traces-dir "$TRACES_DIR" \
        --original \
        -y

    # 3. Build Response Matrix
    echo "--- Step 3: Build Matrix ---"
    python3 scripts/build_response_matrix.py \
        --prefix "$PREFIX" \
        --traces-dir "$TRACES_DIR" \
        --benchmark "$BENCH" \
        --extract-subscores \
        --original \
        --output "$OUTPUT_BASE"
}

run_fast_pipeline "assistantbench" "assistantbench.txt" "assistantbench" "^assistantbench_"
run_fast_pipeline "colbench_backend_programming" "colbench.txt" "colbench" "^colbench_backend_programming_"
run_fast_pipeline "corebench_hard" "corebench.txt" "corebench" "^corebench_hard_"
run_fast_pipeline "scicode" "scicode.txt" "scicode" "^scicode_"
run_fast_pipeline "scienceagentbench" "scienceagentbench.txt" "scienceagentbench" "^scienceagentbench_"
run_fast_pipeline "swebench_verified_mini" "swebench.txt" "swebench" "^swebench_verified_mini_"
run_fast_pipeline "usaco" "usaco.txt" "usaco" "^usaco_"

echo "Done!"