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

run_pipeline() {
    local BENCH=$1
    local RUBRIC_FILE=$2
    local RUBRIC_NAME=$3  # The subdir in rubrics_output
    local PREFIX=$4

    echo "============================================================"
    echo "PROCESSING: $BENCH"
    echo "============================================================"

    # 1. Eval Rubric
    echo "--- Step 1: Eval Rubric ---"
    python3 scripts/eval_rubric.py \
        --traces-dir "$TRACES_DIR" \
        --prefix "$PREFIX" \
        --benchmark "$BENCH" \
        --rubric "rubric_templates/$RUBRIC_FILE" \
        --rubric-model "$RUBRIC_MODEL" \
        --original \
        -y

    # 2. Judge
    echo "--- Step 2: Judge ---"
    python3 scripts/judge.py \
        --rubric-dir "eval_traces/rubrics_output/$RUBRIC_NAME" \
        --prefix "$PREFIX" \
        --model "$RUBRIC_MODEL" \
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

# Run for all
run_pipeline "assistantbench" "assistantbench.txt" "assistantbench" "^assistantbench_"
run_pipeline "colbench_backend_programming" "colbench.txt" "colbench" "^colbench_backend_programming_"
run_pipeline "corebench_hard" "corebench.txt" "corebench" "^corebench_hard_"
run_pipeline "scicode" "scicode.txt" "scicode" "^scicode_"
run_pipeline "scienceagentbench" "scienceagentbench.txt" "scienceagentbench" "^scienceagentbench_"
run_pipeline "swebench_verified_mini" "swebench.txt" "swebench" "^swebench_verified_mini_"
run_pipeline "usaco" "usaco.txt" "usaco" "^usaco_"

echo "Done!"