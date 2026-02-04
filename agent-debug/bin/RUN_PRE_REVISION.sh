#!/bin/bash
# PRE-REVISION EVALUATION PIPELINE
# Aggregates results for the 7 benchmarks in eval_traces/traces_hal_harness

TRACES_DIR="eval_traces/traces_hal_harness"
OUTPUT_BASE="eval_response_matrix/pre-revision"
RUBRIC_MODEL="openai:gpt-5.2"

echo "============================================================"
echo "STEP 1: RUBRIC EVALUATION"
echo "============================================================"

# Helper function to run rubric eval for a benchmark
run_rubric() {
    local BENCH=$1
    local RUBRIC=$2
    local PATTERN=$3
    echo "--- Evaluating $BENCH ---"
    python3 scripts/eval_rubric.py \
        --traces-dir "$TRACES_DIR" \
        --prefix "$PATTERN" \
        --benchmark "$BENCH" \
        --rubric "rubric_templates/$RUBRIC" \
        --rubric-model "$RUBRIC_MODEL" \
        -y
}

run_rubric "assistantbench" "assistantbench.txt" "^assistantbench_"
run_rubric "colbench_backend_programming" "colbench.txt" "^colbench_backend_programming_"
run_rubric "corebench_hard" "corebench.txt" "^corebench_hard_"
run_rubric "scicode" "scicode.txt" "^scicode_"
run_rubric "scienceagentbench" "scienceagentbench.txt" "^scienceagentbench_"
run_rubric "swebench_verified_mini" "swebench.txt" "^swebench_verified_mini_"
run_rubric "usaco" "usaco.txt" "^usaco_"

echo "============================================================"
echo "STEP 2: JUDGE VERDICTS"
echo "============================================================"

# Helper function to run judge for a benchmark
run_judge() {
    local BENCH=$1
    local RUBRIC_SUBDIR=$2
    local PATTERN=$3
    echo "--- Judging $BENCH ---"
    python3 scripts/judge.py \
        --rubric-dir "eval_traces/rubrics_output/$RUBRIC_SUBDIR" \
        --prefix "$PATTERN" \
        --model "$RUBRIC_MODEL" \
        -y
}

run_judge "assistantbench" "assistantbench" "^assistantbench_"
run_judge "colbench_backend_programming" "colbench" "^colbench_backend_programming_"
run_judge "corebench_hard" "corebench" "^corebench_hard_"
run_judge "scicode" "scicode" "^scicode_"
run_judge "scienceagentbench" "scienceagentbench" "^scienceagentbench_"
run_judge "swebench_verified_mini" "swebench" "^swebench"
run_judge "usaco" "usaco" "^usaco_"

echo "============================================================"
echo "STEP 3: RESPONSE MATRIX GENERATION"
echo "============================================================"

# Run build_response_matrix.py for each prefix group
# Note: we use the benchmark name as the prefix since that's how files are named
for BENCH in assistantbench colbench_backend_programming corebench_hard scicode scienceagentbench swebench_verified_mini usaco; do
    echo "--- Generating Matrix for $BENCH ---"
    python3 scripts/build_response_matrix.py \
        --prefix "^${BENCH}_" \
        --traces-dir "$TRACES_DIR" \
        --benchmark "$BENCH" \
        --extract-subscores \
        --output "$OUTPUT_BASE"
done

echo "============================================================"
echo "PRE-REVISION EVALUATION COMPLETE!"
echo "Outputs in: $OUTPUT_BASE"
echo "============================================================"
