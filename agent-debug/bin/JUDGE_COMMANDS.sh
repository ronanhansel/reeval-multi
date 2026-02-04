#!/bin/bash
# JUDGE AGGREGATION - Run after RUBRIC_COMMANDS.sh completes
#
# Usage:
#   ./JUDGE_COMMANDS.sh              # Run all benchmarks
#   ./JUDGE_COMMANDS.sh scicode      # Run only SciCode
#   ./JUDGE_COMMANDS.sh corebench    # Run only CoreBench
#   ./JUDGE_COMMANDS.sh sab          # Run only ScienceAgentBench
#   ./JUDGE_COMMANDS.sh colbench     # Run only ColBench

BENCHMARK="${1:-all}"

case "$BENCHMARK" in
    all|scicode|corebench|sab|colbench)
        echo "Running judge for: $BENCHMARK"
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Usage: $0 [all|scicode|corebench|sab|colbench]"
        exit 1
        ;;
esac

echo "============================================================"
echo "JUDGE AGGREGATION"
echo "============================================================"
echo ""

# SCICODE
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "scicode" ]; then
echo "=== SciCode ==="
python scripts/judge.py \
    --pattern "scicode_sea_*" \
    --pattern "scicode_honey_*" \
    --rubric-dir rubrics_output/scicode \
    --output judge_output/scicode_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 1000 \
    -y

echo "✓ SciCode judge completed"
fi

# COREBENCH
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "corebench" ]; then
echo "=== CoreBench ==="
python scripts/judge.py \
    --pattern "prop_*" \
    --pattern "iter1_*" \
    --rubric-dir rubrics_output/corebench \
    --output judge_output/corebench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 1000 \
    -y

echo "✓ CoreBench judge completed"
fi

# SAB
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "sab" ]; then
echo "=== SAB ==="
python scripts/judge.py \
    --pattern "sab_mate_*" \
    --pattern "sab_cow_*" \
    --pattern "sab_husky_*" \
    --priority-override "sab_husky_*" "sab_cow_*" \
    --rubric-dir rubrics_output/scienceagentbench \
    --output judge_output/scienceagentbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 1000 \
    -y

echo "✓ SAB judge completed"
fi

# COLBENCH
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "colbench" ]; then
echo "=== ColBench ==="
python scripts/judge.py \
    --pattern "col_ivy_*" \
    --pattern "col_zuck_*" \
    --rubric-dir rubrics_output/colbench \
    --output judge_output/colbench_verdict.csv \
    --model openai:gpt-5.2 \
    --parallel 1000 \
    -y

echo "✓ ColBench judge completed"
fi

echo ""
echo "============================================================"
echo "ALL VERDICTS CREATED!"
echo "============================================================"
echo ""
ls -lh judge_output/
