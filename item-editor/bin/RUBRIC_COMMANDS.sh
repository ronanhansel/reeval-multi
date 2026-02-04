#!/bin/bash
# RUBRIC EVALUATION - Run after FINAL_COMMANDS.sh completes
#
# Usage:
#   ./RUBRIC_COMMANDS.sh              # Run all benchmarks
#   ./RUBRIC_COMMANDS.sh scicode      # Run only SciCode
#   ./RUBRIC_COMMANDS.sh corebench    # Run only CoreBench
#   ./RUBRIC_COMMANDS.sh sab          # Run only ScienceAgentBench
#   ./RUBRIC_COMMANDS.sh colbench     # Run only ColBench

BENCHMARK="${1:-all}"

case "$BENCHMARK" in
    all|scicode|corebench|sab|colbench)
        echo "Running rubric for: $BENCHMARK"
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Usage: $0 [all|scicode|corebench|sab|colbench]"
        exit 1
        ;;
esac

echo "============================================================"
echo "RUBRIC EVALUATION"
echo "============================================================"
echo ""

# SCICODE (14 traces total: 4 from honey + 10 from lady)
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "scicode" ]; then
echo "=== SciCode ==="
python scripts/eval_rubric.py \
    --trace-file traces/scicode_honey_openai_gpt-4_1.json \
    --trace-file traces/scicode_honey_openai_o3_2025.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_honey_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/scicode_sea_openai_gpt-4_1.json \
    --trace-file traces/scicode_sea_openai_o3_2025.json \
    --trace-file traces/scicode_sea_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/scicode_sea_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/scicode_sea_openai_gpt-5_2025.json \
    --trace-file traces/scicode_sea_openai_gpt-5-mini_2025.json \
    --trace-file traces/scicode_sea_openai_gpt-4o_2024.json \
    --trace-file traces/scicode_sea_openai_o3-mini_2025.json \
    --trace-file traces/scicode_sea_DeepSeek-R1.json \
    --trace-file traces/scicode_sea_deepseek-ai_DeepSeek-V3.json \
    --rubric rubric_templates/scicode.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000

echo "✓ SciCode rubric completed"
fi

# COREBENCH (14 traces total: 10 from prop + 4 from iter1)
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "corebench" ]; then
echo "=== CoreBench ==="
python scripts/eval_rubric.py \
    --trace-file traces/prop_openai_gpt-4_1.json \
    --trace-file traces/prop_openai_o3_2025-04-16_medium.json \
    --trace-file traces/prop_openai_o3_2025-04-16_low.json \
    --trace-file traces/prop_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/prop_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/prop_openai_gpt-5_2025.json \
    --trace-file traces/prop_openai_gpt-4o_2024.json \
    --trace-file traces/prop_openai_gpt-oss-120b.json \
    --trace-file traces/prop_DeepSeek-R1.json \
    --trace-file traces/prop_deepseek-ai_DeepSeek-V3.json \
    --trace-file traces/iter1_openai_gpt-4_1.json \
    --trace-file traces/iter1_openai_o3_2025.json \
    --trace-file traces/iter1_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/iter1_openai_o4-mini_2025-04-16_low.json \
    --rubric rubric_templates/corebench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000

echo "✓ CoreBench rubric completed"
fi

# SAB (11 traces total: 6 from mate + 4 from cow + 1 from husky)
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "sab" ]; then
echo "=== SAB ==="
python scripts/eval_rubric.py \
    --trace-file traces/sab_mate_openai_gpt-5_2025.json \
    --trace-file traces/sab_mate_openai_gpt-5-mini_2025.json \
    --trace-file traces/sab_mate_openai_gpt-4o_2024.json \
    --trace-file traces/sab_mate_openai_o3-mini_2025.json \
    --trace-file traces/sab_mate_DeepSeek-R1.json \
    --trace-file traces/sab_mate_deepseek-ai_DeepSeek-V3.json \
    --trace-file traces/sab_cow_openai_gpt-4_1.json \
    --trace-file traces/sab_cow_openai_o3_2025.json \
    --trace-file traces/sab_cow_openai_o4-mini_2025-04-16_high.json \
    --trace-file traces/sab_cow_openai_o4-mini_2025-04-16_low.json \
    --trace-file traces/sab_husky_openai_gpt-4_1.json \
    --rubric rubric_templates/scienceagentbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000

echo "✓ SAB rubric completed"
fi

# COLBENCH - Using _WITH_DIALOGUES.json files (created by CREATE_COLBENCH_DIALOGUES.sh)
if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "colbench" ]; then
echo "=== ColBench ==="
echo "NOTE: Using _WITH_DIALOGUES.json files (run CREATE_COLBENCH_DIALOGUES.sh first if missing)"
python scripts/eval_rubric.py \
    --trace-file traces/col_ivy_openai_gpt-4_1_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_o3_medium_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_o4-mini_high_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_o4-mini_low_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_gpt-5_medium_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_gpt-5-mini_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_gpt-4o_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_openai_o3-mini_high_WITH_DIALOGUES.json \
    --trace-file traces/col_ivy_DeepSeek-R1_WITH_DIALOGUES.json \
    --trace-file traces/col_zuck_openai_gpt-4_1_WITH_DIALOGUES.json \
    --trace-file traces/col_zuck_openai_o3_low_WITH_DIALOGUES.json \
    --trace-file traces/col_zuck_openai_o4-mini_high_WITH_DIALOGUES.json \
    --rubric rubric_templates/colbench.txt \
    --rubric-model openai:gpt-5.2 \
    --failed-only -y \
    --max-batch-messages 1000

echo "✓ ColBench rubric completed"
fi

echo ""
echo "============================================================"
echo "RUBRIC EVALUATION COMPLETE!"
echo "============================================================"
echo "Next step: ./bin/JUDGE_COMMANDS.sh $BENCHMARK"
