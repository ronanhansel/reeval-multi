#!/bin/bash
# FINAL MERGE + WEAVE + RUBRIC COMMANDS
# Based on actual trace files that exist
#
# Usage:
#   ./FINAL_COMMANDS.sh              # Run all benchmarks
#   ./FINAL_COMMANDS.sh scicode      # Run only SciCode
#   ./FINAL_COMMANDS.sh corebench    # Run only CoreBench
#   ./FINAL_COMMANDS.sh sab          # Run only ScienceAgentBench
#   ./FINAL_COMMANDS.sh colbench     # Run only ColBench

BENCHMARK="${1:-all}"

case "$BENCHMARK" in
    all|scicode|corebench|sab|colbench)
        echo "Running: $BENCHMARK"
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Usage: $0 [all|scicode|corebench|sab|colbench]"
        exit 1
        ;;
esac

# ============================================================
# SCICODE
# ============================================================

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "scicode" ]; then
echo "=== SCICODE ==="

# SCICODE_HONEY (4 models)
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_honey__scicode_honey_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_honey_scicode \
    --prefix scicode_honey_openai_gpt-4_1 \
    --prefix scicode_honey_openai_o3_2025 \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_high \
    --prefix scicode_honey_openai_o4-mini_2025-04-16_low \
    --merge-input traces/scicode_honey_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_honey_openai_o4-mini_low_MERGED_UPLOAD.json

# scicode_sea (10 models)
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/scicode_sea_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/scicode_sea_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/scicode_sea_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/scicode_sea_openai_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_gpt-5_2025-08-07_*_UPLOAD.json' --output traces/scicode_sea_openai_gpt-5_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/scicode_sea_openai_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/scicode_sea_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/scicode_sea_openai_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_DeepSeek-R1_1_*_UPLOAD.json' --output traces/scicode_sea_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/scicode_sea__scicode_sea_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/scicode_sea_deepseek-v3_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/scicode_sea_scicode \
    --prefix scicode_sea_openai_gpt-4_1 \
    --prefix scicode_sea_openai_o3_2025 \
    --prefix scicode_sea_openai_o4-mini_2025-04-16_high \
    --prefix scicode_sea_openai_o4-mini_2025-04-16_low \
    --prefix scicode_sea_openai_gpt-5_2025 \
    --prefix scicode_sea_openai_gpt-5-mini_2025 \
    --prefix scicode_sea_openai_gpt-4o_2024 \
    --prefix scicode_sea_openai_o3-mini_2025 \
    --prefix scicode_sea_DeepSeek-R1 \
    --prefix scicode_sea_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/scicode_sea_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_gpt-5_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_openai_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/scicode_sea_deepseek-v3_MERGED_UPLOAD.json

echo "✓ SciCode completed"
fi

# ============================================================
# COREBENCH
# ============================================================

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "corebench" ]; then
echo "=== COREBENCH ==="

# PROP (10 models)
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/prop_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o3_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/prop_openai_o3_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-5_2025-08-07_medium_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-4o_2024-11-20_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_openai_gpt-oss-120b_1_capsule-*_UPLOAD.json' --output traces/prop_openai_gpt-oss-120b_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_DeepSeek-R1_1_capsule-*_UPLOAD.json' --output traces/prop_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/prop_deepseek-ai_DeepSeek-V3-0324_capsule-*_UPLOAD.json' --output traces/prop_deepseek-v3_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/prop_corebench_hard \
    --prefix prop_openai_gpt-4_1 \
    --prefix prop_openai_o3_2025-04-16_medium \
    --prefix prop_openai_o3_2025-04-16_low \
    --prefix prop_openai_o4-mini_2025-04-16_high \
    --prefix prop_openai_o4-mini_2025-04-16_low \
    --prefix prop_openai_gpt-5_2025 \
    --prefix prop_openai_gpt-4o_2024 \
    --prefix prop_openai_gpt-oss-120b \
    --prefix prop_DeepSeek-R1 \
    --prefix prop_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/prop_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o3_low_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/prop_openai_gpt-oss-120b_MERGED_UPLOAD.json \
    --merge-input traces/prop_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/prop_deepseek-v3_MERGED_UPLOAD.json

# ITER1 (4 models)
python scripts/merge_traces.py --input 'traces/iter1_openai_gpt-4_1_2025-04-14_capsule-*_UPLOAD.json' --output traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o3_2025-04-16_medium_capsule-*_UPLOAD.json' --output traces/iter1_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_high_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/iter1_openai_o4-mini_2025-04-16_low_capsule-*_UPLOAD.json' --output traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/iter1_corebench_hard \
    --prefix iter1_openai_gpt-4_1 \
    --prefix iter1_openai_o3_2025 \
    --prefix iter1_openai_o4-mini_2025-04-16_high \
    --prefix iter1_openai_o4-mini_2025-04-16_low \
    --merge-input traces/iter1_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/iter1_openai_o4-mini_low_MERGED_UPLOAD.json

echo "✓ CoreBench completed"
fi

# ============================================================
# SAB
# ============================================================

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "sab" ]; then
echo "=== SAB ==="

# SAB_MATE (6 models)
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/sab_mate_openai_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_openai_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_DeepSeek-R1_1_*_UPLOAD.json' --output traces/sab_mate_DeepSeek-R1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_mate__sab_mate_deepseek-ai_DeepSeek-V3-0324_*_UPLOAD.json' --output traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_mate_scienceagentbench \
    --prefix sab_mate_openai_gpt-5_2025 \
    --prefix sab_mate_openai_gpt-5-mini_2025 \
    --prefix sab_mate_openai_gpt-4o_2024 \
    --prefix sab_mate_openai_o3-mini_2025 \
    --prefix sab_mate_DeepSeek-R1 \
    --prefix sab_mate_deepseek-ai_DeepSeek-V3 \
    --merge-input traces/sab_mate_openai_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_openai_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_DeepSeek-R1_MERGED_UPLOAD.json \
    --merge-input traces/sab_mate_deepseek-v3_MERGED_UPLOAD.json

# SAB_COW (4 models)
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/sab_cow_openai_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/sab_cow__sab_cow_openai_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/sab_cow_openai_o4-mini_low_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_cow_scienceagentbench \
    --prefix sab_cow_openai_gpt-4_1 \
    --prefix sab_cow_openai_o3_2025 \
    --prefix sab_cow_openai_o4-mini_2025-04-16_high \
    --prefix sab_cow_openai_o4-mini_2025-04-16_low \
    --merge-input traces/sab_cow_openai_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/sab_cow_openai_o4-mini_low_MERGED_UPLOAD.json

# SAB_HUSKY (1 model)
python scripts/merge_traces.py --input 'traces/sab_husky__sab_husky_openai_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/sab_husky_scienceagentbench \
    --prefix sab_husky_openai_gpt-4_1 \
    --merge-input traces/sab_husky_openai_gpt-4_1_MERGED_UPLOAD.json

echo "✓ SAB completed"
fi

# ============================================================
# COLBENCH
# ============================================================

if [ "$BENCHMARK" = "all" ] || [ "$BENCHMARK" = "colbench" ]; then
echo "=== COLBENCH ==="

# COL_IVY (9 models)
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4_1_2025-04-14_*_UPLOAD.json' --output traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3_2025-04-16_medium_*_UPLOAD.json' --output traces/col_ivy_o3_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_high_*_UPLOAD.json' --output traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o4-mini_2025-04-16_low_*_UPLOAD.json' --output traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5_2025-08-07_medium_*_UPLOAD.json' --output traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-5-mini_2025-08-07_*_UPLOAD.json' --output traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_gpt-4o_2024-11-20_*_UPLOAD.json' --output traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_o3-mini_2025-01-31_high_*_UPLOAD.json' --output traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_ivy__col_ivy_DeepSeek-R1_1_*_UPLOAD.json' --output traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_ivy_colbench_backendprogramming \
    --prefix col_ivy_gpt-4_1_2025-04-14 \
    --prefix col_ivy_o3_2025-04-16_medium \
    --prefix col_ivy_o4-mini_2025-04-16_high \
    --prefix col_ivy_o4-mini_2025-04-16_low \
    --prefix col_ivy_gpt-5_2025-08-07_medium \
    --prefix col_ivy_gpt-5-mini_2025-08-07 \
    --prefix col_ivy_gpt-4o_2024-11-20 \
    --prefix col_ivy_o3-mini_2025-01-31_high \
    --prefix col_ivy_DeepSeek-R1_1 \
    --merge-input traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_gpt-4o_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json \
    --merge-input traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json

# COL_ZUCK (3 models)
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_gpt-4_1-2025-04-14_*_UPLOAD.json' --output traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o3-2025-04-16_low_*_UPLOAD.json' --output traces/col_zuck_o3_low_MERGED_UPLOAD.json --force && \
python scripts/merge_traces.py --input 'traces/col_zuck__col_zuck_o4-mini-2025-04-16_high_*_UPLOAD.json' --output traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json --force

python scripts/extract_weave_traces.py \
    --project ronanhansel-hanoi-university-of-science-and-technology/col_zuck_colbench_backendprogramming \
    --prefix col_zuck_gpt-4_1-2025-04-14 \
    --prefix col_zuck_o3-2025-04-16_low \
    --prefix col_zuck_o4-mini-2025-04-16_high \
    --merge-input traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o3_low_MERGED_UPLOAD.json \
    --merge-input traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json

echo ""
echo "Adding dialogue history from results directory..."
RESULTS_DIR="results/colbench_backend_programming"

# COL_IVY - Add dialogues
echo "  Processing col_ivy (9 models)..."
python scripts/add_colbench_dialogues.py traces/col_ivy_gpt-4_1_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_gpt-4_1_2025-04-14_*" --output traces/col_ivy_openai_gpt-4_1_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_o3_medium_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_o3_2025-04-16_medium_*" --output traces/col_ivy_openai_o3_medium_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_o4-mini_high_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_o4-mini_2025-04-16_high_*" --output traces/col_ivy_openai_o4-mini_high_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_o4-mini_low_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_o4-mini_2025-04-16_low_*" --output traces/col_ivy_openai_o4-mini_low_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_gpt-5_medium_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_gpt-5_2025-08-07_medium_*" --output traces/col_ivy_openai_gpt-5_medium_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_gpt-5-mini_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_gpt-5-mini_2025-08-07_*" --output traces/col_ivy_openai_gpt-5-mini_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_gpt-4o_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_gpt-4o_2024-11-20_*" --output traces/col_ivy_openai_gpt-4o_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_o3-mini_high_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_o3-mini_2025-01-31_high_*" --output traces/col_ivy_openai_o3-mini_high_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_ivy_DeepSeek-R1_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_ivy_DeepSeek-R1_1_*" --output traces/col_ivy_DeepSeek-R1_WITH_DIALOGUES.json

echo "  col_ivy dialogues added!"

# COL_ZUCK - Already has _WITH_DIALOGUES.json, but recreate to be consistent
echo "  Processing col_zuck (3 models)..."
python scripts/add_colbench_dialogues.py traces/col_zuck_gpt-4_1_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_zuck_gpt-4_1-2025-04-14_*" --output traces/col_zuck_openai_gpt-4_1_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_zuck_o3_low_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_zuck_o3-2025-04-16_low_*" --output traces/col_zuck_openai_o3_low_WITH_DIALOGUES.json
python scripts/add_colbench_dialogues.py traces/col_zuck_o4-mini_high_MERGED_UPLOAD.json --results-dir "$RESULTS_DIR" --run-pattern "col_zuck_o4-mini-2025-04-16_high_*" --output traces/col_zuck_openai_o4-mini_high_WITH_DIALOGUES.json

echo "  col_zuck dialogues added!"
echo ""
echo "✓ ColBench completed (with dialogues)"
fi

echo ""
echo "============================================================"
echo "MERGE + WEAVE COMPLETE!"
echo "============================================================"
echo ""

# Print summary of traces with/without remote calls
echo "=== TRACE SUMMARY ==="
echo ""

python3 << 'PYTHON_SUMMARY'
import json
import glob
from pathlib import Path

traces = []
# Only check FINAL output files (not intermediate _MERGED_UPLOAD.json or individual __*.json files)
for pattern in ["scicode_honey_*.json", "scicode_sea_*.json", "prop_*.json", "iter1_*.json",
                "sab_mate_*.json", "sab_cow_*.json", "sab_husky_*.json", "col_ivy_*.json", "col_zuck_*.json"]:
    for f in glob.glob(f"traces/{pattern}"):
        # Skip intermediate files - only check final outputs
        if "_MERGED_UPLOAD.json" in f or "__" in f:
            continue
        traces.append(f)

traces_with_calls = []
traces_without_calls = []
traces_with_logs = []
traces_without_logs = []

for trace_path in sorted(traces):
    try:
        with open(trace_path) as f:
            data = json.load(f)

        # Check if it has conversation logs
        raw_logging = data.get("raw_logging_results", [])
        has_logs = len(raw_logging) > 0

        # Check if it mentions remote calls (from weave)
        config = data.get("config", {})

        name = Path(trace_path).name

        if has_logs:
            traces_with_logs.append((name, len(raw_logging)))
        else:
            traces_without_logs.append(name)

    except Exception as e:
        pass

print(f"✅ Traces WITH conversation logs: {len(traces_with_logs)}")
print(f"❌ Traces WITHOUT conversation logs: {len(traces_without_logs)}")

if traces_without_logs and len(traces_without_logs) <= 20:
    print(f"\nTraces without logs:")
    for name in traces_without_logs:
        print(f"  {name}")
elif traces_without_logs:
    # Group by prefix
    by_prefix = {}
    for name in traces_without_logs:
        prefix = name.split('_')[0] + '_' + name.split('_')[1] if '_' in name else name
        by_prefix.setdefault(prefix, []).append(name)
    print(f"\nTraces without logs (grouped by prefix):")
    for prefix, files in sorted(by_prefix.items()):
        print(f"  {prefix}: {len(files)} traces")

print(f"\nTotal traces processed: {len(traces_with_logs) + len(traces_without_logs)}")
PYTHON_SUMMARY

echo ""
echo "============================================================"
echo "Next step: Run RUBRIC EVALUATION"
echo "  ./bin/RUBRIC_COMMANDS.sh $BENCHMARK"
echo "============================================================"
