# Task 69: Single Cell Analysis UMAP Visualization Fix

## Root Cause Analysis

### Observed Failure
- **valid_program**: 1 (code executed successfully)
- **codebert_score**: 0.851 (high code similarity)
- **success_rate**: 0 (failed visual evaluation)

### GPT-4 Judge Feedback
- "The generated plot has large, overlapping labels which make it difficult to read"
- "The ground truth plot has a clear legend on the right side"
- "The overall clustering structure is somewhat similar"
- Final scores: 30-50

### Failure Reason
The agent's UMAP visualization used **inline text labels** on the scatter plot, while the ground truth uses a **clean legend** on the right side. This stylistic difference caused the GPT-4 visual judge to penalize the output.

The scientific content (UMAP clustering, cell type coloring) appears correct, but the presentation style differs.

### Is This an IFE?

**Partial Yes** - This is a figure evaluation subjectivity issue:
1. The task says "with color representing the cell type" but doesn't specify labeling style
2. The GPT-4 judge penalizes functionally equivalent but stylistically different visualizations
3. This is a known issue in ScienceAgentBench (documented in benchmark paper as "subjective variance")

### Fix Type: Instruction Clarification + Evaluation Tolerance

Add clarification about expected visualization style (legend vs inline labels).

## Why This Preserves Task Difficulty

- The scientific computation remains unchanged (PCA, UMAP, cell type analysis)
- The fix only clarifies presentation style, not the algorithm
- The agent still needs to correctly implement the full single-cell analysis pipeline

## Expected Outcome

After clarification, agents should produce UMAP plots with:
1. Colors for cell types (already correct)
2. A legend (not inline labels) for clarity
3. Clean, professional visualization matching ground truth style
