# Task 73: EEG Signal Visualization Fix

## Root Cause Analysis

### Observed Failure
- **valid_program**: 1 (code executed successfully)
- **codebert_score**: 0.793
- **success_rate**: 0 (failed visual evaluation)

### GPT-4 Judge Feedback
- Generated plot: 16 time points
- Ground truth: 200 time points
- "Signal Shape: The first figure shows more abrupt changes and less smooth curves"
- Final scores: 20

### Failure Reason
The agent's output has significantly fewer time points (16) than the ground truth (200). This suggests the agent may have:
1. Averaged over the wrong dimensions (including time dimension)
2. Subsampled the data incorrectly
3. Misunderstood which axis corresponds to time points

The task instruction is ambiguous:
> "Images and channels corresponding to each timepoint should be averaged before plotting"

This doesn't clearly specify:
- The data array shape/structure
- Which dimensions to average over
- What the expected final plot dimensions should be

### Is This an IFE?

**Partial Yes** - The task instruction lacks clarity about data structure:
1. The data shape (images, channels, timepoints) is not specified
2. The domain knowledge only says "EEG measures electrical activity" - no structure info
3. The expected output (200 time points on x-axis) isn't documented

### Fix Type: Instruction Clarification

Add clarification about:
1. Data array structure
2. Which dimensions to average
3. Expected output format

## Why This Preserves Task Difficulty

- The visualization task remains unchanged
- The fix only clarifies data structure, not the solution
- The agent still needs to implement the averaging and plotting correctly

## Expected Outcome

After clarification, agents should:
1. Load the numpy arrays correctly
2. Average over images and channels (not time)
3. Produce a plot with all 200 time points preserved
