# Task 56: Temperature Statistic Visualization Fix

## Root Cause Analysis

### Observed Failure
- **valid_program**: 1 (code executed successfully)
- **success_rate**: 0 (output didn't match ground truth)

### Failure Reason
The agent produced a **bar chart showing occurrences over time**, while the ground truth is a **geographical heatmap** showing spatial distribution of warm spell occurrences.

### Is This an IFE?

**Partial Yes** - The task instruction is ambiguous:

> "Calculate and plot the number of occurrences where the temperature exceeds 280K for five consecutive years"

This could reasonably be interpreted as:
1. **Temporal**: How many grid cells had warm spells in each time period? (agent's interpretation)
2. **Spatial**: For each grid cell, how many warm spells occurred over the 240 years? (ground truth)

The domain knowledge mentions `cube.collapsed()` for aggregation but doesn't specify the target dimension.

### Fix Type: Instruction Clarification

The fix clarifies that the visualization should be a **geographical/spatial heatmap** showing the count of warm spell occurrences at each location.

## Why This Preserves Task Difficulty

- The scientific computation remains the same (identifying 5-year consecutive warm periods)
- The fix only clarifies the expected output format, not the algorithm
- No hints about implementation are provided
- The agent still needs to understand and process the multi-dimensional NetCDF data correctly

## Expected Outcome

After clarification, agents should produce a spatial heatmap rather than a time series, matching the ground truth format.
