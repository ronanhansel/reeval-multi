# Task 473: Election Majority Calculator - IFE Fix

## Issue Summary
The hidden reference solution simply returns the sum of all per-district vote requirements, ignoring the "majority of districts" logic that the simulated user explicitly requested.

## Root Cause
**Intrinsic Formation Error**: Severe spec/test mismatch:

1. **Hidden solution**: `return sum(winning_votes_per_district)` - just sums all values, ignores electoral_districts and total_votes parameters
2. **Simulated user guidance**: "must win a majority of districts", "return minimum total votes needed to win those districts"
3. These are fundamentally different algorithms:
   - User-specified: Sum the smallest (districts//2 + 1) thresholds
   - Hidden: Sum ALL thresholds regardless of majority rule

## Evidence
- Hidden info: `total_votes_required = sum(winning_votes_per_district); return total_votes_required`
- Simulated user: "The candidate must win a majority of the electoral districts. The function should return an integer representing the minimum total votes needed to win those districts."
- These specifications are mutually exclusive

## Fix Applied
**instruction_override.json**: Clarify that the function should return the sum of ALL per-district vote requirements.

## Why This Fix is Fair
- The task still requires summing a list of values
- The fix clarifies the actual expected behavior (sum all) vs the misleading user guidance (majority)
- The simplicity of the actual task is the task's design, not something we're changing
