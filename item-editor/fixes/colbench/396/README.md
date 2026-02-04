# Task 396: Displaced People Shelter Management - IFE Fix

## Issue Summary
The hidden reference solution returns either an integer (shelters_needed) or the string 'Not enough shelters'. However, the simulated user guided the agent to implement a different return type (tuple with boolean, capacity, and needed count).

## Root Cause
**Intrinsic Formation Error**: Spec/test mismatch between simulated user guidance and hidden expected behavior:

1. Hidden solution: Returns `shelters_needed` (int) if enough shelters, otherwise `'Not enough shelters'` (string)
2. Simulated user first said: "return whether there are enough shelters" (implying boolean)
3. Simulated user later explicitly chose: tuple format `(fits: bool, capacity: int, needed: int)`

The agent correctly followed the user's stated preference, but the evaluation expects the hidden reference behavior.

## Evidence
- Hidden info: `return shelters_needed` or `return 'Not enough shelters'`
- Simulated user: "I would like the function to return a tuple like (fits: bool, capacity: int, needed: int)"
- These return types are mutually exclusive - no implementation can satisfy both

## Fix Applied
**instruction_override.json**: Clarify the expected return type in the problem description.

## Why This Fix is Fair
- The task still requires algorithmic thinking (ceiling division for shelters needed)
- The only change is clarifying the return format, which was contradictory
- The agent must still implement the comparison and calculation logic
