# Task 292: Average Yards Per Game - IFE Fix

## Problem Summary
The task asks for "average yards per game" but the hidden implementation returns a **season total**, not an average:
```python
return (total_rushing_yards * read_option_percentage / 100) + total_passing_yards
```

The function name and description say "average per game" but:
1. There's no `games_played` parameter in the signature
2. The hidden implementation performs no division/averaging
3. The formula is non-obvious: only read-option-adjusted rushing yards plus all passing yards

## IFE Type
**Specification Mismatch / Misleading Function Name**

The natural language requirement ("average yards per game") directly contradicts the hidden expected behavior (no averaging). An agent following the stated requirement would fail.

## Fix Approach
Clarify that despite the function name, this calculates a "weighted total" rather than a per-game average, and specify the exact formula.

## Evidence
- Function name: `calculate_average_yards_per_game`
- Task description: "calculates the average yards per game... for all the games in a season"
- Hidden implementation: `return (total_rushing_yards * read_option_percentage / 100) + total_passing_yards` (no division)
- Function signature has only 4 params with no way to compute games played
