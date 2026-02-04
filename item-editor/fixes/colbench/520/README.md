# Task 520 Fix: Fantasy Football Player Calculation

## IFE Summary

**Task**: Calculate maximum number of footballers that can be bought given budget, skill level, and cost.

**Problem Description**: "The agent should create a function that can calculate the maximum number of footballers I can buy with my budget without exceeding it, given their skill level."

**Hidden Reference Implementation**:
```python
def calculate_players(skill_points, budget, player_cost):
    return budget // (skill_points * player_cost)
```

## Intrinsic Formation Error

**Type**: Under-specified requirements with arbitrary hidden implementation detail

**Issue**: The hidden reference treats `skill_points` as a **cost multiplier** (effective cost = skill_points * player_cost), but the prompt only says "given their skill level" without specifying how skill affects the calculation. This is an arbitrary implementation choice that cannot be derived from the problem description.

**Why this is an IFE (not capability issue)**:
1. The phrase "given their skill level" is ambiguous - it could mean:
   - Skill multiplies cost (hidden interpretation)
   - Skill is a filter (only buy players above certain skill)
   - Skill is informational only
   - Skill determines value-for-money
2. The simulated user cannot explain the formula because it's arbitrary
3. When asked, user says "I don't know" - confirming the requirement is undiscoverable through dialogue
4. Multiple reasonable interpretations exist, but tests enforce one specific arbitrary formula

## Fix Type

**Instruction Clarification** - Add missing specification that skill_points multiplies player_cost to determine effective cost.

## Fix Rationale

The fix adds the missing specification that reveals the intended formula relationship: skill_points acts as a cost multiplier. This makes the task solvable through normal dialogue while maintaining the core challenge of implementing the calculation correctly.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement integer division, handle the formula correctly, and produce working code. We're only revealing the specification that was arbitrarily hidden.
