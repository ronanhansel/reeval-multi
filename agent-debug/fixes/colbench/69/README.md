# Task 69: Brownie Baking Time Calculator

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Arbitrary Magic Constants Not Recoverable from Dialogue

### Problem Description
```
I want to bake some rich and classic brownies, but I don't want to overbake them.
Can you write a python function to help me determine the baking time? I want to be
able to input the type of pan I'm using, the size of the pan, and the oven temperature,
and the function should output the baking time in minutes.
The signature: def calculate_baking_time(pan_size, oven_temperature, pan_type)
```

### Hidden Ground Truth
```python
def calculate_baking_time(pan_size, oven_temperature, pan_type):
    baking_time_per_area = 0.15 if pan_type == "round" else 0.1
    pan_area = 201.06176 if pan_size == 8 else 117
    return baking_time_per_area * pan_area * (oven_temperature / 325)
```

### Evidence from Rubric Evaluations

1. **Evaluation 4 (Grade 1.0):** "pan_size is treated as a categorical selector (8 -> area 201.06176 else 117) and the temperature factor is (oven_temperature / 325), while the task prompt frames pan_size as a general 'size of the pan'... there is no way for an agent to infer (a) the two hard-coded areas 201.06176 and 117, (b) that only pan_size==8 has special handling"

2. **Evaluation 10 (Grade 1.0):** "The simulated user did not communicate these exact constants/behaviors to the agent; instead it replied with vague/incorrect natural-language constraints... The agent then proceeded with arbitrary defaults"

### Root Cause

The hidden function contains completely arbitrary magic constants:
- `201.06176` and `117` as pan areas (only two values, not computed from pan_size)
- `0.15` vs `0.1` for time-per-area ratios
- Temperature scaling anchored at 325 (not derivable from baking knowledge)
- `pan_size == 8` is the only special case (why 8?)

The simulated user explicitly says "I don't know" when asked for these constants because they only exist as code literals, not as answerable facts.

### Why This is NOT Fixable at Task Level

**Cannot be fixed without "nerfing"** - Revealing these magic constants would give away the implementation. The problem is that the benchmark tests against arbitrary numbers that have no real-world baking basis.

The constants (201.06176, 117, 0.15, 0.1, 325) are not standard baking formulas - they're arbitrary values in the hidden code.

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error. The benchmark would need to either:
1. Add the exact formula/constants to the problem description
2. Use realistic baking formulas that could be reasonably derived
3. Accept any plausible baking time calculation (not just these specific constants)

Documenting as unfixable IFE.
