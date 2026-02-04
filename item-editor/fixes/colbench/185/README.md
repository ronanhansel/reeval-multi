# Task 185: Time Dilation Calculator

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Physics Formula Direction Mismatch Between Task and Implementation

### Problem Description
```
I want the LLM agent to help me write a Python function that calculates the time dilation
effect according to Einstein's theory of relativity. Think of a car moving at high speed
and explain to me how to calculate the time that passes for the observer inside the car
relative to the observer outside the car.
The signature: def calculate_time_dilation(rest_time, relative_velocity, speed_of_light)
```

### Hidden Ground Truth
```python
def calculate_time_dilation(rest_time, relative_velocity, speed_of_light):
    return rest_time * (1 - (relative_velocity ** 2 / speed_of_light ** 2)) ** 0.5
```

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "Hidden info defines: `return rest_time * (1 - (relative_velocity ** 2 / speed_of_light ** 2)) ** 0.5`. This corresponds to converting coordinate time to proper time (delta_tau = delta_t * sqrt(1-v^2/c^2)), but the task prompt frames the goal as 'time that passes for the observer inside the car relative to the observer outside'"

2. **Evaluation 6 (Grade 1.0):** "The benchmark's hidden reference implementation uses a nonstandard/incorrect mapping... the dialogue task and subsequent user clarification state `rest_time` is proper time inside the car and the function should return outside time, which physically requires division by sqrt(1 - v^2/c^2)"

### Root Cause - Physics Direction Reversal

**The Time Dilation Formula:**
- Standard notation: `t = gamma * t_0` where `t` is dilated time (outside observer), `t_0` is proper time (moving observer)
- `gamma = 1 / sqrt(1 - v^2/c^2)`

**Task Description:**
- "time that passes for the observer **inside the car** relative to **observer outside**"
- Input: `rest_time` (described as time in moving frame = proper time)
- Expected output: time as seen by outside observer (coordinate time)
- Formula should be: `t_outside = rest_time * gamma = rest_time / sqrt(1 - v^2/c^2)`

**Hidden Implementation:**
- Returns: `rest_time * sqrt(1 - v^2/c^2)`
- This is the INVERSE operation (coordinate → proper, not proper → coordinate)

**The evaluations confirm:**
- User clarified: "rest_time represents the time inside the car (proper time)"
- User clarified: "I want the function to return the time measured by the outside observer"
- But hidden code returns `rest_time * sqrt(...)` which is proper time from coordinate time!

### Why This is NOT Fixable at Task Level

**Cannot be fixed** because the hidden ground truth is physically incorrect for the stated task:
1. Task asks: proper time → coordinate time (multiply by gamma)
2. Hidden code does: rest_time * sqrt(...) = divide by gamma
3. These are inverse operations

Fixing would require:
- Either changing the task description to match the hidden formula
- Or changing the hidden formula to match the task description

Neither is possible through instruction_override or evaluation_override.

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error (physics formula direction mismatch). The benchmark would need to:
1. Clarify in task that `rest_time` is coordinate time (outside observer's time)
2. Or fix hidden formula to use `gamma * rest_time` instead of `sqrt(...) * rest_time`

Documenting as unfixable IFE (spec/implementation physics contradiction).
