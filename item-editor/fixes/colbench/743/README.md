# Task 743 Fix: THC Effects Duration Calculation

## IFE Summary

**Task**: Calculate how long THC effects take to kick in and wear off.

**Problem Description**: "Can you write a Python function to determine how long it will take for the effects of a pot brownie to kick in and wear off based on the amount of THC consumed and the user's weight?"

**Hidden Reference Implementation**:
```python
def calculate_thc_effects(thc_amount, user_weight):
    thc_per_pound = thc_amount / user_weight
    if thc_per_pound < 0.01:
        effects_duration = 30
    elif thc_per_pound < 0.05:
        effects_duration = 60
    else:
        effects_duration = 120
    return effects_duration
```

## Intrinsic Formation Error

**Type**: Specification-Implementation Mismatch + Arbitrary Hidden Thresholds

**Issue 1 - Output Mismatch**: The prompt asks for TWO values ("kick in AND wear off"), but the implementation returns only ONE value (duration).

**Issue 2 - Arbitrary Thresholds**: The specific thresholds (0.01, 0.05 mg THC per pound) and duration values (30, 60, 120 minutes) are completely arbitrary and cannot be derived from the prompt.

**Why this is an IFE (not capability issue)**:
1. A faithful implementation of "kick in and wear off" would return two values (onset time + duration)
2. The piecewise threshold function (30/60/120 based on concentration) is arbitrary
3. The specific breakpoints (0.01, 0.05) have no basis in the prompt
4. No agent could guess these exact magic numbers from the description

## Fix Type

**Instruction Clarification** - Specify that only duration is returned, and provide the threshold logic.

## Fix Rationale

The fix reveals that despite the prompt wording, only duration is needed, and specifies the piecewise calculation rules. This corrects the spec-implementation mismatch and reveals the arbitrary thresholds.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the conditional logic correctly. We're correcting the misleading prompt and revealing arbitrary thresholds that couldn't be guessed.
