# Task 62: Sportscaster Rating Function

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Arbitrary Implementation Details Not Recoverable from Dialogue

### Problem Description
```
I want the LLM agent to write a python function to rate sportscasters based on certain
attributes, but I am not sure what attributes to include and how to quantify their
performance. The signature of the function is:
def evaluate_sportscaster(tonal_clarity, wit, experience, on_air_appearance)
```

### Hidden Ground Truth
```python
def evaluate_sportscaster(tonal_clarity, wit, experience, on_air_appearance):
    rating = 0
    if tonal_clarity > 7:
        rating += 2
    if wit > 6:
        rating += 2
    if experience > 5:
        rating += 1
    if on_air_appearance > 6:
        rating += 1
    return rating
```

### Evidence from Rubric Evaluations

1. **Evaluation 3 (Grade 1.0):** "The hidden ground truth is a specific threshold-based additive scoring rule (tonal_clarity>7 adds 2, wit>6 adds 2, experience>5 adds 1, on_air_appearance>6 adds 1). The simulated user is instructed to only answer using hidden info and otherwise say they don't know, but the user responses do not reveal any of these crucial thresholds/point increments."

2. **Evaluation 10 (Grade 1.0):** "User responses instead specify an incompatible scheme: 'custom weights: tonal_clarity 0.3, wit 0.3, experience 0.2, on_air_appearance 0.2... final score should be in the range 0–10'"

### Root Cause

The simulated user CANNOT provide the exact thresholds and point values because:
1. Hidden info is code, not natural-language facts
2. When asked about scoring rules, user says "I don't know" or provides invented schemes
3. The specific thresholds (7, 6, 5, 6) and point increments (2, 2, 1, 1) are arbitrary
4. No principled way to derive these from the task description

### Why This is NOT Fixable at Task Level

**Cannot be fixed without "nerfing"** - Any instruction clarification that reveals the exact thresholds would be giving away the answer. The problem is fundamental: the benchmark expects specific arbitrary constants but provides no mechanism for the agent to discover them through dialogue.

This is a benchmark design flaw where the hidden information contains implementation details that should be communicated to the agent but cannot be due to the simulated user constraints.

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error that cannot be fixed without revealing the answer. The benchmark would need to either:
1. Add the thresholds/scoring rules to the problem description
2. Allow the simulated user to extract these constants from the hidden code
3. Accept any reasonable threshold-based scoring scheme (not just the specific one)

Documenting as unfixable IFE.
