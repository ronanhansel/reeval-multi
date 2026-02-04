# Task 15 Fix - Tip Calculation Hidden Solution Logic Error

## IFE Detected: YES

## Issue Description

Task 15 has **two separate issues** depending on whether it's processed as backend or frontend:

### Backend Task: Hidden Reference Solution Logic Error

The hidden reference solution contains an **incorrect/arbitrary aggregation formula** that contradicts the natural interpretation of the task:

**Task Description**: Calculate total tips based on "number of individuals I need to tip in each category"

**Natural Interpretation**: Multiply each category's tip amount by its count, then sum:
```python
total = sum(tip * count for tip, count in zip(tip_amounts, individuals_tipped))
```

**Hidden Reference Solution** (from evidence):
```python
for tip_amount in tip_amounts.values():
    total_tips += tip_amount
return total_tips * len(individuals_tipped)
```

This formula:
1. Sums all tip amounts (ignoring per-category counts)
2. Multiplies by the LENGTH of the individuals_tipped list
3. Completely ignores the actual count values in individuals_tipped

This is mathematically different from what the task describes and cannot be derived from natural language understanding of the problem.

### Frontend Task: Parser Regex Mismatch (Same as other frontend tasks)

Evidence shows the same parser issue:
> "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it."

### Simulated User Information Gap

The simulated user reports "I don't know" when asked about the formula because the hidden information is code, not natural language requirements. The user cannot naturally communicate the arbitrary formula.

## Root Cause

1. **Backend**: Hidden solution uses an arbitrary/incorrect formula that doesn't match task semantics
2. **Frontend**: Parser configured for Python on HTML tasks
3. **Simulated User**: Cannot extract/communicate arbitrary implementation details from code

## Fix Required

### For Backend Task
The hidden reference solution should be corrected to match the natural task interpretation:
```python
def calculate_total_tips(tip_amounts, individuals_tipped):
    return sum(tip * count for tip, count in zip(tip_amounts.values(), individuals_tipped))
```

OR the task description should be clarified to explain the unusual formula.
