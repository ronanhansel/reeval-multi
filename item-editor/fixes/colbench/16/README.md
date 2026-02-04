# Task 16 Fix - Emission Cuts Calculation Hidden Formula Issue

## IFE Detected: YES

## Issue Description

Task 16 has issues depending on whether it's a backend or frontend task:

### Backend Task: Hidden Solution Contains Arbitrary Baseline Year

**Task Description**: Calculate emission cuts "by a certain year" (mentions "aiming to halve emissions by 2050")

**Hidden Reference Solution** (from evidence):
```python
def calculate_emission_cuts(current_emissions, reduction_percentage, year_target):
    return (current_emissions * reduction_percentage) / 100 * (year_target - 2023)
```

**Issues with hidden solution**:
1. Uses hardcoded baseline year `2023` that is NOT mentioned in the task description
2. Multiplies by `(year_target - 2023)` implying linear scaling per year
3. This baseline year and scaling formula cannot be inferred from the task description

**Simulated User Response**:
> "I don't know the unit... I'm not sure... and I don't have information on the other points."

The user cannot naturally communicate the arbitrary 2023 baseline because the hidden info is code.

### Frontend Task: Parser Regex Mismatch

Evidence shows same parser issue as other frontend tasks:
> "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it."

### Frontend Task: Spec vs Ground Truth Contradiction

Task spec mentions:
- "sticky navigation menu"
- "two-column layout"
- "gradient colors"

But simulated user feedback says ground truth has:
- "plain white background with no navigation bar"
- Minimalist centered sections

## Root Cause

1. **Backend**: Hidden solution uses arbitrary hardcoded value (2023) not specified anywhere
2. **Frontend**: Parser rejects valid HTML; spec doesn't match ground truth

## Fix Required

### For Backend Task
Either:
1. Add the baseline year (2023) and scaling formula to the task description
2. OR change the hidden solution to not require undiscoverable information

### For Frontend Task
1. Fix parser to accept HTML
2. Align spec with ground truth (or vice versa)
