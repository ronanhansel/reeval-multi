# Task 344: Oxygen Supply Calculation - IFE Fix

## Problem Summary
The task asks to "calculate the oxygen supply in the body based on deep breathing exercises" taking into account breathing depth, frequency, and consumption rate.

The hidden implementation is trivially:
```python
def calculate_oxygen_supply(breathing_depth, breathing_frequency, oxygen_consumption_rate):
    return breathing_depth * breathing_frequency * oxygen_consumption_rate
```

This is:
1. Not physiologically meaningful (multiplying consumption rate INTO supply makes no sense)
2. An arbitrary formula with no domain justification
3. Not specified in the task (no formula, no units provided)

## IFE Type
**Hidden Information Design Issue / Arbitrary Formula**

The hidden formula is arbitrary and cannot be derived from the task description or any reasonable physiological model.

## Fix Approach
Specify that this uses a simplified model where oxygen supply is the product of all three inputs.

## Evidence
- Task: "calculate the oxygen supply" with "depth, frequency, and consumption rate"
- Hidden: simple multiplication of all three inputs
- Simulated user: "I don't know the specific units" and "I also don't have a specific formula"
- A reasonable physiological model would likely involve subtraction (intake - consumption), not multiplication
