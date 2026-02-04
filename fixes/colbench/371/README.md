# Task 371: Groundhog Day Spring Prediction - IFE Fix

## Problem Summary
The task asks to "predict the number of days until spring" using groundhog shadow data and temperatures. The hidden implementation uses completely arbitrary magic numbers:

```python
def predict_days_until_spring(shadows, dates, state_temperatures):
    shadowdays = [date for date, shadow in zip(dates, shadows) if shadow]
    avgtemp = sum(state_temperatures) / len(state_temperatures)
    if avgtemp < 40:
        return 42 if shadowdays else 34
    else:
        return 21 if shadowdays else 14
```

Key arbitrary details:
1. Temperature threshold: 40°F
2. Magic return values: exactly 42, 34, 21, or 14 (no other values possible)
3. `shadowdays` is truthy if ANY shadow was True (ignores dates)
4. The `dates` parameter is effectively unused except for zip pairing

## IFE Type
**Hidden Information Design Issue / Arbitrary Magic Constants**

The 40°F threshold and the four magic return values (42/34/21/14) are completely arbitrary and not derivable from any stated requirement or Groundhog Day tradition.

## Fix Approach
Specify the exact thresholds and return values expected.

## Evidence
- Hidden: `if avgtemp < 40` threshold (arbitrary)
- Hidden: returns only {42, 34, 21, 14} based on avgtemp and shadow presence
- Simulated user: "I don't know the specific temperature threshold"
- No traditional Groundhog Day folklore uses these specific numbers
