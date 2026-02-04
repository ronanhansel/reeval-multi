# Task 797 Fix: Bread Slashing with Hidden Thresholds

## IFE Summary

The hidden reference solution contains specific magic constants and piecewise branching logic that the simulated user explicitly stated they "don't know":

```python
def get_slashing_recommendation(dough_type, proofing_time, oven_temperature):
    if dough_type == 'sourdough':
        if proofing_time > 2:
            return 45, 0.75
        else:
            return 30, 0.5
    elif dough_type == 'yeast':
        if oven_temperature > 425:
            return 40, 0.6
        else:
            return 35, 0.4
```

The agent asked appropriate clarifying questions about baseline values and thresholds, but the simulated user responded:
- "I don't have specific slashing angles or depths in mind"
- "I don't know the baseline recommendations for sourdough or yeast breads"

## Evidence

- Hidden thresholds: `proofing_time > 2`, `oven_temperature > 425`
- Hidden return values: `(45, 0.75)`, `(30, 0.5)`, `(40, 0.6)`, `(35, 0.4)`
- Simulated user explicitly denied knowing these values despite them being in hidden info

## Fix Strategy

The arbitrary constants (why 45 degrees? why 0.75 inches? why the threshold of 2 hours?) cannot be derived from baking science without specification. These must be communicated.

## Fix Applied

- `instruction_override.json`: Specifies the exact thresholds and return values expected
