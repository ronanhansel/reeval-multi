# Task 819 Fix: Driver Estimation with Hidden Constants

## IFE Summary

The hidden reference solution contains a specific formula with magic constants and a special-case year:

```python
def estimate_drivers(total_travelers, year, economy_impact):
    if year == 2012:
        return total_travelers * 0.89
    else:
        return total_travelers * (0.89 - (economy_impact / 100))
```

The simulated user repeatedly stated they didn't know these values:
- "I don't have historical baseline percentages or specific details on how economy_impact should affect the percentage"
- "I don't know how economy_impact should be treated or how it affects the driving share"

## Evidence

- Hidden baseline share: `0.89` (89%)
- Hidden special case: `year == 2012` ignores economy_impact
- Hidden economy formula: `0.89 - (economy_impact / 100)`
- User explicitly denied having these values despite them being in hidden info

## Fix Strategy

The constant 0.89, the special year 2012, and the economy_impact scaling are arbitrary domain knowledge that cannot be inferred. These must be specified.

## Fix Applied

- `instruction_override.json`: Specifies the exact formula, baseline share, and special-case year
