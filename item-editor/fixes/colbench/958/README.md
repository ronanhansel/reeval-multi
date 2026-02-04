# Task 958 Fix: Clothing Recommender with Arbitrary Constants

## IFE Summary

The hidden reference solution contains very specific, non-intuitive logic with exact string outputs:

```python
def get_ideal_clothing(climate, temperature, season, has_sensory_needs):
    if climate == 'cold' and temperature < 60:
        return ['undershirt', 't-shirt', 'hoodie', 'rain jacket']
    elif season == 'summer':
        return ['t-shirt']
    elif has_sensory_needs:
        return ['ribbed tank top', 't-shirt']
    else:
        return ['t-shirt', 'hoodie']
```

Key issues:
1. Temperature threshold of 60 (units not specified - Fahrenheit assumed)
2. Climate value must be exactly 'cold'
3. Season check for 'summer' overrides sensory needs (priority order)
4. Exact clothing item strings (e.g., 'ribbed tank top') not derivable
5. Simulated user said: "I don't know the temperature scale or range, climate categories, or temperature comfort bands"

## Evidence

- Hidden threshold: `temperature < 60` (Fahrenheit)
- Hidden climate check: `climate == 'cold'`
- Hidden priority: summer season check before sensory needs
- Hidden exact strings: 'ribbed tank top', 'undershirt', 'hoodie', 'rain jacket'
- User explicitly denied knowing these values

## Fix Strategy

The specific threshold, exact string values, and priority order are all arbitrary and not derivable from general clothing recommendations. These must be specified.

## Fix Applied

- `instruction_override.json`: Specifies the exact piecewise logic, thresholds, and clothing items
