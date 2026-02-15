# Task 798 Fix: Lung Disease Risk with Undisclosed Formula

## IFE Summary

The hidden reference solution contains a specific formula with arbitrary constants that the simulated user explicitly denied knowing:

```python
def calculate_lung_disease_risk(asbestos_exposure_level, age, years_of_exposure):
    risk = (asbestos_exposure_level * years_of_exposure) / (100 * age)
    return risk > 0.5
```

Key issues:
1. The simulated user said "I don't know the specific threshold or rule for 'high risk'" despite the formula being in hidden info
2. The user approved adding a 4th parameter (`smoking_pack_years`) which contradicts the 3-arg expected function
3. The user suggested categorical labels ("low/medium/high") while hidden solution returns boolean

## Evidence

- Hidden formula: `(asbestos_exposure_level * years_of_exposure) / (100 * age)`
- Hidden threshold: `> 0.5` returning boolean
- User contradictions:
  - "I don't know the specific threshold or rule for 'high risk'"
  - Approved 4-parameter signature
  - Requested categorical output instead of boolean

## Fix Strategy

The formula's magic constant (100) and threshold (0.5) are arbitrary and not medically derivable. The return type (boolean vs categorical) must be specified.

## Fix Applied

- `instruction_override.json`: Specifies the exact formula, threshold, and return type
