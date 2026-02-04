# Task 851 Fix: Soap Calculation with Hidden Piecewise Logic

## IFE Summary

The hidden reference solution contains specific piecewise logic with thresholds and multipliers:

```python
if water_amount > 0 and soil_alkalinity > 6:
    soap_amount = fat_amount * 0.8
elif water_amount > 0 and soil_alkalinity > 0:
    soap_amount = fat_amount * 0.5
else:
    soap_amount = 0
```

However, the simulated user:
1. Did not communicate these exact thresholds (6, 0) or multipliers (0.8, 0.5)
2. Instead engaged in choosing alternative modeling approaches not in hidden info:
   - "I prefer the linear option with a default of 1 g water per gram of fat"
   - "I prefer the linear option with pH_min set to 7 and pH_max set to 9"
3. Said "I don't know the specifics for base type or saponification value"

## Evidence

- Hidden thresholds: `soil_alkalinity > 6` and `soil_alkalinity > 0`
- Hidden multipliers: `0.8` and `0.5` of fat_amount
- User guided agent toward linear models and different pH ranges not in hidden info

## Fix Strategy

The thresholds and multipliers are arbitrary and not derivable from chemistry without specification. The user's responses actively misdirected toward incompatible models.

## Fix Applied

- `instruction_override.json`: Specifies the exact piecewise logic expected
