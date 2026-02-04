# Task 476: Long-Toss Distance Calculator - IFE Fix

## Issue Summary
The hidden reference solution uses a specific piecewise formula with arbitrary constants that the simulated user explicitly could not provide.

## Root Cause
**Intrinsic Formation Error**: Undiscoverable arbitrary constants:

1. **Hidden formula**:
   - age < 25: `(30 - age) * (velocity/100) * 50`
   - age < 31: `(35 - age) * (velocity/100) * 40`
   - age >= 31: `(40 - age) * (velocity/100) * 30`
2. **Task description**: "I'm not sure how the age and velocity should affect the distance"
3. **Simulated user**: "I don't know the exact reference point values", "I don't know the specific slopes"

The constants 25/31 (thresholds), 30/35/40 (offsets), and 50/40/30 (multipliers) are completely arbitrary.

## Evidence
- Hidden info: `if age < 25: distance = (30 - age) * (velocity / 100) * 50` etc.
- Simulated user: "I don't know the exact reference point values... I also don't know the specific slopes for age and velocity"
- The problem description itself acknowledges uncertainty about the relationship

## Fix Applied
**instruction_override.json**: Specify the exact piecewise formula expected.

## Why This Fix is Fair
- The task still requires implementing conditional logic and arithmetic
- The agent must understand piecewise functions and velocity scaling
- Only the arbitrary constants are clarified (which user explicitly said were unknown)
