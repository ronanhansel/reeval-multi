# Task 705 Fix: Pelican Fish Calculation

## IFE Summary

**Task**: Calculate total number of fish a pelican can store.

**Problem Description**: "Can you write a Python function to calculate the total number of fish a pelican can store in its pouch? I want to know the relationship between the pouch capacity and the total fish it can hold."

**Hidden Reference Implementation**:
```python
def calculate_total_fish(pouch_capacity, fish_per_dive, dives_per_flight):
    return pouch_capacity * dives_per_flight
```

## Intrinsic Formation Error

**Type**: Misleading Parameter / Decoy Variable

**Issue**: The hidden implementation **completely ignores** `fish_per_dive` despite:
1. The parameter name strongly suggesting it should be used
2. The prompt mentioning "relationship" between parameters
3. The natural formula being `fish_per_dive * dives_per_flight`

**Why this is an IFE (not capability issue)**:
1. Including a named parameter that's ignored is misleading
2. The natural interpretation: total_fish = fish_per_dive × dives_per_flight
3. Alternative interpretation: total_fish = min(pouch_capacity, fish_per_dive × dives_per_flight)
4. Hidden interpretation (pouch_capacity × dives_per_flight) is semantically odd
5. Why would you multiply pouch capacity by dives? That implies the pouch is filled/emptied each dive.

**Simulated User Issues**:
- When asked for clarification, user responses were inconsistent with hidden formula
- User cannot explain why fish_per_dive is irrelevant

## Fix Type

**Instruction Clarification** - Specify that fish_per_dive is not used and the formula is pouch_capacity × dives_per_flight.

## Fix Rationale

The fix reveals the actual formula relationship. While unusual, this is necessary because the presence of an unused parameter is actively misleading.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the calculation. We're only correcting the misleading function signature.
