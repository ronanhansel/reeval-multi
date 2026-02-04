# Task 648 Fix: Dump Truck Rock Volume Calculation

## IFE Summary

**Task**: Calculate the volume of rocks a dump truck can carry.

**Problem Description**: "I want to know how to calculate the volume of rocks a large dump truck can carry."

**Hidden Reference Implementation**:
```python
def calculate_volume_truck_can_carry(weight_capacity, rock_density, height, width, length):
    volume = (weight_capacity / rock_density) / (height * width * length)
    return volume
```

## Intrinsic Formation Error

**Type**: Specification-Implementation Mismatch / Dimensionally Incorrect Formula

**Issue**: The hidden formula computes `(weight_capacity / rock_density) / (height * width * length)` which yields a **dimensionless ratio** (fraction), not a volume as the prompt requests.

**Dimensional Analysis**:
- `weight_capacity / rock_density` = mass / (mass/volume) = volume (correct for weight-limited volume)
- Dividing by `height * width * length` (another volume) gives: volume / volume = dimensionless

**Why this is an IFE (not capability issue)**:
1. The prompt asks for "volume of rocks" - a volume quantity
2. The natural interpretation is min(weight_limited_volume, geometric_volume)
3. The hidden formula produces a ratio (fraction of bed volume used), not a volume
4. This formula is physically unusual and not derivable from the prompt
5. The simulated user cannot explain units or the expected calculation

**Reasonable Interpretations (all would fail tests)**:
- `min(weight_capacity/rock_density, height*width*length)` - bounded volume
- `weight_capacity / rock_density` - weight-limited volume only
- `height * width * length` - geometric bed volume

## Fix Type

**Instruction Clarification** - Specify that the function should return the weight-limited volume as a fraction of the bed volume.

## Fix Rationale

The fix reveals the actual formula relationship: the output is a utilization ratio (what fraction of bed volume can be filled given weight constraints), not an absolute volume.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the formula correctly with proper division order. We're only correcting the misleading task description.
