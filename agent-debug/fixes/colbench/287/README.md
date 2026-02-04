# Task 287: Family Car Suitability Function

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Simulated User Contradicts Hidden Requirements + Magic String

### Problem Description
```
I want you to write a python function that determines if a car is suitable for a family
based on its specifications. The function should consider the number of seats, horsepower,
trunk volume, mpg highway, and door layout. Help me clarify what values are feasible for
a family car.
The signature: def is_family_car(seats, horsepower, trunk_volume, mpg_highway, door_layout)
```

### Hidden Ground Truth
```python
def is_family_car(seats, horsepower, trunk_volume, mpg_highway, door_layout):
    if seats >= 4 and trunk_volume >= 2 and mpg_highway >= 20 and door_layout == 'family_friendly':
        return True
    else:
        return False
```

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "None of these numeric thresholds/label are justified by the task description... trunk_volume >= 2 has no stated unit and is unrealistically small in common units... the required exact string 'family_friendly' for door_layout is not inferable from the prompt"

2. **Evaluation 6 (Grade 1.0):** "The simulated user responses steer the agent to implement a different, incompatible requirement: treating `door_layout` as an integer door count with `>= 4`, and explicitly ignoring horsepower"

### Root Cause - Multiple Issues:

**Issue 1: Magic String Requirement**
- Hidden code requires `door_layout == 'family_friendly'` (exact string match)
- How would agent know to use exactly this string?
- Other valid values like 'sedan', '4-door', 'minivan' would fail

**Issue 2: Simulated User Contradicts Hidden Code**
- User first says: "The door layout should be 'family_friendly'"
- Later contradicts: "For the door layout, let's require at least 4 doors"
- Later: "The `door_layout` argument will be provided as an integer number of doors"
- Agent follows user → implements `door_layout >= 4` as integer → fails tests

**Issue 3: Horsepower Ignored**
- Task says "consider... horsepower"
- Hidden code completely ignores horsepower parameter!
- This is misleading and inconsistent

**Issue 4: Unrealistic Thresholds**
- `trunk_volume >= 2` - 2 what? Liters? Cubic feet?
- 2 cubic feet is about 56 liters - extremely small for a family car
- Real family cars have 15-20+ cubic feet of trunk space

### Why This is NOT Fixable at Task Level

**Cannot be fixed** because:
1. Revealing `'family_friendly'` as magic string gives away the answer
2. Simulated user is actively contradicting the hidden requirements
3. The thresholds are unrealistic (trunk_volume >= 2 is nonsensical)
4. Task mentions horsepower but hidden code ignores it

## Fix Recommendation

**No code fix applied** - The task has multiple intrinsic formation errors:
1. Simulated user contradicts hidden requirements (door_layout type/value)
2. Magic string requirement is unguessable
3. Thresholds have no units and are unrealistic
4. Parameter (horsepower) is mentioned but ignored

The benchmark would need to:
1. Fix simulated user to consistently report hidden requirements
2. Add valid door_layout values to task description
3. Fix trunk_volume threshold to realistic values with units
4. Either use horsepower in the function or remove it from signature

Documenting as unfixable IFE (multiple contradictions and arbitrary requirements).
