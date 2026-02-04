# Task 370: Witness Demographics Inference - IFE Fix

## Problem Summary
The task asks to "infer the demographics of individuals involved in violent incidents" from witness descriptions. The hidden implementation uses very specific, hardcoded keyword triggers:

```python
def get_incident_demographics(description):
    description = description.lower()
    age_demographics = {'young': 0, 'old': 0, 'unknown': 0}
    role_demographics = {'aggressor': 0, 'defender': 0, 'unknown': 0}
    if 'older' in description:
        age_demographics['old'] += 1
    elif 'young' in description or 'younger' in description:
        age_demographics['young'] += 1
    if 'throwing punches' in description or 'started a fight' in description:
        role_demographics['aggressor'] += 1
    elif 'no aggression' in description or "didn't want any part" in description:
        role_demographics['defender'] += 1
    return age_demographics, role_demographics
```

Key arbitrary details:
1. Only specific exact substrings trigger detection (not synonyms)
2. Returns a TUPLE of two dicts (not a single dict)
3. Each dict has an 'unknown' key
4. Only increments by 1 per description regardless of mentions

## IFE Type
**Hidden Information Design Issue / Arbitrary Keyword Triggers**

The exact trigger phrases and output format are arbitrary implementation details not derivable from the task description.

## Fix Approach
Specify the exact keywords to look for and the output format.

## Evidence
- Hidden triggers: exactly 'older', 'young'/'younger', 'throwing punches', 'started a fight', 'no aggression', "didn't want any part"
- Hidden output: tuple of (age_demographics, role_demographics) dicts with 'unknown' keys
- Task only says "infer demographics" without specifying keywords or format
