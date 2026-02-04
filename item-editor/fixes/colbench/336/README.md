# Task 336: WoW Priest Leveling Spec - IFE Fix

## Problem Summary
The task asks for a WoW priest leveling spec recommendation based on level, role, and group composition. The hidden implementation has very specific, arbitrary rules:

```python
def get_recommended_leveling_spec(level, role, group_comp):
    if level < 50 and role == 'priest' and 'mage' in group_comp:
        return 'Holy'
    elif level >= 50 and role == 'priest' and 'mage' in group_comp:
        return 'Shadow'
    else:
        return 'Not enough information to recommend a leveling spec'
```

Key arbitrary details:
- Level 50 is the threshold (not stated anywhere)
- Only considers 'mage' in group_comp (simple string membership)
- Returns exact strings 'Holy', 'Shadow', or 'Not enough information...'

## IFE Type
**Hidden Information Design Issue / Arbitrary Implementation Constants**

The level-50 cutoff and the specific recommendation logic are arbitrary constants that:
1. Are not stated in the task
2. Don't follow any standard WoW game knowledge
3. Cannot be elicited from the simulated user

## Fix Approach
Clarify the specific rules and thresholds expected.

## Evidence
- Hidden: `if level < 50` threshold (arbitrary)
- Hidden: `'mage' in group_comp` (simple membership check)
- Hidden: exact return strings including fallback message
- Simulated user responds "I don't know" to clarifying questions about thresholds
