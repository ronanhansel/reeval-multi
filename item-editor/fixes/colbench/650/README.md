# Task 650 Fix: Omer Activity Categorization

## IFE Summary

**Task**: Categorize activities into two Omer time periods.

**Problem Description**: "I want to categorize different activities into two time periods during the Omer, but I'm not sure how to split them correctly."

**Hidden Reference Implementation**:
```python
def categorize_omer_activities(activities, omer_days):
    # Period 1: Days 1-24, Period 2: Days 25-49
    activities_in_period1 = ['shaving', 'haircuts', 'music']
    activities_in_period2 = ['parties', 'weddings', 'concerts']

    categorized_activities = {}
    for activity in activities:
        if activity in activities_in_period1:
            categorized_activities[activity] = omer_days[:24]
        elif activity in activities_in_period2:
            categorized_activities[activity] = omer_days[24:]
        else:
            categorized_activities[activity] = 'N/A'

    return categorized_activities
```

## Intrinsic Formation Error

**Type**: Under-specified Requirements with Arbitrary Hardcoded Data

**Issue**: The hidden implementation contains hardcoded activity-to-period mappings that cannot be derived from the prompt:
- Period 1 activities: shaving, haircuts, music
- Period 2 activities: parties, weddings, concerts

The prompt explicitly says "I'm not sure how to split them correctly" but provides no rules for the categorization.

**Why this is an IFE (not capability issue)**:
1. The activity lists are arbitrary - no logical rule connects them to period 1 vs 2
2. The prompt provides no categorization criteria
3. The output format (returning day slices vs labels) is unspecified
4. Unknown activities should return 'N/A' - also unspecified
5. The 24/25 split (days 1-24 vs 25-49) is not mentioned in the prompt

**Even a domain expert couldn't derive this mapping**: The Omer restrictions in Jewish law don't neatly split activities this way.

## Fix Type

**Instruction Clarification** - Specify the activity-to-period mapping, day split, and output format.

## Fix Rationale

The fix reveals the hardcoded categorization rules that are currently hidden. While this significantly reduces ambiguity, it's necessary because the rules are arbitrary and cannot be reasonably discovered.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the dictionary logic, handle unknown activities, and correctly slice the omer_days list.
