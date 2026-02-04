# Task 334: Artist Performance Scoring - IFE Fix

## Problem Summary
The task says:
- "sales figure in thousands" (implies pre-scaled values)
- "number of awards they've received" (implies integer count)

But the hidden implementation expects:
```python
def calculate_artist_performance(releases, awards, years_experience):
    base_score = 0
    for release in releases:
        base_score += 1 + (release['sales'] // 100000)
    return base_score + (years_experience * 2) + len(awards)
```

This requires:
1. `releases` to be a list of dicts with `'sales'` key in ABSOLUTE units (not thousands)
2. `awards` to be a LIST (uses `len()`), not a count
3. Specific formula: `1 + (sales // 100000)` per release, plus `years * 2`, plus `len(awards)`

## IFE Type
**Specification Mismatch / Input Type Contradiction**

The task description specifies different input types/units than the hidden implementation expects.

## Fix Approach
Clarify the exact input structure and scoring formula.

## Evidence
- Task: "sales figure in thousands" vs hidden: `release['sales'] // 100000` (absolute)
- Task: "number of awards" (int) vs hidden: `len(awards)` (requires list)
- Test case confirms: `calculate_artist_performance([{'sales': 100000}, {'sales': 200000}], ['Grammy'], 10)`
