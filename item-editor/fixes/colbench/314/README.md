# Task 314: Student Performance Improvement - IFE Fix

## Problem Summary
The task description says:
- Grades are "from 1 to 10"
- Function should "tell me if a student improved" (implies boolean)

But the hidden implementation:
```python
def calculate_performance_improvement(before_grade, after_grade):
    if before_grade == 0:
        if after_grade == 0:
            return 0
        else:
            return 1
    else:
        return (after_grade - before_grade) / before_grade * 100
```

This has TWO contradictions:
1. Returns percentage (numeric), not boolean for "did they improve?"
2. Has `before_grade == 0` handling, which is impossible under stated 1-10 domain

## IFE Type
**Specification Mismatch / Contradictory Requirements**

The natural-language spec (boolean improvement, grades 1-10) directly contradicts the hidden expected behavior (percentage return, 0-handling).

## Fix Approach
Clarify that the function returns percentage improvement (not boolean), and specify the edge case handling.

## Evidence
- Task: "tell me if a student improved" and "grades from 1 to 10"
- Hidden: returns `(after_grade - before_grade) / before_grade * 100` (percentage)
- Hidden: has unreachable `before_grade == 0` logic under stated constraints
