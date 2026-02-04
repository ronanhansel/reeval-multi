# Task 900 Fix: Student Separation with Return Type Mismatch

## IFE Summary

There is a fundamental mismatch between what the hidden solution expects and what the simulated user requested:

**Hidden solution:**
```python
def separate_students(students_grades, passing_grade):
    return [student for student, grade in students_grades.items() if grade >= passing_grade]
```
- Returns a single flat list of student names who passed
- Only returns passing students, no "wicked" (failing) students

**User-requested behavior:**
- "I would like the function to return a tuple of two lists, with the first list containing the names of 'righteous' students and the second list containing the names of 'wicked' students"
- Asked for a dictionary with keys 'righteous' and 'wicked', including both names and grades

## Evidence

- Hidden return type: `List[str]` (only passing students)
- User-requested return type: `Tuple[List, List]` or `Dict[str, List]` (both groups, with grades)
- Direct contradiction in return structure

## Fix Strategy

This is a specification contradiction: the user's stated requirements cannot satisfy the hidden tests. The fix must clarify the actual expected return format.

## Fix Applied

- `instruction_override.json`: Specifies the exact return format (single list of passing student names only)
