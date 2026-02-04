# Task 144: Georgian Kings in Date Range

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Spec/Tests Mismatch (Containment vs Overlap) + Reference Data Unavailable

### Problem Description
```
I want you to write a Python function that takes two parameters, start_year and end_year,
and returns a list of Georgian kings who were reigning in that period. The function
should use the data from the provided text as a reference. Make sure to include all
kings mentioned in the text.
The signature: def get_georgian_kings(start_year, end_year)
```

### Hidden Ground Truth
```python
def get_georgian_kings(start_year, end_year):
    georgian_kings = [
        {'name': 'Bagrat III', 'reign': (975, 1014)},
        {'name': 'David II (Aghmashenebeli)', 'reign': (1089, 1125)},
        # ... 11 kings total
    ]
    kings_in_range = [king for king in georgian_kings if
                      start_year <= king['reign'][0] and king['reign'][1] <= end_year]
    return kings_in_range
```

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "The hidden reference implementation filters kings using full containment: `start_year <= king['reign'][0] and king['reign'][1] <= end_year`. However, the simulated user instructions in the transcript explicitly define the requirement as overlap-inclusive."

2. **Evaluation 6 (Grade 1.0):** "A formation defect exists because the task specification requires using 'the provided text' as the sole reference, but the simulated user is unable to provide that text despite it being present in hidden information as executable code."

### Root Cause - Two Issues:

**Issue 1: The "provided text" doesn't exist**
- Task says "use the data from the provided text as a reference"
- But there is no text provided - only code with a hardcoded list
- Simulated user cannot provide this because hidden info is code, not text
- User responds: "I don't have the text listing the Georgian kings and their reign dates"

**Issue 2: Containment vs Overlap semantics**
- Hidden code uses FULL CONTAINMENT: `start_year <= reign_start AND reign_end <= end_year`
- Meaning: king's entire reign must be within the query range
- But natural interpretation is OVERLAP: include if any part of reign overlaps the range
- Simulated user confirmed overlap semantics when asked!

### Why This is NOT Fixable at Task Level

**Issue 1 could be fixed** by adding the king list to problem_description, BUT:
- This would not fix Issue 2 (containment vs overlap)
- Revealing the containment logic would require explaining a non-intuitive filter

The deeper problem is that "full containment" is counter-intuitive for "kings reigning in that period." A king who reigned 1180-1220 was clearly "reigning during 1184-1213" but the hidden code would exclude them!

## Fix Recommendation

**No code fix applied** - The task has multiple intrinsic formation errors:
1. Required reference data ("provided text") is not accessible via dialogue
2. Hidden implementation uses counter-intuitive containment semantics
3. Simulated user confirms overlap semantics that contradict hidden tests

The benchmark would need to either:
1. Add the king list to the task description
2. Change the filter to overlap semantics (more intuitive)
3. Clarify that only fully-contained reigns should be returned

Documenting as unfixable IFE.
