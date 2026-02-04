# Task 27 Fix Analysis

## Task Overview
- **Backend (ID 27)**: Sort game reviews by rating - "highest rated to lowest"
- **Frontend (ID 27)**: Travel Agency Website

## Backend Analysis

### IFE Detected: YES - Specification vs Hidden Code Mismatch

**Problem Description says:**
> "I want to see the reviews in order from highest rated to lowest"

**Hidden Implementation:**
```python
def sort_reviews(reviews):
    # A list of reviews in the format: (game_name, review_text, rating)
    return sorted(reviews, key=lambda x: x[2])
```

**Issue:** The hidden code sorts **ascending** (lowest to highest) but the task description explicitly asks for **descending** (highest to lowest). An agent following the natural language specification would implement `reverse=True`, which would fail tests expecting ascending order.

This is an **Intrinsic Formation Error** because:
1. The task description explicitly contradicts the hidden implementation
2. An agent correctly understanding the user's stated needs would produce wrong output
3. This is not a capability issue - it's a benchmark data error

### Fix Type: evaluation_override.json

The fix should accept both ascending and descending implementations since the task description asks for descending but tests expect ascending. The ambiguity should not penalize agents who follow the stated requirements.

## Frontend Analysis

### IFE Detected: YES - Harness Parser Mismatch

The frontend task uses an evaluation harness that expects Python fenced code blocks (`(?:py|python)?\s*\n(.*?)\n`) but the task requires raw HTML output wrapped in `<html>` tags.

**Evidence from rubric evaluations:**
> "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\s*\n(.*?)\n``` was not found in it."

This is a **benchmark infrastructure issue** affecting all frontend tasks, not specific to task 27.

### Fix: This is a systemic issue that should be fixed in the harness, not per-task.
