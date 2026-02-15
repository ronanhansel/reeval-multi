# Task 20 Fix - Tribal Leaders Statement Parser + Frontend Regex

## IFE Detected: YES

## Issue Description

Task 20 exhibits issues depending on whether processed as backend or frontend:

### Backend Task: Hidden Solution Contains Undiscoverable Fixed Vocabularies

**Task Description**: "statements about tribal leaders... identify leader, location, organization, sentiment"

**Hidden Reference Solution** (from evidence) hardcodes specific vocabularies:
```python
if word.lower() in ["Shakirullah", "Ajmal"]  # leaders
elif word.lower() in ["Pakistan", "Afghanistan", "East Timor", "Kashmir"]  # locations
elif word.lower() in ["Al Qaeda", "Taliban", "US", "Jews"]  # organizations
elif word.lower() in ["hate", "like"]  # sentiment
```

**Problems**:
1. These exact word lists are NOT provided in the task description
2. The simulated user cannot naturally provide these lists from code
3. User response: "I don't have specific example statements or expected outputs" and "I don't have explicit lists of leaders, organizations, or locations"

The agent implemented a reasonable heuristic approach (proper nouns, sentiment keywords), but the tests expect the exact hardcoded lists.

### Frontend Task: Parser Regex Mismatch

Evidence shows same parser issue:
> "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it."

## Root Cause

1. **Backend**: Test expectations require specific word lists that are undiscoverable through dialogue
2. **Frontend**: Parser configured for Python on HTML tasks

## Fix Required

### For Backend Task
Either:
1. Provide the expected word lists in the task description or allow simulated user to share them
2. OR change evaluation to accept any reasonable NLP-based approach (rather than exact word matching)

### For Frontend Task
Fix parser to accept HTML fenced blocks
