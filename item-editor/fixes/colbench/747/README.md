# Task 747 Fix: Counseling Recommendations

## IFE Summary

**Task**: Provide counseling recommendations based on problem description and budget.

**Problem Description**: "I need a python function that can provide counseling recommendations based on a person's problem and budget."

**Hidden Reference Implementation**:
```python
def get_counseling_recommendations(problem_description, budget):
    if 'relationship' in problem_description and 'low' in budget:
        return 'Contact Al-Anon and get some counseling through local hospitals or United Way.'
    elif 'general' in problem_description and 'medium' in budget:
        return 'Seek counseling through a university psychology department.'
    elif 'specific' in problem_description and 'high' in budget:
        return 'Get private counseling.'
    else:
        return 'Please provide more information.'
```

## Intrinsic Formation Error

**Type**: Arbitrary Hardcoded Keywords and Responses

**Issue**: The implementation uses exact substring matching with specific hardcoded keywords and canned response strings that cannot be derived from the prompt:

| Condition | Response |
|-----------|----------|
| 'relationship' in problem AND 'low' in budget | "Contact Al-Anon..." |
| 'general' in problem AND 'medium' in budget | "Seek counseling through a university..." |
| 'specific' in problem AND 'high' in budget | "Get private counseling." |
| else | "Please provide more information." |

**Why this is an IFE (not capability issue)**:
1. The trigger keywords (relationship, general, specific) are arbitrary
2. The budget keywords (low, medium, high) need substring matching
3. The exact response strings are arbitrary canned text
4. The return type (single string, not list) is unspecified
5. An agent implementing reasonable recommendation logic would fail

## Fix Type

**Instruction Clarification** - Specify the exact keyword-response mapping.

## Fix Rationale

The fix reveals the specific conditional logic required. While this reveals significant implementation detail, it's necessary because the exact string responses are arbitrary and cannot be discovered through dialogue.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the substring checking logic correctly. We're revealing arbitrary canned responses that couldn't be guessed.
