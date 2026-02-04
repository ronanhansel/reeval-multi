# Task 146: Stack Overflow Question Popularity Scorer

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Arbitrary Keyword List Not Recoverable from Dialogue

### Problem Description
```
I want you to write a python function that predicts how popular a question will be
on Stack Overflow based on its title, tags, and content. Can you create a function
that takes these parameters and returns a score indicating how likely the question
is to receive answers?
The signature: def score_question_popularity(title, tags, content)
```

### Hidden Ground Truth
```python
def score_question_popularity(title, tags, content):
    attention_keywords = ['google', 'docs', 'api', 'stackoverflow']
    score = 0
    if any(keyword in title.lower() for keyword in attention_keywords):
        score += 1
    for tag in tags:
        if tag in attention_keywords:
            score += 1
    if any(keyword in content.lower() for keyword in attention_keywords):
        score += 1
    return score
```

### Evidence from Rubric Evaluations

1. **Evaluation 6 (Grade 1.0):** "The benchmark's hidden reference solution defines the acceptance criteria as: checking for presence of hard-coded keywords ['google', 'docs', 'api', 'stackoverflow']... This is a very specific, nonstandard definition that cannot be inferred from the task prompt."

### Root Cause

The hidden function uses a completely arbitrary keyword list:
- `['google', 'docs', 'api', 'stackoverflow']`
- Why these 4 keywords specifically? No justification in task.
- Why +1 for each (title/tags/content)? Not derivable.
- Why these exact keywords and not 'python', 'javascript', 'error', etc.?

The simulated user responds with generic guidance ("presence of certain keywords") but cannot reveal the exact 4-keyword list because it exists only as a code literal.

### Why This is NOT Fixable at Task Level

**Cannot be fixed without "nerfing"** - Revealing the keyword list would give away the implementation. Any keyword list revealed in instruction clarifications would be "making it easy" rather than "making it fair."

The fundamental issue is:
1. Task asks for "popularity prediction"
2. Hidden solution is just a keyword-count heuristic
3. The specific keywords are arbitrary and unguessable

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error. The benchmark would need to either:
1. Add the keyword list to the problem description
2. Accept any reasonable popularity scoring approach
3. Use a more principled scoring method (not arbitrary keywords)

Documenting as unfixable IFE.
