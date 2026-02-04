# Task 8 Fix - Restaurant Website Parser Issue

## IFE Detected: YES

## Issue Description

Task 8 (Restaurant website with full-screen image carousel) exhibits the **parser regex mismatch** that affects multiple frontend tasks:

### Parser Regex Mismatch

Evidence from evaluations:
> "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it."

The agent provided valid HTML:
```html
<!DOCTYPE html>
<html lang="en">...
```

But this was rejected because:
1. The task requires: "Write the code inside a tag `<html>`"
2. The parser expects: Python code fences only

### Additional Backend Task Evidence

One evaluation also shows a **tool name mismatch**:
> "InterpreterError: Forbidden function evaluation: 'finish_answer' is not among the explicitly allowed tools"

Task prompt instructs agent to use `finish_answer` but only `final_answer` is available.

### Simulated User Issues

The simulated user also provides incomplete/unhelpful feedback:
- Returns generic image comparison text instead of answering specific design questions
- Cannot provide actionable feedback to help agent converge on design

## Root Cause

1. **Infrastructure Issue**: Parser configured for Python on HTML tasks
2. **Tool Name Inconsistency**: Different tool names in prompt vs available tools
3. **Simulated User Limitation**: Cannot reliably extract and communicate visual requirements

## Fix Required

1. **Parser Fix**: Accept HTML fenced blocks for frontend tasks
2. **Tool Name Fix**: Ensure task prompt uses correct tool name (`final_answer` not `finish_answer`)
