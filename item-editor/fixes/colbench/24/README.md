# Task 24 Fix - Ad Revenue Calculation + Frontend Simulated User Issues

## IFE Detected: YES

## Issue Description

Task 24 exhibits issues depending on whether processed as backend or frontend:

### Backend Task: Hidden Solution Information Gap

**Hidden Reference Solution** (from evidence) uses a complex formula:
```python
revenue = num_ads * ad_type['revenue_per_ad'] * clicks_per_ad * ad_duration * click_revenue * ad_display_probability
```

**Simulated User Responses**:
> "I don't know how 'ad_duration' influences revenue."
> "I don't know if 'click_revenue' is redundant"

The user cannot explain how all these parameters interact because the hidden info is code. The agent cannot reliably determine which parameters multiply vs. which might be redundant without this information.

### Frontend Task: Simulated User Broken + Parser Issues

Multiple critical issues:

1. **Simulated User Returns Meta-Instructions**:
   Instead of answering questions, the user returns:
   > "Your task is to simulate a human user... The agent did not provide any visualization."

   This is the benchmark's system prompt being echoed back, not actual user feedback.

2. **User Refuses to Answer**:
   > "I'm sorry, I can't provide a description of the image."
   > "I'm sorry, I can't provide a description of the agent's image."

   Agent asked basic questions (business name, CTA text) but got refusals.

3. **Parser Regex Mismatch**:
   > "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it."

## Root Cause

1. **Backend**: Hidden formula parameters cannot be fully explained through dialogue
2. **Frontend**:
   - Simulated user tool/routing is broken (returns system prompts instead of answers)
   - Parser rejects valid HTML
   - User refuses to provide basic task requirements

## Fix Required

### For Backend Task
Either:
1. Document the formula parameters and their relationships in the task
2. OR modify simulated user to be able to explain parameter interactions

### For Frontend Task
1. Fix ask_user tool routing (should not echo system prompts)
2. Fix simulated user to answer clarifying questions
3. Fix parser to accept HTML
