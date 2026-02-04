# Task 22 Fix - Wizard of Oz Character Color Mapping + Frontend Regex

## IFE Detected: YES

## Issue Description

Task 22 exhibits issues depending on whether processed as backend or frontend:

### Backend Task: Hidden Solution Contains Specific Undiscoverable Mappings

**Task Description**: "different characters from The Wizard of Oz" without specifying which characters must be supported

**Hidden Reference Solution** expects specific character names and exact color values:
```python
elif character_name == 'Emerald City guard': return 'green'
elif character_name == 'Munchkin': return 'colorful'
elif character_name == 'Glinda, the Good Witch of the North': return 'light'
elif character_name == 'Tin Man': return 'metal silver'
```

**Problems**:
1. Task doesn't specify which characters must be included
2. Exact string keys differ (e.g., "Glinda, the Good Witch of the North" vs "glinda")
3. Exact values differ (e.g., "metal silver" vs "silver")
4. Non-obvious characters included (e.g., "Emerald City guard", "Munchkin")

**Simulated User Response**:
> "I don't have a complete list... Can you help me with the rest?"

The user cannot provide the complete mapping, leaving the agent to guess.

### Frontend Task: Parser Regex Mismatch + Simulated User Issues

1. **Parser Issue**:
   > "Error in code parsing: Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it."

2. **Simulated User Hallucination** (for frontend variant):
   - Claims agent provided no visualization when it did
   - Provides comparison feedback based on non-existent images
   - Gives contradictory requirements that conflict with task spec

## Root Cause

1. **Backend**: Exact character-color mappings are undiscoverable through dialogue
2. **Frontend**: Parser rejects HTML; simulated user behavior is broken

## Fix Required

### For Backend Task
Either:
1. Provide complete character-color mapping in task description
2. OR accept reasonable Wizard of Oz character-color associations (not exact string matching)

### For Frontend Task
1. Fix parser to accept HTML
2. Fix simulated user to properly render and compare images
