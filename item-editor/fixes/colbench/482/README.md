# Task 482: Phrase Sound Similarity - IFE Fix

## Issue Summary
The task description asks for phonetic similarity ("sounds similar") but the hidden solution implements simple positional character matching, which is completely different.

## Root Cause
**Intrinsic Formation Error**: Fundamental specification/implementation mismatch:

1. **Task description**: "determine if a phrase sounds similar to another" with example "'do want to' almost sounds like '212'" (phonetic wordplay)
2. **Hidden implementation**:
   - Remove spaces, lowercase
   - Zip characters positionally
   - Count exact matches
   - Divide by target length
   - Compare to tolerance

This is NOT phonetic similarity - it's positional character overlap. Any reasonable interpretation of "sounds similar" would involve phonetic encoding (soundex, metaphone, etc.) or pronunciation comparison.

## Evidence
- Task description: "sounds similar", "For example 'do want to' almost sounds like '212'"
- Hidden info: `for char1, char2 in zip(input_phrase, target_phrase): if char1 == char2: similar_score += 1`
- These test completely different things: phonetics vs character positions

## Fix Applied
**instruction_override.json**: Clarify that the actual algorithm is positional character matching, not phonetic similarity.

## Why This Fix is Fair
- The task still requires string manipulation and comparison logic
- The agent must implement the algorithm correctly
- Only the misleading "sounds similar" framing is corrected to match actual expected behavior
