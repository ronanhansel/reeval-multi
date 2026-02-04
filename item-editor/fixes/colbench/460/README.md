# Task 460: Movie Rating Calculator - IFE Fix

## Issue Summary
The hidden reference solution uses a specific formula with arbitrary constants that the simulated user contradicted during dialogue.

## Root Cause
**Intrinsic Formation Error**: Formula specification mismatch:

1. **Hidden formula**: `(stars + (likes - dislikes)/(likes + dislikes) + len(comments) * 0.1) / 3`, rounded to 2 decimal places
2. **Simulated user guidance**: Claimed star scale is 1-5, requested output range 0-5, and approved a different formula with weights/log scaling

The specific constants (divide by 3, multiply comments count by 0.1) are arbitrary and cannot be derived from the task description.

## Evidence
- Hidden info: `rating = (stars + (likes - dislikes) / (likes + dislikes) + len(comments) * 0.1) / 3`
- Simulated user: "The star rating scale is 1-5, and I would like the final score to be in the range of 0-5"
- Simulated user approved: "Yes, this meets my needs. Please provide the Python function using this formula" (for a different formula)

## Fix Applied
**instruction_override.json**: Specify the exact formula expected.

## Why This Fix is Fair
- The task still requires understanding how to combine different factors into a rating
- The agent must implement the arithmetic correctly
- Only the arbitrary formula constants are clarified (which were contradicted by user)
