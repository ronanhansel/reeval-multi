# Task 166: Camera Shake Minimization Function

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Return Type Mismatch + Arbitrary Magic String Requirement

### Problem Description
```
I'm experiencing camera shake in my panoramic HDR photography setup using a mirror ball
mounted on a tripod. I'm using mirror lock-up and a 2-second delay, but I'm still getting
blurry images. I've checked that the tripod is sturdy and I'm not touching the camera or
tripod during exposure. Can you help me identify the cause and propose a solution?
Can you write a python function that suggests the optimal approach to eliminate camera shake?
The signature: def minimize_camera_shake(use_mirror_lockup, use_delay, turn_off_is, tripod_type)
```

### Hidden Ground Truth
```python
def minimize_camera_shake(use_mirror_lockup, use_delay, turn_off_is, tripod_type):
    if use_mirror_lockup and use_delay and turn_off_is and tripod_type == 'sturdy':
        return True
    else:
        return False
```

### Evidence from Rubric Evaluations

1. **Evaluation 2 (Grade 1.0):** "The hidden specification is a simple boolean predicate: return True iff use_mirror_lockup, use_delay, turn_off_is are all True and tripod_type == 'sturdy'... the simulated user is constrained to only answer from hidden code; when asked about camera settings, the user necessarily replied 'I don't know'"

2. **Evaluation 5 (Grade 1.0):** "Hidden info defines: `tripod_type == 'sturdy'`, which introduces an arbitrary, undiscoverable requirement... This is not inferable from the prompt and is inconsistent with the task's stated goal ['identify the cause and propose a solution']"

3. **Evaluation 6 (Grade 1.0):** "A better agent following the prompt would likely return explanatory advice (string/dict) and accept more tripod_type categories (e.g., 'medium') rather than the exact magic string 'sturdy'"

### Root Cause - Multiple Issues:

**Issue 1: Task/Return Type Mismatch**
- Task says: "help me identify the cause and **propose a solution**"
- Task says: "write a function that **suggests** the optimal approach"
- Natural interpretation: return advice/recommendations (string/dict)
- Hidden solution: returns a boolean (True/False)

**Issue 2: Arbitrary Magic String**
- Hidden code requires `tripod_type == 'sturdy'` exactly
- Why 'sturdy' and not 'heavy', 'solid', 'stable', 'professional'?
- This is an unguessable sentinel value

**Issue 3: Simulated User Cannot Help**
- User says "I don't know the camera make/model or environmental conditions"
- User says "I'm unsure about the stabilization settings"
- None of the required boolean/string values are extractable

### Why This is NOT Fixable at Task Level

**Cannot be fixed without "nerfing"** because:
1. Revealing that return type is boolean contradicts "suggest/propose" language
2. Revealing 'sturdy' as the magic tripod_type gives away the implementation
3. The task description implies advisory function, not boolean gate

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error. The benchmark would need to either:
1. Change task description to say "returns True if settings are optimal"
2. Specify valid tripod_type values in problem description
3. Accept advisory-style responses (not just boolean)

Documenting as unfixable IFE.
