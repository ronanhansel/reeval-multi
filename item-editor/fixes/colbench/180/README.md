# Task 180: IMAP Security Check Function

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Semantically Incorrect/Contradictory Requirement

### Problem Description
```
I want the agent to write a python function that determines if a login process to an
IMAP server is secure, taking into account the port used and whether mandatory SSL is
enabled. I am worried about the security of my email credentials when using certain
devices and I would like to verify if my email setups are secure.
The signature: def is_imap_secure(imap_server, username, password, port=993, ssl=True)
```

### Hidden Ground Truth
```python
def is_imap_secure(imap_server, username, password, port=993, ssl=True):
    return port == 993 and ssl == True and imap_server.startswith('smtps://')
```

### Evidence from Rubric Evaluations

1. **Evaluation 1 (Grade 0.0):** "A benchmark formation defect exists because the hidden expected solution requires an additional, arbitrary condition unrelated to the stated task: `imap_server.startswith('smtps://')`. This is both (a) semantically inconsistent with IMAP (it references SMTPS) and (b) not derivable from the natural-language specification"

2. **Evaluation 5 (Grade 1.0):** "The hidden reference solution encodes an arbitrary and semantically incorrect requirement: it deems IMAP secure only if the server string starts with 'smtps://'"

### Root Cause - Semantic Error in Hidden Code

**The hidden code is WRONG:**
- Task is about **IMAP** security
- Hidden code requires `imap_server.startswith('smtps://')`
- **SMTPS is SMTP-over-SSL, NOT IMAP!**
- IMAP URLs should start with `imaps://` (for IMAP-over-SSL)

This is a factual error in the benchmark's ground truth:
- SMTP = Simple Mail Transfer Protocol (sending email)
- IMAP = Internet Message Access Protocol (reading email)
- SMTPS = SMTP over SSL (port 465)
- IMAPS = IMAP over SSL (port 993)

The task asks about IMAP but the hidden code checks for SMTPS protocol prefix!

### Why This is NOT Fixable at Task Level

**Cannot be fixed** because the ground truth code contains a factual error:
- If we clarify that IMAP servers use `imaps://` prefix, agents will fail tests
- If we clarify to use `smtps://` prefix, we're propagating incorrect information
- The benchmark's test cases likely use `smtps://` URLs for IMAP servers

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error (incorrect ground truth). The benchmark would need to:
1. Fix the hidden code to check for `imaps://` instead of `smtps://`
2. Update all test cases to use correct IMAP URLs
3. Or accept both `imaps://` and check for just `port==993 and ssl==True`

Documenting as unfixable IFE (ground truth contains semantic error).
