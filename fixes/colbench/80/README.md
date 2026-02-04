# Task 80 Fix Analysis

## Task Type
This task appears in both backend and frontend benchmarks:
- **Backend Task 80**: Categorize people based on physical descriptions/preferences
- **Frontend Task 80**: Art Gallery website design

## Issues Identified

### Backend Task 80: Simulated User Schema Inconsistency (IFE)

**Issue**: The simulated user provides internally inconsistent and schema-conflicting requirements about `people_data` and `preferences` structures.

**Evidence from rubric**:
- Hidden reference code: `for person, details in people_data.items():` and `for person, preference in preferences.items():` - both are **dicts keyed by person name**
- Simulated user tells agent: "Each person in `people_data` has keys like 'hair_color' and 'eye_color'" (implies records)
- Simulated user later: "Each person in `people_data` is identified by their name" / "return a dict mapping each person's name" (implies list of dicts with 'name' key)
- This conflict led agent to implement against wrong input shapes

**Root Cause**: The simulated user is supposed to convey hidden info accurately but provides contradictory descriptions of the data schema. This is an information extraction/communication failure.

**Verdict**: This IS an IFE. The agent follows the user's guidance but the guidance is wrong, leading to implementation mismatch with tests.

### Frontend Task 80: Parser/Format Mismatch (Same as Task 79)

**Issue**: Same Python-regex parsing requirement issue affecting all frontend tasks.

## Fix Decision

### Backend Task 80 - Potential Fix

The core issue is that the simulated user fails to accurately convey the data schema from the hidden code. However, this is a simulated user behavioral issue, not something fixable via instruction clarifications.

**Options**:
1. `instruction_override.json` - Add explicit schema documentation to the problem description
2. `simulated_user_override.json` - Provide additional prompt to help user convey schema correctly

I'll create an instruction clarification to make the expected input format explicit:

### Frontend Task 80 - No Fix

Same infrastructure-level parser issue as Task 79.
