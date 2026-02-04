# Task 338: Bowl Eligibility - IFE Fix

## Problem Summary
The hidden implementation requires THREE conditions for eligibility:
```python
if team_conf in conf_teams and team_wins >= min_wins_required and prev_bowl_appearances >= min_conf_teams_required:
    return True
```

But according to the rubric evaluation, the simulated user:
1. Said "I don't know the specific eligibility rules"
2. Then explicitly confirmed to the agent: "let's ignore prev_bowl_appearances and min_conf_teams_required for now"
3. Confirmed: "Eligibility will be: team_wins >= min_wins_required AND team_conf in conf_teams"

This creates a direct contradiction where following the user's confirmed specification leads to failing the hidden tests.

## IFE Type
**Simulated User Response Issue / Contradictory Confirmation**

The simulated user explicitly confirmed a simplified spec that contradicts the hidden expected behavior.

## Fix Approach
Clarify that ALL three conditions must be checked.

## Evidence
- Hidden: requires `prev_bowl_appearances >= min_conf_teams_required` in addition to the other two conditions
- Simulated user: "let's ignore prev_bowl_appearances" (confirmed by agent, agreed by user)
- This is a clear case where following the user leads to failure
