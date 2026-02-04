# Task 545 Fix: Football Player Stats by Facial Hair

## IFE Summary

**Task**: Get player stats filtered by facial hair style.

**Problem Description**: "Can you write a function that can find out how well each footballer did given their mustache or beard style during their match?"

**Hidden Reference Implementation**:
```python
def get_player_stats(team_roster, facial_hair):
  stats = {'players': []}
  for player in team_roster:
    if player['name'] in facial_hair:
      player_stats = {'name': player['name'], 'games_played': player['games_played']}
      stats['players'].append(player_stats)
  return stats
```

## Intrinsic Formation Error

**Type**: Specification-Implementation Mismatch / Misleading Parameter Semantics

**Issue**: The hidden reference interprets `facial_hair` as a **collection of player names** (checking `if player['name'] in facial_hair`), while the prompt and parameter name clearly suggest it should contain facial hair styles (mustache, beard, etc.).

**Why this is an IFE (not capability issue)**:
1. The prompt explicitly mentions "mustache or beard style" as the filtering criteria
2. The parameter is named `facial_hair`, suggesting it contains facial hair data
3. The hidden implementation treats it as a name filter - semantically inconsistent with the prompt
4. The output format `{'players': [{'name', 'games_played'}]}` is also not specified
5. An agent cannot reasonably guess that `facial_hair` means "player names to include"

**Evidence from evaluations**:
- User cannot explain what `facial_hair` should contain
- User fabricates additional fields (goals, assists, etc.) that aren't in hidden reference
- Hidden behavior is fundamentally different from what prompt describes

## Fix Type

**Instruction Clarification** - Reveal that `facial_hair` is actually a collection of player names to filter by, and specify the output format.

## Fix Rationale

The fix reveals the actual semantics of the `facial_hair` parameter and the expected output format. This addresses the misleading parameter name and prompt while maintaining the implementation challenge.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the filtering logic, handle the data structures, and produce working code. We're only correcting the misleading specification.
