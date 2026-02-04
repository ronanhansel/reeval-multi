# Task 419: Baseball Player Performance Stats - IFE Fix

## Issue Summary
The hidden reference solution relies on reading values from team_stats that the simulated user explicitly told the agent to ignore or approximate differently.

## Root Cause
**Intrinsic Formation Error**: Multiple specification mismatches:

1. **Function signature mismatch**: Problem description mentions "walks" as a parameter, but the signature doesn't include it. The hidden solution reads walks from `team_stats[player_name]['walks']`.

2. **Simulated user contradicts hidden implementation**:
   - User said: "assume walks=0" (but hidden uses `team_stats[player_name]['walks']`)
   - User said: "approximate SLG as hits/at_bats" (but hidden uses `team_stats[player_name]['slg_pct']`)

3. **Hidden formula differs from standard baseball formulas**:
   - Hidden OBP: `(hits + walks) / at_bats` (non-standard; standard OBP includes walks in denominator)
   - Hidden SLG: directly from team_stats, not computed from total bases

## Evidence
- Hidden info: `walks = team_stats[player_name]['walks']`, `slg_pct = team_stats[player_name]['slg_pct']`
- Simulated user: "assume 0" for walks, "approximate SLG as hits/at_bats"
- Test cases provide team_stats with walks and slg_pct values that the hidden solution uses

## Fix Applied
**instruction_override.json**: Clarify that walks, slg_pct, and career_avg should be read from team_stats, and specify the exact formulas expected.

## Why This Fix is Fair
- The task still requires understanding baseball statistics concepts
- The agent must implement the calculations and dictionary structure
- Only the data source and formulas are clarified (which were contradicted by simulated user)
