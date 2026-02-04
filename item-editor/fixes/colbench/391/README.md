# Task 391: Greenhouse Gas Emissions Calculator - IFE Fix

## Issue Summary
The task requires calculating total greenhouse gas emissions with specific emission rates (20, 150, 350, 50, 100 for electricity, industry, transportation, buildings, oil_and_gas respectively), but these arbitrary constants are not discoverable through the collaborative dialogue.

## Root Cause
**Intrinsic Formation Error**: The hidden information contains specific numeric constants that:
1. Are not mentioned in the problem description
2. Cannot be derived from domain knowledge (they are arbitrary, not based on real-world emission factors)
3. The simulated user is instructed to only answer with hidden info but claims to "not know the units" and suggests using "widely cited average values"

This creates an impossible situation: the agent must guess 5 specific magic numbers (20, 150, 350, 50, 100) with no way to discover them.

## Evidence
- Hidden info: `electricity_rate = 20`, `industry_rate = 150`, `transportation_rate = 350`, `buildings_rate = 50`, `oil_and_gas_rate = 100`
- Simulated user response: "I don't know the units for each parameter" and "you can use widely cited average values"
- Test cases expect exact values like `calculate_emissions(1, 0, 0, 0, 0) == 20`

## Fix Applied
**instruction_override.json**: Add the emission rate constants to the problem description, making the task solvable while preserving the coding challenge.

## Why This Fix is Fair
- The task is still a coding task (implementing weighted sum calculation)
- The arbitrary constants become specified requirements rather than undiscoverable magic numbers
- The collaborative nature is preserved (agent can still ask about formula structure, validation, etc.)
