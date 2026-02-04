# Task 467: ATM Legitimacy Checker - IFE Fix

## Issue Summary
The hidden reference solution uses a simple blacklist check against two hardcoded tuples, which are completely arbitrary and cannot be discovered through dialogue.

## Root Cause
**Intrinsic Formation Error**: Arbitrary undiscoverable blacklist:

1. **Hidden logic**: Return True unless the 5-tuple matches one of two known skimmer tuples:
   - `(123456, 480, 800, 'blue', True)`
   - `(789012, 640, 400, 'red', True)`
2. **Simulated user**: "I don't know" when asked about valid serial numbers, rules, or pinpad colors
3. These magic tuples are completely arbitrary - there's no way to derive or discover them

## Evidence
- Hidden info: `known_skimmers = [(123456, 480, 800, 'blue', True), (789012, 640, 400, 'red', True)]`
- Hidden info: `return (atm_serial_number, ...) not in known_skimmers`
- Simulated user: "I don't have a list of valid ATM serial numbers", "I don't know which ATM pinpad colors should be considered legitimate"

## Fix Applied
**instruction_override.json**: Specify the blacklist tuples that define non-legitimate ATMs.

## Why This Fix is Fair
- The task still requires implementing tuple comparison and blacklist checking
- The agent must correctly structure the comparison logic
- Only the arbitrary, undiscoverable blacklist data is clarified
