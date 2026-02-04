# Task 58 Fix Analysis

## Summary

**IFE Found**: YES - Docstring inconsistency and test case error

## Issue Details

### Issue 1: Step 58.2 Docstring Error
The function `eos_rho_from_press` has a docstring that incorrectly states the output:

**Current (incorrect)**:
```
Outputs:
eps: the specific internal energy corresponding to the given pressure, a float.
```

**Should be**:
```
Outputs:
rho: the density corresponding to the given pressure, a float.
```

The function name clearly indicates it computes density (`rho`) from pressure, but the docstring says it outputs energy (`eps`).

### Issue 2: Step 58.3 Test Cases Call Wrong Function
Step 58.3 is for `eos_eps_from_press` (compute specific internal energy from pressure), but ALL test cases call `eos_rho_from_press` instead:

**Current test case (incorrect)**:
```python
press = 10
eos_Gamma = 15
eos_kappa = 20
assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)
```

**Should be**:
```python
press = 10
eos_Gamma = 15
eos_kappa = 20
assert np.allclose(eos_eps_from_press(press, eos_Gamma, eos_kappa), target)
```

## Impact

- **Step 58.2**: Agents may implement incorrect logic (returning energy instead of density) due to misleading docstring
- **Step 58.3**: Even if agents implement `eos_eps_from_press` correctly, the test cases will not test it - they test the wrong function

## Recommended Fix

1. Update Step 58.2 docstring to say output is `rho` (density)
2. Update Step 58.3 test cases to call `eos_eps_from_press` instead of `eos_rho_from_press`

**Note**: This requires dataset modification which cannot be done via the harness override mechanism. The `instruction_override.json` can only clarify instructions, not fix test case function calls.

## Reported "IFEs" That Are NOT Actual Benchmark Defects

The rubric evaluations mention issues like:
- "python_interpreter tool forbids numpy/scipy imports"
- "NotImplementedError: Binary operation MatMult is not implemented"
- "Regex code parsing failures"

**These are NOT benchmark defects.** They are issues with specific agent scaffolding systems (e.g., smolagents' restricted python_interpreter). The actual HAL evaluation harness runs code in a full `python:3.11` Docker container with complete numpy/scipy access.
