# Task 12 Fix - Scipy Integration Function Deprecation

## Issue Identified

**IFE Type**: Dependency/API Mismatch

The rubric evaluations identify an intrinsic formation error:

**scipy.integrate.simps deprecation**: The task's step 12.3 instructs agents to "Normalize the result using Simpson's rule", and the required_dependencies specify `from scipy import integrate`. In SciPy 1.14+, `scipy.integrate.simps` was deprecated, and in SciPy 1.17.0+ (used in the evaluation Docker container), it has been completely removed and replaced with `scipy.integrate.simpson`.

This causes any solution that uses `integrate.simps` (following older documentation, tutorials, or the function name that was standard for years) to fail with an ImportError or AttributeError.

## Evidence

From Docker container check:
```
$ docker run --rm python:3.11 sh -c "pip install scipy && python -c 'from scipy.integrate import simps'"
ImportError: cannot import name 'simps' from 'scipy.integrate'
```

SciPy version in evaluation container: 1.17.0 (latest)

Multiple rubric evaluations cite this issue:
- Evaluation 2: "InterpreterError: Module scipy.integrate has no attribute simps"
- Evaluation 6: "Object <module 'scipy.integrate' ...> has no attribute simps"

## Fix Applied

**evaluation_override.json**: Added a compatibility shim that aliases `scipy.integrate.simps` to `scipy.integrate.simpson` if simps doesn't exist. This is prepended to the agent's code before evaluation.

```python
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
```

## Rationale

1. **Preserves scientific rigor**: The fix does not simplify the physics or mathematics. Agents must still implement the Numerov method and SCF correctly.

2. **Fixes API mismatch only**: The shim simply provides backward compatibility for a renamed function with identical functionality.

3. **Fair to agents**: Agents who learned from older documentation (pre-2024) would naturally use `simps`. The function was only fully removed in SciPy 1.17 (late 2024).

## Note on Agent Sandbox Issues

The rubric also mentions agent sandbox restrictions (python_interpreter blocking numpy/scipy). This is an agent framework configuration issue, not a SciCode benchmark issue. The evaluation harness properly installs and uses scipy.

## Verdict

This is a genuine IFE caused by API deprecation. The fix restores backward compatibility without reducing scientific challenge.
