# Task 2 Fix - Scipy Integration Function Deprecation

## Issue Identified

**IFE Type**: Dependency/API Mismatch

The task's `required_dependencies` explicitly specifies:
```python
import numpy as np
from scipy.integrate import simps
```

However, in SciPy 1.17.0+ (used in the evaluation Docker container), `scipy.integrate.simps` has been completely removed and replaced with `scipy.integrate.simpson`.

This means:
1. Any agent that correctly follows the required_dependencies will get an ImportError
2. Any agent using `simps` (following documentation, tutorials, or the specified import) will fail

## Evidence

From Docker container check:
```
$ docker run --rm python:3.11 sh -c "pip install scipy && python -c 'from scipy.integrate import simps'"
ImportError: cannot import name 'simps' from 'scipy.integrate'
```

SciPy version in evaluation container: 1.17.0 (latest)

Multiple rubric evaluations cite this exact error:
- Evaluation 2: `InterpreterError: Module scipy.integrate has no attribute simps`
- Evaluation 5: `InterpreterError: Module scipy.integrate has no attribute simps`
- Evaluation 6: `InterpreterError: Module scipy.integrate has no attribute simps`
- Evaluation 8: `InterpreterError: Import from scipy.integrate is not allowed` (agent sandbox) and `Module scipy.integrate has no attribute simps`

## Fix Applied

**evaluation_override.json**: Added a compatibility shim that aliases `scipy.integrate.simps` to `scipy.integrate.simpson` if simps doesn't exist. This is prepended to the agent's code before evaluation.

```python
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
```

## Rationale

1. **Preserves scientific rigor**: The fix does not simplify the physics or optics calculations. Agents must still correctly implement diffraction simulation.

2. **Fixes API mismatch only**: The shim provides backward compatibility for a renamed function with identical functionality (`simpson` is the direct replacement for `simps`).

3. **Fair to agents**: The benchmark explicitly requires `from scipy.integrate import simps` in its dependencies. This import doesn't work in modern scipy. The fix makes the specified dependency actually importable.

## Note on Agent Sandbox Issues

Some rubric evaluations mention import restrictions in agent sandboxes (python_interpreter blocking scipy). This is an agent framework issue, not a SciCode benchmark issue. The evaluation harness properly installs scipy.

## Verdict

This is a genuine IFE caused by the benchmark explicitly requiring a deprecated/removed function. The fix restores the ability to use the specified dependency without reducing scientific challenge.
