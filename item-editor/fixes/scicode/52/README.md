# Task 52 - Schroedinger Shooting Method

## IFE Analysis: FIX REQUIRED

### Identified Intrinsic Formation Error

**Issue**: Step 52.2 instructs to "normalize the result using Simpson's rule for numerical integration" but the function `scipy.integrate.simps` has been **removed** in SciPy 1.12+ (deprecated since SciPy 1.10).

The current evaluation environment uses **SciPy 1.17.0** where:
- `scipy.integrate.simps` → **REMOVED** (raises ImportError)
- `scipy.integrate.simpson` → Available (replacement function)
- `scipy.integrate.trapz` → **REMOVED** (raises ImportError)
- `scipy.integrate.trapezoid` → Available (replacement function)

### Evidence

From the step description:
> "After integration, normalize the result using Simpson's rule for numerical integration."

Agents following this instruction may reasonably attempt:
```python
from scipy.integrate import simps
# or
from scipy import integrate
norm = integrate.simps(ur**2, R)
```

This will fail with:
```
ImportError: cannot import name 'simps' from 'scipy.integrate'
```

### Fix Applied

**instruction_override.json**: Updated step 52.2 description to clarify that `scipy.integrate.simpson` (with an 'on' suffix, not 'simps') should be used for Simpson's rule integration, matching the current SciPy API.

The fix:
1. Clarifies the correct function name: `scipy.integrate.simpson`
2. Does NOT simplify the scientific problem
3. Does NOT provide implementation hints
4. Only corrects the outdated API reference

### Scientific Validity Preserved

The task still requires:
- Solving the radial Schroedinger equation via ODE integration
- Proper normalization of the wavefunction
- Understanding of Simpson's rule for numerical integration

The fix only addresses the API naming change, not the underlying scientific concepts.
