# Task 46 - VMC Helium (Metropolis Sampling)

## IFE Analysis: NO FIX NEEDED

### Claimed IFEs from Rubric Evaluations

The rubric evaluations report errors related to:
- "Import of numpy is not allowed"
- "Forbidden access to module: numpy.random"
- "Forbidden access to module: numpy.linalg"

### Why These Are NOT Intrinsic Formation Errors

1. **Agent Tool Environment ≠ Evaluation Environment**: These errors occur in the agent's development/testing tools (like `python_interpreter` in smolagents), which have restricted import allowlists. These are scaffolding constraints, NOT benchmark constraints.

2. **Actual Evaluation Environment**: The HAL harness runs evaluation in a Docker container (`python:3.11`) with full NumPy available:
   - `numpy` is installed
   - `numpy.random` is available for Metropolis sampling
   - `numpy.linalg` is available for norm calculations

3. **Benchmark Dependencies Are Clear**: The task correctly specifies:
   ```python
   import numpy as np
   ```
   This dependency is available in the evaluation Docker environment.

### Task Scientific Validity

The task implements a standard Variational Monte Carlo (VMC) calculation for the helium atom:
- **Step 1**: Slater determinant wave function ψ = exp(-αr₁)exp(-αr₂)
- **Step 2**: Hamiltonian with electron-electron and electron-ion potentials
- **Step 3**: Metropolis sampling algorithm
- **Step 4**: Energy calculation with kinetic, potential_ei, and potential_ee components

All physics and algorithms are well-defined and standard in computational quantum chemistry.

### Root Cause of Agent Failures

- Agents using restricted interpreter tools cannot test NumPy-based code during development
- Some agents produce formatting errors (malformed `final_answer` calls)
- Code-fence parsing issues in agent scaffolding
- These are capability/scaffolding issues, not benchmark formation errors

### Conclusion

The benchmark is correctly formed. The scientific problem (VMC for helium) is well-specified and solvable with NumPy in the actual evaluation environment.
