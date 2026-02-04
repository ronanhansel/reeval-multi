# ScienceAgentBench IFE Analysis Summary

## Tasks Analyzed: 39, 43, 44, 52, 55

All 5 tasks share a common root cause: **Sandbox Import Restrictions**

## Root Cause

The `hal_generalist_agent` uses a smolagents `CodeAgent` with a restricted `AUTHORIZED_IMPORTS` list. This list doesn't include specialized scientific packages required by ScienceAgentBench tasks:

```python
# From hal-harness/agents/hal_generalist_agent/main.py line 100-136
AUTHORIZED_IMPORTS = [
    "requests", "zipfile", "os", "pandas", "numpy.*", "sympy", "json",
    "bs4", "pubchempy", "xml", "yahoo_finance", "Bio", "sklearn",
    "scipy.*", "pydub", "io", "PIL", "chess", "PyPDF2", "pptx", "torch",
    "datetime", "fractions", "csv", "time", "pickle", "itertools",
    "random", "copy", "math", "cmath", "collections", "functools",
    "mpl_toolkits.mplot3d", "sympy"
]
```

**Missing packages for these tasks:**
- Task 39: `MDAnalysis`, `prolif` (protein-ligand interaction fingerprints)
- Task 43: `neurokit2`, `mne` (neuroimaging, EOG analysis)
- Task 44: `biopsykit` (sleep IMU data processing)
- Task 52: `deepchem`, `rdkit` (graph neural networks, cheminformatics)
- Task 55: `iris` (Scitools Iris for meteorological/oceanographic data)

## Impact

1. **During Agent Execution**: Agents cannot import required packages → cannot test/validate generated code
2. **During Docker Evaluation**: Packages are available via pipreqs, but code may be incorrect because agent couldn't test it

## Fixes Created

Each task has:
- `env_override.json` - Documents required packages and justification
- `README.md` - Detailed analysis and fix rationale

| Task | Required Packages | IFE Type |
|------|-------------------|----------|
| 39 | prolif, MDAnalysis, matplotlib | Sandbox import restriction |
| 43 | neurokit2, mne, matplotlib, scipy | Sandbox restriction + hidden dependency (mne) |
| 44 | biopsykit, pandas, numpy | Sandbox restriction (explicit API requirement) |
| 52 | deepchem, rdkit, matplotlib, dgl | Sandbox restriction |
| 55 | scitools-iris, matplotlib, cartopy, netCDF4 | Sandbox restriction |

## Solution Path

### Option 1: Modify AUTHORIZED_IMPORTS (Recommended)
Add these packages to `AUTHORIZED_IMPORTS` in `hal_generalist_agent/main.py`:

```python
AUTHORIZED_IMPORTS = [
    # ... existing imports ...
    # ScienceAgentBench requirements
    "MDAnalysis.*",
    "prolif.*",
    "neurokit2.*",
    "mne.*",
    "biopsykit.*",
    "deepchem.*",
    "rdkit.*",
    "iris.*",
    "cartopy.*",
    "netCDF4",
]
```

### Option 2: Create Per-Task Agent Configurations
Modify the fix runner to:
1. Read `env_override.json` for each task
2. Inject packages into `additional_authorized_imports` dynamically
3. This requires modifying how the agent is instantiated

### Option 3: Accept Sandbox Limitations
Document that agents must "code blind" for these packages - they can reference the packages in generated code but cannot test imports during development. This is the current behavior.

## Model Execution Results Summary

| Task | Valid Program | Success Rate | Issue |
|------|--------------|--------------|-------|
| 39 | 0 | 0 | MDAnalysis deprecation warnings |
| 43 | 0 | 0 | MNE module not found (neurokit2 dependency) |
| 44 | 0 | 0 | Empty log - code didn't produce output |
| 52 | 0 | 0 | TensorFlow/CUDA warnings (benign) |
| 55 | 0 | 0 | Empty CubeList error in Iris |

## Conclusion

These are genuine **Intrinsic Formation Errors** caused by the mismatch between:
1. Task requirements (use specific domain libraries)
2. Agent sandbox restrictions (those libraries are blocked)

The fix packages document the required changes. Applying them requires either:
- Expanding the global AUTHORIZED_IMPORTS (affects all benchmarks)
- Creating benchmark-specific import allowlists
- Modifying the fix runner to inject imports dynamically

## Files Created

```
fixes/scienceagentbench/
├── 39/
│   ├── env_override.json
│   └── README.md
├── 43/
│   ├── env_override.json
│   └── README.md
├── 44/
│   ├── env_override.json
│   └── README.md
├── 52/
│   ├── env_override.json
│   └── README.md
├── 55/
│   ├── env_override.json
│   └── README.md
└── ANALYSIS_SUMMARY.md
```
