# Task 43 Fix: NeuroKit2/MNE Environment Issue

## Root Cause Analysis

**IFE Type**: Execution Environment Issue - Missing Required Dependency (transitive)

### Problem Description
Task 43 requires analyzing Electrooculography (EOG) signals using NeuroKit2. The task involves:
1. Using `nk.eog_clean()` to filter EOG signals
2. Using `nk.eog_findpeaks()` to detect eye blink peaks
3. Saving visualization to PNG file

### Evidence of IFE

**Sandbox Import Restrictions** (from rubric evaluations):
- `InterpreterError: Import of neurokit2 is not allowed`
- `InterpreterError: Import of matplotlib.pyplot is not allowed`
- `InterpreterError: Import from scipy.signal is not allowed`
- `Forbidden access to module: PIL.Image`

**Docker Evaluation Failure** (from model execution log):
```
ModuleNotFoundError: No module named 'mne'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File ".../neurokit2/eog/eog_findpeaks.py", line 231, in _eog_findpeaks_mne
    import mne
ModuleNotFoundError: No module named 'mne'
```

### Root Cause
NeuroKit2's `nk.eog_findpeaks()` function calls `_eog_findpeaks_mne()` which has a **hidden dependency on the MNE package** (a neuroimaging library). The Docker evaluation's pipreqs only detected `neurokit2` but not `mne` because it's not a direct import in the generated code.

## Fix Applied

### Environment Override (`env_override.json`)
- **HAL_PIP_PACKAGES**: neurokit2 mne matplotlib scipy

The key fix is adding `mne` as an explicit dependency, which is a transitive requirement of `neurokit2.eog_findpeaks()`.

### Why This Fix is Fair (Not a Nerf)
1. **Preserves Scientific Rigor**: The task still requires understanding EOG signal processing, peak detection algorithms, and visualization
2. **No Hints Given**: We provide no solution logic - just ensure the required toolchain works
3. **Transitive Dependency**: MNE is a hidden requirement of NeuroKit2's EOG functions, not documented in the task
4. **Cross-Model Evidence**: All 4 models failed identically, first at sandbox import, then at MNE import
5. **Standard Toolchain**: NeuroKit2, MNE, matplotlib, scipy are standard neuroscience analysis libraries

## Expected Outcome After Fix
- Agents can generate code using NeuroKit2's EOG functions
- Docker evaluation will have neurokit2 AND its mne dependency available
- Task success depends on correctly applying nk.eog_clean and nk.eog_findpeaks

## Technical Notes
- MNE is a comprehensive neuroimaging library (EEG, MEG, EOG analysis)
- NeuroKit2 wraps MNE's functionality for easier biosignal analysis
- The task uses CSV data from `biosignals/eog_100hz.csv`
- Output should be saved to `pred_results/EOG_analyze_pred.png`
