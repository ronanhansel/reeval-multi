# Task 63: NeuroKit2 Bio Signal Analysis Fix

## Root Cause Analysis

### Observed Failure
- **valid_program**: 0 (code failed to execute)
- **success_rate**: 0
- **Error**: `TypeError: ecg_plot() got an unexpected keyword argument 'sampling_rate'`

### Failure Reason
The agent used an **outdated API signature** for `nk.ecg_plot()`.

The agent called:
```python
nk.ecg_plot(ecg_signals, sampling_rate=100, show=False)
```

But the current NeuroKit2 (0.2.x) API is:
```python
nk.ecg_plot(ecg_signals, info=None)
```

The `sampling_rate` parameter was removed in newer versions of NeuroKit2. The older API (0.0.x) had:
```python
ecg_plot(ecg_signals, rpeaks=None, sampling_rate=None, show_type="default")
```

### Is This an IFE?

**Yes** - This is an API documentation/version mismatch:
1. The domain knowledge doesn't specify the correct API signature
2. The Docker container uses a different NeuroKit2 version than what documentation suggests
3. All models would encounter this issue if they rely on outdated documentation

### Fix Type: Instruction Clarification + Environment Check

Add clarification about the current NeuroKit2 API and ensure proper `info` dict is passed.

## Why This Preserves Task Difficulty

- The scientific task remains unchanged (process and visualize ECG/RSP signals)
- The fix only provides accurate API documentation, not implementation hints
- The agent still needs to correctly call ecg_process/rsp_process and save visualizations

## Expected Outcome

After clarification, agents will use the correct API signature:
```python
signals, info = nk.ecg_process(ecg_signal, sampling_rate=100)
nk.ecg_plot(signals, info)  # No sampling_rate argument
```
