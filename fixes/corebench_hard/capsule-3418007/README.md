# Fix for capsule-3418007

## Problem Diagnosis

The task failed for all models due to two environmental barriers:

### 1. Unclear Working Directory Requirement
The task instruction says "Run 'main.py'" but doesn't specify:
- The script is located at `code/main.py`, not in the repository root
- The script must be run from the `code/` directory

The code in `performance_measurement.py` uses relative paths like:
```python
time_features = ast.literal_eval(open("../data/features/time_features.txt").read())
```

These paths only resolve correctly when running from the `code/` subdirectory. When agents tried to run from the root `environment/` directory, they got `FileNotFoundError` for the feature files.

The data files ARE present in the capsule - the rubric's claim about "missing data files" was incorrect. The actual issue was the working directory.

### 2. Outdated Dependency Version Pins
The original `requirements.txt` pins very old versions:
- `numpy==1.19.5`
- `pandas==1.2.5`
- `xgboost==1.3.3`

These versions don't install on modern Python (3.10+).

## Fix Applied

**Type: Input/Prompt Clarification** (input_override.json)

Added clarifications to the task instructions:
1. Explicitly state that `main.py` is at `code/main.py`
2. Specify that the script must be run from the `code/` directory due to relative path dependencies
3. Advise installing modern versions of dependencies without the strict version pins
4. Mention using `MPLBACKEND=Agg` for headless plotting

## Why This is NOT Nerfing

This fix does NOT:
- Give hints about the answer (F1 score or AUC values)
- Pre-compute results
- Reduce computational requirements (still requires training XGBoost models 10 iterations)
- Simplify the task

The core challenge remains: the agent must:
- Install appropriate dependencies
- Navigate the codebase structure
- Run the ML pipeline
- Parse the output to extract the F1 score for "statistical general only" and AUC from ROC curves
- Format the answer as a Python dictionary

The fix only clarifies environmental setup that was ambiguous in the original instructions.
