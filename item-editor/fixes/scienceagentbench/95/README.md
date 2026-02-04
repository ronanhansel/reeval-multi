# Task 95 Fix: DeepChem ScScore Synthetic Complexity Model

## Root Cause Analysis

**IFE Type**: Sandbox Import Restriction + pipreqs Detection Failure

### Problem Description
Task requires using DeepChem's ScScoreModel for predicting synthetic complexity scores using ECFP fingerprints.

### Why This Was Failing

1. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `deepchem` and `pickle` imports during development
2. **Cannot Load Data**: Without `pickle`, agents cannot even conceptualize loading the `.pkl` files
3. **pipreqs Detection Failure**: Agent code without proper imports → pipreqs doesn't detect deepchem → DGL not installed

### Evidence from Verdict
"importing deepchem fails immediately with ModuleNotFoundError: no module named 'tensorflow'"

**CORRECTION**: The Docker base image has **TensorFlow 2.17** pre-installed. The real issue was pipreqs not detecting deepchem because agents couldn't write proper import statements.

## Fix Applied

### 1. Instruction Override
Added critical clarifications:
- **MUST include proper imports** at the top of code
- Explicit API usage for ScScoreModel and CircularFingerprint
- How to load pickle files

### 2. Critical Imports for pipreqs Detection
```python
import deepchem as dc
from deepchem.models import ScScoreModel
from deepchem.feat import CircularFingerprint
import pickle
import numpy as np
```

When pipreqs detects `deepchem` in imports, the Docker harness automatically adds DGL.

## Why This Fix is Fair

- Agent still must understand ScScore model architecture
- No hints about training strategy or hyperparameters
- Only compensates for sandbox limitation preventing normal development

## Expected Outcome After Fix

- Agents write code with proper deepchem imports
- pipreqs detects deepchem → DGL installed
- Code executes properly in evaluation container
