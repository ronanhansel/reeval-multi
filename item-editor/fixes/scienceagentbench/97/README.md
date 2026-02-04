# Task 97 Fix: DeepChem CGCNN for Formation Energy Prediction

## Root Cause Analysis

**IFE Type**: Sandbox Import Restriction + pipreqs Detection Failure

### Problem Description
Task requires training a DeepChem CGCNN (Crystal Graph Convolutional Neural Network) model for formation energy prediction on perovskite structures.

### Why This Was Failing

1. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `deepchem` and `pickle` imports
2. **pipreqs Detection Failure**: Agent code without proper imports → pipreqs doesn't detect deepchem → DGL not installed
3. **CGCNN Requires DGL**: The CGCNN model specifically requires DGL (Deep Graph Library) for graph operations

### Evidence from Verdict
"importing deepchem / CGCNN fails with ModuleNotFoundError: no tensorflow"

**CORRECTION**: The Docker base image has **TensorFlow 2.17** pre-installed. The issue was the DGL post-install hook not triggering because pipreqs didn't detect deepchem imports.

## Fix Applied

### 1. Instruction Override
Added critical clarifications:
- **MUST include proper imports** for deepchem and CGCNN
- How to load pickled NumpyDataset files
- API for CGCNNModel and CGCNNFeaturizer

### 2. Critical Imports for pipreqs Detection
```python
import deepchem as dc
from deepchem.models import CGCNNModel
from deepchem.feat import CGCNNFeaturizer
import pickle
import numpy as np
```

When pipreqs detects `deepchem` in imports, the Docker harness automatically adds DGL.

## Why This Fix is Fair

- Agent still must understand CGCNN architecture and crystal structure featurization
- No hints about model configuration or training
- Only compensates for sandbox limitation

## Expected Outcome After Fix

- Agents write code with proper deepchem imports
- pipreqs detects deepchem → DGL installed via post-install hook
- CGCNN model can be properly instantiated and trained
