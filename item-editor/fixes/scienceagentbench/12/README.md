# Task 12 Fix: DeepPurpose DAVIS DTI Drug Repurposing

## Root Cause Analysis

**IFE Type**: Sandbox Import Restriction + pipreqs Detection Failure + Package Structure

### Problem Description
Task requires using DeepPurpose library for drug-target interaction (DTI) prediction on the DAVIS dataset.

### Why This Was Failing

1. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `DeepPurpose` imports
2. **pipreqs Detection Failure**: Without proper imports, pipreqs doesn't detect DeepPurpose → descriptastorus not installed
3. **Package Structure**: DeepPurpose has an unusual structure - note the capitalization `DeepPurpose` not `deeppurpose`

### Evidence from Verdict
"DeepPurpose is unavailable/unusable due to dependency incompatibilities (TensorFlow/Keras stack)"

**CORRECTION**: The Docker base image has **TensorFlow 2.17** and **torch 2.3** pre-installed. The issue was:
1. pipreqs not detecting DeepPurpose
2. descriptastorus (required dependency) not being installed as a result

## Fix Applied

### 1. Instruction Override
Added critical clarifications:
- **MUST include proper imports** with correct capitalization
- DeepPurpose package structure: `from DeepPurpose import utils, models`
- How to use DTI module for drug-target interaction

### 2. Critical Imports for pipreqs Detection
```python
from DeepPurpose import utils
from DeepPurpose import dataset
from DeepPurpose import DTI
from DeepPurpose import models
import torch
import pandas as pd
import numpy as np
```

**IMPORTANT**: The capitalization must be `DeepPurpose` not `deeppurpose` for pipreqs to detect it correctly.

When pipreqs detects `DeepPurpose` in imports, the Docker harness automatically installs:
- `git+https://github.com/bp-kelley/descriptastorus` (required dependency)

## Why This Fix is Fair

- Agent still must understand DTI modeling and drug-target interaction prediction
- No hints about model architecture, encoders, or training
- Only clarifies import structure that agents cannot discover without testing

## Expected Outcome After Fix

- Agents write code with properly capitalized `DeepPurpose` imports
- pipreqs detects DeepPurpose → descriptastorus installed
- DeepPurpose models module properly importable
- DTI prediction pipeline works
