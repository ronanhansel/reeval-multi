# Task 102 Fix: MODNet Refractive Index Prediction

## Root Cause Analysis

**IFE Type**: Missing Package Handler in Docker Harness + Pickle Deserialization Failure

### Problem Description
Task requires using MODNet (Material Optimal Descriptor Network) to predict refractive index from materials data. The training data is stored in pickle files that contain MODNet-specific objects.

### Why This Was Failing - CRITICAL ISSUE

1. **No Special Handler**: Unlike DeepChem, OGGM, and Scanpy, **MODNet has NO special handling** in the Docker harness's Dockerfile
2. **Pickle Deserialization Failure**: The training data pickles contain MODNet class instances - they CANNOT be loaded without MODNet installed
3. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `modnet` and `pickle` imports

### Evidence from Verdict
"ModuleNotFoundError: No module named 'modnet'" and "the dataset pickles require MODNet classes to unpickle"

This is the ONLY task where even loading the input data is impossible without the package installed.

## Fix Applied

### 1. Added MODNet Handler to Docker Harness

**Modified `dockerfiles.py`** to add MODNet detection:
```bash
if echo "$extracted_pkgs" | grep -q 'modnet'; then \
    echo 'pymatgen<=2024.5.1' >> /testbed/instance_requirements.txt && \
    echo 'matminer' >> /testbed/instance_requirements.txt; \
fi;
```

This ensures when pipreqs detects `modnet` imports, the required dependencies (pymatgen, matminer) are also installed.

### 2. Instruction Override
Added critical clarifications:
- **MUST include proper imports** for modnet, pymatgen
- How to load MODData objects from pickle
- MODNetModel architecture specification

### 3. Critical Imports for pipreqs Detection
```python
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
import pymatgen
from pymatgen.core import Structure
import pandas as pd
import numpy as np
import pickle
```

## Why This Fix is Fair

- Agent still must understand MODNet architecture and materials property prediction
- No hints about model configuration or training
- Only addresses the missing Docker handler that makes the task impossible

## Expected Outcome After Fix

- Docker harness detects `modnet` imports
- pymatgen and matminer are added as dependencies
- MODNet package is installed
- Pickle files with MODData objects can be loaded
- Refractive index prediction pipeline works

## Technical Notes

MODNet requires:
- `modnet` - The main package
- `pymatgen` - For crystal structure handling
- `matminer` - For material feature extraction
- TensorFlow/Keras backend (already in base image)

The pickle files contain serialized MODData objects which reference:
- `modnet.preprocessing.MODData` class
- pymatgen Structure objects
- matminer featurizer outputs
