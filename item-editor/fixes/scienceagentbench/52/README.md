# Task 52 Fix: DeepChem GCN for Aquatic Toxicity QSAR

## Root Cause Analysis

**IFE Type**: Sandbox Import Restriction + pipreqs Detection Failure

### Problem Description
Task 52 requires training a Graph Convolutional Network (GCN) for aquatic toxicity QSAR and visualizing atomic contributions using DeepChem and RDKit.

### Why This Was Failing

1. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `deepchem`, `rdkit` imports during development
2. **Agent Cannot Test Code**: Without being able to import these packages, agents cannot verify their code works
3. **pipreqs Detection Failure**: If agent code doesn't have proper `import deepchem` statements, pipreqs doesn't detect it
4. **Missing DGL Post-Install**: The Docker harness has special handling that adds DGL when deepchem is detected - but only if deepchem is in the pipreqs output

### Evidence from Verdict
The verdict confirmed this is an IFE: "pip install deepchem==2.7.1 fails because available DeepChem versions require Python <3.10"

**CORRECTION**: The Docker base image actually uses **Python 3.10** and has **TensorFlow 2.17** pre-installed. The real issue was that agents couldn't write proper imports because sandbox blocked them during development.

## Fix Applied

### 1. Instruction Override
Added critical clarifications telling agents:
- **MUST include proper imports** at the top of their code
- The evaluation Docker has all packages pre-installed
- Write code as if imports work (even though sandbox blocks them during development)

### 2. Critical Imports for pipreqs Detection
```python
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models import GraphConvModel
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
```

When pipreqs detects `deepchem` in imports, the Docker harness automatically adds DGL.

## Why This Fix is Fair (Not a Nerf)

1. **Preserves Scientific Rigor**: Agent still must understand GCN architecture and implement atomic contribution visualization
2. **No Hints Given**: No solution logic about model hyperparameters, training, or visualization algorithm
3. **Standard Toolchain**: DeepChem and RDKit are the task's specified libraries
4. **Only Compensates for Infrastructure**: The fix only addresses the sandbox limitation that prevents normal development

## Expected Outcome After Fix

- Agents will write code with proper `import deepchem` statements
- pipreqs will detect deepchem and add it to instance_requirements.txt
- Docker post-install hook will add DGL
- Code will execute properly in evaluation container

## Environment Details

The Docker base image includes:
- Python 3.10
- TensorFlow 2.17
- RDKit 2023.09.5
- torch 2.3
- numpy, scipy, matplotlib, pandas, scikit-learn
