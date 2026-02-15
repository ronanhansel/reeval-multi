# Grade=1 IFE Fixes Summary (sab_cow run)

## Overview

This document summarizes the comprehensive fixes applied to all Grade=1 (IFE - Intrinsic Formation Error) tasks from the `sab_cow` verdict.

**Total Grade=1 Tasks Fixed: 7**

| Task | Package | IFE Type | Fix Applied |
|------|---------|----------|-------------|
| 52 | DeepChem | Sandbox import restriction + pipreqs detection failure | instruction_override |
| 95 | DeepChem | Sandbox import restriction + pipreqs detection failure | instruction_override |
| 97 | DeepChem | Sandbox import restriction + pipreqs detection failure | instruction_override |
| 64 | OGGM | Sandbox import restriction + pipreqs detection failure | instruction_override |
| 74 | OGGM | Sandbox import restriction + API location confusion | instruction_override |
| 12 | DeepPurpose | Sandbox import restriction + pipreqs detection failure | instruction_override |
| 102 | MODNet | Missing Docker handler + pipreqs detection failure | instruction_override + harness modification |

## Root Cause Analysis

### The Common Pattern

All Grade=1 tasks shared a common root cause:

1. **Agent Sandbox Blocks Imports**: The smolagents CodeAgent runs in a restricted sandbox that blocks imports of specialized scientific packages (deepchem, oggm, modnet, DeepPurpose)

2. **Agent Cannot Verify Code**: Without being able to import packages, agents cannot test their code during development

3. **pipreqs Detection Failure**: If the agent's generated code doesn't have proper import statements, `pipreqs` doesn't detect the packages

4. **Packages Not Installed**: When pipreqs doesn't detect the package, it's not added to `instance_requirements.txt`, and:
   - For DeepChem: DGL (Deep Graph Library) is not installed via post-install hook
   - For OGGM: salem, tables, geopandas are not added
   - For DeepPurpose: descriptastorus is not installed
   - For MODNet: pymatgen, matminer are not added (NOW FIXED)

### The Misconception in the Verdict

The verdict incorrectly attributed failures to:
- "DeepChem requires Python <3.10" (WRONG - Docker uses Python 3.10)
- "TensorFlow is missing" (WRONG - Docker has TensorFlow 2.17 pre-installed)
- "Packages not installed" (PARTIALLY CORRECT - but caused by pipreqs failure, not environment issue)

**The Docker base image includes:**
- Python 3.10
- TensorFlow 2.17
- RDKit 2023.09.5
- torch 2.3.0
- numpy, scipy, matplotlib, pandas, scikit-learn

## Fixes Applied

### 1. Instruction Overrides

For each task, we added `instruction_override.json` that tells the agent:

> **CRITICAL: Your final Python code MUST include these import statements at the top: [list of imports]. The evaluation environment has these packages pre-installed. The development sandbox blocks these imports, but the evaluation Docker container has them properly configured. Write your code as if imports work.**

This ensures:
- Agent writes code with proper import statements
- pipreqs detects the packages
- Post-install hooks trigger (DGL for deepchem, salem for oggm, etc.)

### 2. Docker Harness Modification (Task 102 only)

For MODNet (Task 102), we modified `dockerfiles.py` to add:

```bash
if echo "$extracted_pkgs" | grep -q 'modnet'; then \
    echo 'pymatgen<=2024.5.1' >> /testbed/instance_requirements.txt && \
    echo 'matminer' >> /testbed/instance_requirements.txt; \
fi;
```

This was necessary because MODNet did NOT have special handling like deepchem/oggm/scanpy.

### 3. Environment Overrides

Each task has `env_override.json` documenting the required packages. While these are not directly applied by the current harness, they serve as documentation for future harness modifications.

## Task-Specific Details

### Task 52: DeepChem GCN
- **Imports to include**: `import deepchem as dc`, `from deepchem.feat import MolGraphConvFeaturizer`, `from deepchem.models import GraphConvModel`, `from rdkit import Chem`
- **Post-install trigger**: deepchem detected → DGL installed

### Task 95: DeepChem ScScore
- **Imports to include**: `import deepchem as dc`, `from deepchem.models import ScScoreModel`, `from deepchem.feat import CircularFingerprint`, `import pickle`
- **Post-install trigger**: deepchem detected → DGL installed

### Task 97: DeepChem CGCNN
- **Imports to include**: `import deepchem as dc`, `from deepchem.models import CGCNNModel`, `from deepchem.feat import CGCNNFeaturizer`, `import pickle`
- **Post-install trigger**: deepchem detected → DGL installed

### Task 64: OGGM Mass Balance
- **Imports to include**: `import oggm`, `from oggm import cfg, workflow, tasks, utils`, `from oggm.core.massbalance import MultipleFlowlineMassBalance`, `import salem`
- **Post-install trigger**: oggm detected → salem, tables, geopandas installed

### Task 74: OGGM distribute_2d
- **Imports to include**: `import oggm`, `from oggm.sandbox.distribute_2d import distribute_thickness_from_simulation`, `import salem`
- **Critical note**: API is in `oggm.sandbox.distribute_2d`, NOT `oggm.core.flowline`
- **Post-install trigger**: oggm detected → salem, tables, geopandas installed

### Task 12: DeepPurpose DTI
- **Imports to include**: `from DeepPurpose import utils, dataset, DTI, models`, `import torch`
- **Critical note**: Capitalization must be `DeepPurpose` not `deeppurpose`
- **Post-install trigger**: DeepPurpose detected → descriptastorus installed

### Task 102: MODNet
- **Imports to include**: `from modnet.models import MODNetModel`, `from modnet.preprocessing import MODData`, `import pymatgen`, `import pickle`
- **Post-install trigger**: modnet detected → pymatgen, matminer installed (NEW HANDLER)

## Expected Outcome

After applying these fixes:

1. Agents will receive instruction clarifications telling them to include proper imports
2. Generated code will have correct import statements
3. pipreqs will detect the packages
4. Post-install hooks will install required dependencies
5. Code will execute properly in evaluation container

All 7 Grade=1 tasks should resolve to Grade=0 in the next run.

## Files Modified

1. **New/Updated instruction_override.json** for tasks: 52, 95, 97, 64, 74, 12, 102
2. **Updated env_override.json** for tasks: 52, 95, 97, 64, 74, 12, 102
3. **Updated README.md** for tasks: 52, 95, 97, 64, 74, 12, 102
4. **New status.json** for tasks: 52, 95, 97, 64, 74, 12, 102
5. **Modified dockerfiles.py** to add MODNet handler
