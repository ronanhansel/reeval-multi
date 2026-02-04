# Task 64 Fix: OGGM Flowline Mass Balance Plotting

## Root Cause Analysis

**IFE Type**: Sandbox Import Restriction + pipreqs Detection Failure + Configuration Unclear

### Problem Description
Task requires using OGGM (Open Global Glacier Model) to compute and visualize glacier mass balance using local pre-computed glacier data.

### Why This Was Failing

1. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `oggm` and `salem` imports
2. **pipreqs Detection Failure**: Without proper `import oggm`, pipreqs doesn't detect OGGM → salem/tables/geopandas not added
3. **Configuration Unclear**: Task provides local glacier data but doesn't explain how to configure OGGM to use it instead of downloading

### Evidence from Verdict
"importing oggm fails immediately with ModuleNotFoundError: No module named 'oggm'"

The Docker harness DOES have special handling for OGGM (adds salem, tables, geopandas) - but only if pipreqs detects 'oggm' in the agent's code.

## Fix Applied

### 1. Instruction Override
Added critical clarifications:
- **MUST include proper imports** for oggm and salem
- How to initialize OGGM configuration
- How to configure working directory for local data
- Location of local glacier data files

### 2. Critical Imports for pipreqs Detection
```python
import oggm
from oggm import cfg, workflow, tasks, utils
from oggm.core.massbalance import MultipleFlowlineMassBalance
import matplotlib.pyplot as plt
import salem
import geopandas as gpd
```

When pipreqs detects `oggm` in imports, the Docker harness automatically adds:
- salem
- tables
- geopandas

### 3. Environment Override
- Extended timeout for potential network operations
- Network access enabled for climate data if needed

## Why This Fix is Fair

- Agent still must understand glacier mass balance computation
- No hints about the actual scientific implementation
- Only clarifies configuration that agents cannot discover without testing imports

## Expected Outcome After Fix

- Agents write code with proper oggm imports
- pipreqs detects oggm → salem, tables, geopandas installed
- OGGM initialized with local data configuration
- Mass balance computation and visualization works
