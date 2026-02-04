# Task 74 Fix: OGGM Glacier Area and Thickness Distribution

## Root Cause Analysis

**IFE Type**: Sandbox Import Restriction + pipreqs Detection Failure + API Location Confusion

### Problem Description
Task requires using OGGM's 2D distribution functionality to compute and visualize glacier thickness distribution.

### Why This Was Failing

1. **Agent Sandbox Blocks Imports**: The smolagents sandbox blocks `oggm` and `salem` imports
2. **pipreqs Detection Failure**: Without proper `import oggm`, pipreqs doesn't detect OGGM → dependencies not added
3. **API Location Confusion**: `distribute_2d` is in `oggm.sandbox.distribute_2d`, NOT `oggm.core.flowline`

### Evidence from Verdict
"oggm missing; salem explicitly blocked"

The agents were also trying to import from the wrong location (`oggm.core.flowline` instead of `oggm.sandbox.distribute_2d`).

## Fix Applied

### 1. Instruction Override
Added critical clarifications:
- **MUST include proper imports** for oggm and salem
- **IMPORTANT**: The correct import path is `from oggm.sandbox.distribute_2d import distribute_thickness_from_simulation`
- This is publicly documented but agents cannot discover it through testing

### 2. Critical Imports for pipreqs Detection
```python
import oggm
from oggm import cfg, workflow
from oggm.sandbox.distribute_2d import distribute_thickness_from_simulation
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
- Network access enabled

## Why This Fix is Fair

- Agent still must understand glacier thickness computation and 2D distribution
- No hints about the scientific implementation
- Only clarifies API location that agents cannot discover without testing imports
- The `oggm.sandbox` namespace is publicly documented for experimental features

## Expected Outcome After Fix

- Agents write code with correct import paths
- pipreqs detects oggm → salem, tables, geopandas installed
- distribute_2d functionality properly imported
- Thickness distribution visualization works
