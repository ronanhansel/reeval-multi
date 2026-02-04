# Task 55 Fix: Iris Library Environment Issue

## Root Cause Analysis

**IFE Type**: Execution Environment Issue - Sandbox Import Restrictions + Potential Data Handling Issue

### Problem Description
Task 55 requires analyzing South Atlantic Ocean temperature and salinity data. The task involves:
1. Using Iris library to load NetCDF data (`atlantic_profiles.nc`)
2. Using `iris.Constraint` to filter data by latitude, longitude, depth
3. Creating a temperature-salinity profile plot with `iris.plot`
4. Saving visualization to PNG

### Evidence of IFE

**Sandbox Import Restrictions** (from rubric evaluations):
```
InterpreterError: Import of iris is not allowed
InterpreterError: Import of matplotlib.pyplot is not allowed
```

The task explicitly requires using "the Iris library (iris.load, iris.Constraint, iris.plot)" but the sandbox blocks all iris imports.

**Docker Evaluation Result**:
```
valid_program: 0
codebert_score: 0.7928
success_rate: 0
log_info: ValueError: can't merge an empty CubeList
```

### Analysis of the "Empty CubeList" Error
The Docker error `can't merge an empty CubeList` occurs when:
1. `iris.load()` returns cubes, but
2. `iris.load_cube()` or filtering returns no matching data

This could indicate:
- Agent's constraint parameters don't match any data in the NetCDF file
- Incorrect variable name for loading (temperature/salinity)
- The agent couldn't test/validate its code due to import restrictions

The Docker Dockerfile already has special handling to map `iris` to `scitools-iris`, so the package IS available. The issue is that agents can't develop correct constraint logic without being able to test.

## Fix Applied

### Environment Override (`env_override.json`)
- **HAL_PIP_PACKAGES**: scitools-iris matplotlib cartopy netCDF4

### Why This Fix is Fair (Not a Nerf)
1. **Preserves Scientific Rigor**: The task still requires:
   - Understanding NetCDF oceanographic data structure
   - Proper use of `iris.Constraint` for lat/lon/depth filtering
   - Creating meaningful temperature-salinity profile visualizations
2. **No Hints Given**: We don't provide constraint values, variable names, or plotting logic
3. **Standard Toolchain**: Iris is the specified library (SciTools/iris from Met Office)
4. **Cross-Model Evidence**: All 4 models encountered identical import barriers

### Note on Residual Issues
The "empty CubeList" error may persist even after this fix if agents generate incorrect constraint logic. This would be an agent capability issue, not an IFE. The fix addresses the infrastructure barrier (import restrictions); solving the scientific problem correctly remains the agent's responsibility.

## Expected Outcome After Fix
- Agents can conceptually work with Iris library (even if sandbox blocks execution)
- Docker evaluation has full Iris/matplotlib/cartopy stack available
- Task success depends on correct NetCDF data loading and T-S profile plotting

## Technical Notes
- Iris is developed by the UK Met Office for meteorological/oceanographic data
- NetCDF is a common format for scientific climate/ocean data
- `iris.Constraint` is used for data subsetting by coordinates
- Cartopy may be needed for geographical coordinate handling
- Input: `ocean_profiles/atlantic_profiles.nc`
- Task requires filtering by latitude, longitude, and depth ranges
