# Task 89 - Geoplot Quadtree Tree Species Visualization

## Root Cause Analysis

**Execution Result**: valid_program: 0, FutureWarning about geometry column

The Docker execution shows:
```
FutureWarning: You are adding a column named 'geometry' to a GeoDataFrame
constructed without an active geometry column. Currently, this automatically
sets the active geometry column to 'geometry' but in the future that will
no longer happen.
```

### Why This Happened

1. **Agent Sandbox Limitation**: The smolagents CodeAgent sandbox does NOT allow importing `geopandas` or `geoplot`, which are explicitly required by the task.

2. **API Deprecation**: The code produced by the agent uses deprecated GeoDataFrame patterns that trigger FutureWarnings. The agent couldn't test the code due to sandbox restrictions.

3. **Task Requirements**: The task explicitly requires:
   - `geoplot.quadtree()` for visualization
   - `geoplot.polyplot()` for polygons
   - `geodataframe.assign()` for data manipulation

The agent sandbox blocks these imports entirely, making it impossible to develop working code.

## Is This an IFE?

**Yes** - This is an **Agent Sandbox Environment Mismatch** issue.

The task requires libraries (geopandas, geoplot) that are:
1. Blocked in the agent's sandbox during development
2. Available in the Docker evaluation container

This creates an impossible situation where:
- The agent cannot test geopandas/geoplot code
- The agent must produce correct code blind
- Any deprecation issues cannot be discovered during development

## Fix Applied

**Instruction Override**: Added clarifications about:
1. The correct GeoDataFrame construction pattern to avoid deprecation warnings
2. The expected geoplot API usage

This fix is FAIR because:
- It doesn't simplify the scientific problem
- It only addresses API compatibility issues
- It compensates for the agent's inability to test the code

## Expected Outcome

After applying this fix, agents should:
1. Use correct GeoDataFrame construction patterns
2. Produce valid geoplot quadtree visualizations
3. Avoid FutureWarning deprecation issues

## Files Changed

- `instruction_override.json`: Added API usage clarifications
- `README.md`: This documentation
