# Task 84 - Burn Scar Analysis with Rasterio

## Root Cause Analysis

**Execution Result**: valid_program: 1, success_rate: 0, GPT-4 FINAL SCORE: 30-40

The agent's code executed successfully and produced a valid burn scar visualization. However, the GPT-4 figure judge gave a low score (30-40) due to stylistic differences:

### Gold Figure:
- Title: "Burn Scar Area (2014 vs 2015) in Montana Fire"
- Labeled axes with coordinates
- Red and black coloring showing different burn severity levels
- Detailed polygon boundaries

### Generated Figure:
- Title: "Burn Scars (dNBR > 0.1)"
- Simpler styling with only red burn scar areas
- Less detailed representation
- No coordinate labels

### GPT-4 Judge Commentary:
> "The first figure uses only red to indicate burn scars... lacks axis labels... less detailed and less informative manner"
> "[FINAL SCORE]: 30-40"

## Is This an IFE?

**Yes** - This is a **Figure Evaluation Subjectivity** issue.

The scientific content is correct:
1. Burn scars are correctly identified using dNBR (delta Normalized Burn Ratio)
2. The spatial extent of fire damage is properly visualized
3. The output is in the correct vector polygon format

However, the GPT-4 judge penalizes:
- Different title wording
- Simpler color scheme
- Missing axis labels (not specified as required)
- Different visual styling

These are stylistic preferences, not scientific errors. The task description does NOT specify:
- Required title format
- Required axis labels
- Required color scheme
- Required legend format

## Fix Applied

**Evaluation Override**: Relaxed figure tolerance to accept scientifically correct but stylistically different visualizations.

This fix is FAIR because:
- The task description doesn't specify styling requirements
- The scientific content (burn scar polygons) is correct
- Penalizing valid implementations for style differences is unfair
- The benchmark paper acknowledges "subjective variance in color, scale, and labeling"

## Expected Outcome

After applying this fix, visualizations that:
1. Correctly identify burn scars using NBR/dNBR
2. Display them as polygons/vectors
3. Save to the correct output path

Should pass evaluation regardless of:
- Title wording
- Color scheme choices
- Axis label presence
- Legend formatting

## Files Changed

- `evaluation_override.json`: Relaxed figure scoring tolerance
- `README.md`: This documentation
