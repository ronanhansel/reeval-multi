# Task 782 Fix: Programming Language Production Readiness

## IFE Summary

**Task**: Determine if a programming language is "production ready" based on stats.

**Problem Description**: "I want a function that determines if a programming language is 'production ready'. The function should consider several factors such as the number of releases, stability, and other relevant information."

**Hidden Reference Implementation**:
```python
def is_production_ready(language_stats, release_threshold=1000, stability_threshold=0.9):
    if len(language_stats) < release_threshold or \
       sum(1 for stat in language_stats if stat[1] >= stability_threshold) / len(language_stats) < 0.9:
        return False
    return True
```

## Intrinsic Formation Error

**Type**: Under-specified Data Structure + Hidden Ratio Rule

**Issue 1 - Data Structure**: The implementation expects `language_stats` to be:
- An iterable with `len()` support
- Each element is indexable, where `stat[1]` is the stability value
- This is a very specific format (e.g., list of tuples) not mentioned in prompt

**Issue 2 - Hidden 90% Rule**: There's an additional hardcoded requirement that at least 90% of entries must meet the stability threshold. This rule is not in the function parameters or prompt.

**Why this is an IFE (not capability issue)**:
1. The prompt says "consider several factors" without specifying the data structure
2. The positional indexing `stat[1]` implies tuples/lists, not dicts
3. The 0.9 (90%) proportion check is completely hidden
4. An agent implementing reasonable "production ready" logic with different data structures would fail

**The logic**:
- Return False if `len(language_stats) < release_threshold`
- Return False if proportion of stats meeting `stat[1] >= stability_threshold` is < 0.9
- Otherwise return True

## Fix Type

**Instruction Clarification** - Specify the data structure format and the hidden 90% rule.

## Fix Rationale

The fix reveals the expected data structure and the hidden proportion check. These are implementation details that cannot be derived from the generic prompt.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the conditional logic correctly. We're revealing the data structure contract and hidden business rule.
