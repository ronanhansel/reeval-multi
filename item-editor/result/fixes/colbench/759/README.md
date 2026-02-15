# Task 759 Fix: DNA Patent Eligibility

## IFE Summary

**Task**: Determine if a DNA sequence is patent eligible.

**Problem Description**: "I need a python function that determines if a DNA sequence is patent eligible or not. It has something to do with isolated human DNA, and some other factors like sequence length and human DNA characteristics."

**Hidden Reference Implementation**:
```python
def is_patent_eligible(dna_sequence, sequence_length, is_isolated, is_human_dna):
    return is_isolated and is_human_dna and (sequence_length > 1000) and (
        'hand of man' in dna_sequence or dna_sequence.startswith('cDNA')
    )
```

## Intrinsic Formation Error

**Type**: Arbitrary Hidden Thresholds and Magic String Checks

**Issue**: The hidden implementation includes highly specific, arbitrary conditions that cannot be derived from the vague prompt:

1. **Threshold**: `sequence_length > 1000` - arbitrary magic number
2. **String Check 1**: `'hand of man' in dna_sequence` - arbitrary substring
3. **String Check 2**: `dna_sequence.startswith('cDNA')` - arbitrary prefix

**Why this is an IFE (not capability issue)**:
1. The prompt mentions "some other factors" without specifying them
2. "Hand of man" is a legal concept in patent law but checking for it as a literal substring is unusual
3. The 'cDNA' prefix check is a valid scientific concept but not mentioned in prompt
4. The exact 1000 bp threshold is not specified
5. No agent could guess these exact conditions from "something to do with isolated human DNA"

## Fix Type

**Instruction Clarification** - Specify the exact eligibility conditions.

## Fix Rationale

The fix reveals the specific boolean conditions required. These are arbitrary implementation details that cannot be discovered through normal dialogue.

**This fix makes evaluation FAIR, not EASY**: The agent still needs to implement the boolean logic correctly with substring/prefix checks. We're revealing conditions that couldn't be reasonably guessed.
