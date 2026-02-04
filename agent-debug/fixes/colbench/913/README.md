# Task 913 Fix: Peace Index with Arbitrary Weights

## IFE Summary

The hidden reference solution contains specific weights and a violence formula:

```python
def calculate_peace_index(election_turnout, rebel_support, economic_justice,
                         minority_rights, num_violent_incidents, total_population):
    return (election_turnout * 0.3 +
            rebel_support * 0.2 +
            economic_justice * 0.2 +
            minority_rights * 0.1 +
            (1 - num_violent_incidents / total_population) * 0.2)
```

The task description only says the index should be "based on" various factors - it does not specify:
- The exact weights (0.3, 0.2, 0.2, 0.1, 0.2)
- The violence transformation: `1 - num_violent_incidents / total_population`
- That rebel_support should be added directly (not inverted as some agents reasonably assumed)

## Evidence

- Hidden weights: `0.3` for election_turnout, `0.2` for rebel_support, etc.
- Hidden violence formula: `(1 - num_violent_incidents/total_population) * 0.2`
- Task description is vague: "based on how fair the elections are, how much support the rebels have, and some other stuff"

## Fix Strategy

The weights are completely arbitrary and the violence transformation cannot be inferred from a vague description. Some runs succeeded only after the simulated user "leaked" the weights.

## Fix Applied

- `instruction_override.json`: Specifies the exact weights and violence formula
