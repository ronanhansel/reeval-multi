# Task 248: Tau Protein Spread Calculator

## IFE Analysis

**Task Type:** Backend Programming (test case evaluation)

**IFE Identified:** Yes - Arbitrary Algorithm Details Not Recoverable from Dialogue

### Problem Description
```
I need a python function to analyze the spread of tau protein in the brain of Alzheimer's
disease. Given a list of region connections and the tau protein values for each region,
calculate the regions where tau protein spreads. I don't have the details, can you ask
some questions to help me clarify?
The signature: def calculate_tau_spread(tau_values, region_connections)
```

### Hidden Ground Truth
```python
def calculate_tau_spread(tau_values, region_connections):
    tau_distribution = {region: 0 for region in set([k for connection in region_connections for k in connection])}
    for connection in region_connections:
        tau_distribution[connection[0]] += tau_values[connection[0]]
        tau_distribution[connection[1]] += tau_values[connection[1]]
    max_value = max(tau_distribution.values())
    tau_spread_regions = [region for region, value in tau_distribution.items() if value >= max_value/2]
    return tau_spread_regions
```

### Evidence from Rubric Evaluations

1. **Evaluation 6 (Grade 1.0):** "The benchmark's hidden reference solution defines 'tau spread' via an arbitrary aggregation rule and threshold that are not described in the natural-language task... Hidden code constructs tau_distribution by summing tau_values for each endpoint over all connections, then labels spread regions as those with value >= max_value/2"

2. **Evaluation 6:** "When the agent asked for the spread rule and threshold, the simulated user responded that they don't know, despite the hidden code containing the exact rule/threshold"

### Root Cause - Arbitrary Algorithm

The hidden algorithm has several arbitrary, unguessable components:

1. **Aggregation method:** Sum tau_values at each endpoint across all connections
   - Not standard neuroscience (usually propagation/diffusion models)
   - Counts each region's tau value once per connection it appears in

2. **Threshold:** `>= max_value/2`
   - Why half? Not derivable from any scientific principle
   - Could be any threshold (0.75, 0.5, mean, etc.)

3. **No actual "spread" calculation:**
   - Despite task mentioning "spread," the algorithm doesn't model spread
   - It just aggregates existing values

**Simulated user cannot help:**
- User says "I don't know the specific biological rule for tau spread"
- User says "I don't know the specific numeric threshold to use"
- The constants only exist as code literals

### Why This is NOT Fixable at Task Level

**Cannot be fixed without "nerfing"** because:
1. Revealing the aggregation method (sum at endpoints) gives away the algorithm
2. Revealing the `max/2` threshold gives away the selection criteria
3. The task explicitly invites clarifying questions, but the answers aren't available

The fundamental problem is that "tau spread" has no well-defined scientific meaning in this context, and the hidden implementation is just one arbitrary interpretation.

## Fix Recommendation

**No code fix applied** - The task has an intrinsic formation error. The benchmark would need to:
1. Specify the exact aggregation/spread algorithm in the task
2. Specify the threshold criterion
3. Or accept any reasonable tau spread interpretation

Documenting as unfixable IFE.
