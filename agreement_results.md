# Inter-Rater Agreement Analysis

This report summarizes the consistency between raters using metrics robust to unbalanced datasets.

### Why was Cohen's Kappa removed?
Cohen's Kappa is unreliable for these benchmarks due to the **Kappa Paradox**. Because our data is heavily skewed towards one category (Passing/0), Kappa penalizes agreement on that majority class, leading to near-zero scores even when raters agree on 95%+ of items. Gwet's AC1 and PABAK provide a more accurate measure of reliability for this data.

| Benchmark         |   Raters |   Tasks |   % Agree |   Gwet's AC1 |   PABAK |   Consensus Agreement (AC1) |
|:------------------|---------:|--------:|----------:|-------------:|--------:|----------------------------:|
| SciCode           |       10 |      29 |    0.9686 |       0.9666 |  0.9372 |                      0.9286 |
| ScienceAgentBench |       10 |      24 |    1.0000 |       1.0000 |  1.0000 |                      1.0000 |
| CORE              |       10 |      11 |    0.9818 |       0.9801 |  0.9636 |                      0.9900 |
| ColBench          |       51 |    1000 |    0.7255 |       0.5595 |  0.4511 |                      0.9600 |

### Metric Interpretations
- **% Agree**: The raw percentage of identical verdicts.
- **Gwet's AC1**: The primary reliability metric. It adjusts for chance while remaining stable even if one category is very rare (like our Faults).
- **PABAK**: Prevalence-Adjusted Bias-Adjusted Kappa. A normalized version of Kappa that assumes a neutral distribution to eliminate bias.
- **Consensus Agreement (AC1)**: Consistency between individual raters and their specific benchmark consensus.

