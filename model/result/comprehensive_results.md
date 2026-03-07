# Comprehensive Model Comparison

| Model Configuration           | AUC         | RMSE        |
|:------------------------------|:------------|:------------|
| Naive (Post-max Baseline)     | 0.500       | 0.265±0.006 |
| Rasch IRT (Post-max Baseline) | 0.574±0.012 | 0.269±0.008 |
| Naive (Post-1 Baseline)       | 0.500       | 0.265±0.005 |
| Rasch IRT (Post-1 Baseline)   | 0.580±0.009 | 0.327±0.015 |
| SAE Post (N=1)                | 0.680±0.020 | 0.254±0.006 |
| SAE Post (N=max)              | 0.696±0.015 | 0.256±0.007 |
| PCA Post (N=1)                | 0.683±0.029 | 0.251±0.005 |
| PCA Post (N=max)              | 0.708±0.015 | 0.251±0.007 |
| RAW Post (N=1)                | 0.651±0.034 | 0.269±0.006 |
| RAW Post (N=max)              | 0.715±0.014 | 0.254±0.007 |
| Naive-8 (Pre Baseline)        | 0.500       | 0.451±0.002 |
| Rasch-8 (Pre Baseline)        | 0.578±0.007 | 0.478±0.007 |
| SAE Pre-8 (N=1)               | 0.652±0.016 | 0.432±0.004 |
| PCA Pre-8 (N=1)               | 0.668±0.013 | 0.427±0.003 |
| RAW Pre-8 (N=1)               | 0.641±0.012 | 0.452±0.006 |
| Naive Pre-max (Baseline)      | 0.500       | 0.447±0.001 |
| Rasch Pre-max (Baseline)      | 0.653±0.010 | 0.429±0.002 |
| SAE Pre-max (N=max)           | 0.736±0.014 | 0.414±0.005 |
| PCA Pre-max (N=max)           | 0.740±0.013 | 0.416±0.005 |
| RAW Pre-max (N=max)           | 0.733±0.011 | 0.431±0.006 |
