# Comprehensive Model Comparison

| Model Configuration           | AUC         | RMSE        |
|:------------------------------|:------------|:------------|
| Naive (Post-max Baseline)     | 0.500       | 0.266±0.002 |
| Rasch IRT (Post-max Baseline) | 0.571±0.006 | 0.272±0.002 |
| Naive (Post-1 Baseline)       | 0.500       | 0.266±0.002 |
| Rasch IRT (Post-1 Baseline)   | 0.574±0.004 | 0.340±0.005 |
| SAE Post (N=1)                | 0.697±0.001 | 0.252±0.000 |
| SAE Post (N=max)              | 0.693±0.001 | 0.256±0.000 |
| PCA Post (N=1)                | 0.706±0.001 | 0.250±0.000 |
| PCA Post (N=max)              | 0.710±0.001 | 0.251±0.000 |
| RAW Post (N=1)                | 0.697±0.001 | 0.263±0.000 |
| RAW Post (N=max)              | 0.714±0.000 | 0.253±0.000 |
| Naive-32 (Pre Baseline)       | 0.000       | 0.000       |
| Rasch-32 (Pre Baseline)       | 0.000       | 0.000       |
| SAE Pre-32 (N=1)              | 0.687±0.000 | 0.419±0.000 |
| PCA Pre-32 (N=1)              | 0.695±0.000 | 0.416±0.000 |
| RAW Pre-32 (N=1)              | 0.683±0.000 | 0.433±0.000 |
| Naive Pre-max (Baseline)      | 0.000       | 0.000       |
| Rasch Pre-max (Baseline)      | 0.000       | 0.000       |
| SAE Pre-max (N=max)           | 0.726±0.000 | 0.423±0.000 |
| PCA Pre-max (N=max)           | 0.728±0.001 | 0.422±0.000 |
| RAW Pre-max (N=max)           | 0.722±0.001 | 0.435±0.000 |
| SAE Post (N=1, No-TAU)        | 0.691±0.011 | 0.250±0.002 |
| SAE Post (N=max, No-TAU)      | 0.686±0.007 | 0.253±0.002 |
| PCA Post (N=max, No-TAU)      | 0.660±0.008 | 0.246±0.002 |
| RAW Post (N=max, No-TAU)      | 0.665±0.006 | 0.244±0.002 |
| ONES Post (N=1)               | 0.572±0.000 | 0.261±0.000 |
| ONES Post (N=max)             | 0.570±0.000 | 0.266±0.000 |
| ONES Pre-32 (N=1)             | 0.622±0.000 | 0.434±0.000 |
| ONES Pre-max (N=max)          | 0.676±0.000 | 0.452±0.000 |
