# Experiment Appendix: Consolidated Statistics Table

| Benchmark | Traces | Unique Models | Input Tokens | Input Cost ($) | Output Tokens | Output Cost ($) | Reasoning | Cached | Total Cost ($) | Cumulative Trace Runtime | Avg Trace Runtime | Median Trace Runtime | P90 Trace Runtime | Min Trace Runtime | Max Trace Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ColBench Backend | 439 | 11 | 1,141,691,588 | $2515.82 | 1,482,775,583 | $18247.05 | 1,121,108,608 | 230,144 | $20762.87 | 823h 31m 35s | 2h 8m 1s | 1h 30m 14s | 5h 18m 24s | 5m 28s | 18h 6m 50s |
| CoreBench Hard | 88 | 8 | 84,902,855 | $186.24 | 20,375,577 | $260.13 | 11,543,462 | 16,803,072 | $446.37 | 71h 13m 57s | 48m 34s | 29m 22s | 1h 31m 46s | 2m 56s | 9h 22m 2s |
| SciCode | 88 | 8 | 743,461,031 | $1504.12 | 196,829,922 | $2483.70 | 142,022,877 | 247,713,792 | $3987.81 | 417h 29m 15s | 4h 44m 39s | 2h 53m 12s | 7h 14m 28s | 11m 5s | 36h 14m 2s |
| ScienceAgentBench | 88 | 8 | 2,783,608 | $5.65 | 9,955,260 | $127.99 | 6,692,032 | 658,688 | $133.64 | 5h 24m 55s | 3m 42s | 2m 25s | 4m 34s | 18s | 37m 20s |

# Section 4.1 Cost of Evaluation

Trace unit: each JSON file is one complete model-on-benchmark run over that benchmark's full item set. The Section 4.1 economics below use the primary 8-model cohort shared across the four benchmarks.

## Full Evaluation Cost per New Model

| Benchmark | Full-run traces | Avg cost / run | Avg runtime / run |
| :--- | ---: | ---: | ---: |
| ColBench Backend | 432 | $47.88 | 2h 3m 28s |
| CoreBench Hard | 88 | $5.07 | 48m 34s |
| SciCode | 88 | $45.32 | 4h 44m 39s |
| ScienceAgentBench | 88 | $1.52 | 3m 42s |
| **Total 4-benchmark suite** |  | **$99.79** | **7h 40m 23s** |

## ARAF Setup Cost

Tau grid recovered from `model/reproduce.sh`: 107 values. Research-faithful count assumes 3 embeddings x 2 canonical Section 4.1 regimes x 107 tau values x 50 seeds.

| Setup version | Seed-config runs | GPU-hours | Dollars |
| :--- | ---: | ---: | ---: |
| Research-faithful | 32,100 | 802.500 | $240.75 |
| Deployment-faithful | 1 | 0.025 | $0.01 |

## Marginal Cost per New Model

Per-new-model ARAF cost uses `C_ARAF(p) = p * C_full_eval + C_inference`. `C_inference` is conservatively upper-bounded by one full ARAF training run: $0.01 and 1m 30s.

| Observed fraction p | Per-new-model cost | Per-new-model runtime | Break-even (research) | Break-even (deploy) |
| :---: | ---: | ---: | ---: | ---: |
| 0.3 | $29.94 | 2h 19m 37s | 3.45 | 0.0001 |
| 0.5 | $49.90 | 3h 51m 41s | 4.83 | 0.0002 |
| 0.7 | $69.86 | 5h 23m 46s | 8.04 | 0.0003 |

