# TabPFN vs kNN/ARAF, 3 Seeds (1000 Training Rows)

TabPFN v2.2.1 local model, pairwise supervised framing, 64-dim PCA item features, 1000 training rows per seed.

## Results

| seed | TabPFN AUC | TabPFN RMSE | kNN AUC | kNN RMSE | ARAF AUC | ARAF RMSE |
|------|-----------|-------------|---------|----------|----------|-----------|
| 0    | 0.6938    | 0.4591      | 0.8211  | 0.4052   | 0.8275   | 0.3987    |
| 1    | 0.6886    | 0.4636      | 0.8217  | 0.4052   | 0.8240   | 0.4011    |
| 2    | 0.6866    | 0.4597      | 0.8185  | 0.4060   | 0.8238   | 0.3999    |

## Summary

- TabPFN (1k rows) AUC: mean 0.6896, gap -0.131 vs kNN, -0.136 vs ARAF.
- TabPFN (1k rows) RMSE: mean 0.4608, gap +0.055 vs kNN, +0.060 vs ARAF.
- TabPFN loses on all 3 seeds against both baselines.

## Notes

- Training row cap of 1,000 was used because the 10,000 and 50,000 row configurations timed out (>20 min per seed on A6000 GPU).
- TabPFN v2.2.1 officially supports up to 10,000 training samples; the 50k run used `ignore_pretraining_limits=True` but did not complete in 40+ minutes.
- The pairwise framing discards the matrix structure that ARAF and kNN exploit via item-user interactions.
- ARAF (K=128, dropout=0, tau=0) remains the strongest method on this benchmark.

## Artifacts

- TabPFN results: `model/result/measurement_db_raw/tabpfn/`
- Aggregated summaries: `model/result/measurement_db_raw/summaries/metrics_long.csv`
