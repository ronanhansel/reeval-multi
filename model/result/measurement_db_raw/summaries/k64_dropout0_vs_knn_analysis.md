# K=64 Dropout 0.0 vs kNN Diagnostic Analysis

Selection note: ARAF tau is selected by held-out test AUC in this diagnostic sweep, so ARAF numbers are optimistic relative to kNN validation-selected k.

## Paired Per-Seed Results

|     seed |   lambda_tau |   araf_latent_dim |   araf_dropout |   auc_amortized |   auc_knn |   rmse_amortized |   rmse_knn |   selected_knn_k |   auc_diff_araf_minus_knn |   rmse_diff_araf_minus_knn |
|---------:|-------------:|------------------:|---------------:|----------------:|----------:|-----------------:|-----------:|-----------------:|--------------------------:|---------------------------:|
| 0.000000 |     0.002000 |         64.000000 |       0.000000 |        0.820163 |  0.821105 |         0.402912 |   0.405230 |        10.000000 |                 -0.000942 |                  -0.002318 |
| 1.000000 |     0.002000 |         64.000000 |       0.000000 |        0.817187 |  0.821672 |         0.405157 |   0.405161 |        10.000000 |                 -0.004485 |                  -0.000004 |
| 2.000000 |     0.002000 |         64.000000 |       0.000000 |        0.815872 |  0.818480 |         0.404315 |   0.406019 |        10.000000 |                 -0.002608 |                  -0.001704 |


## Paired Difference Tests

| metric   |   araf_mean |   knn_mean |   n |      mean |      std |       min |       max |   paired_t_stat |   paired_t_p_two_sided |   sign_test_p_two_sided |
|:---------|------------:|-----------:|----:|----------:|---------:|----------:|----------:|----------------:|-----------------------:|------------------------:|
| auc      |    0.817740 |   0.820419 |   3 | -0.002678 | 0.001772 | -0.004485 | -0.000942 |       -2.617573 |               0.120196 |                0.250000 |
| rmse     |    0.404128 |   0.405470 |   3 | -0.001342 | 0.001199 | -0.002318 | -0.000004 |       -1.939236 |               0.192030 |                0.250000 |


Interpretation: K=64/dropout=0.0 does not beat kNN on AUC. It slightly improves RMSE, but with only 3 split seeds the paired evidence is not statistically significant by exact sign test or paired t-test.
