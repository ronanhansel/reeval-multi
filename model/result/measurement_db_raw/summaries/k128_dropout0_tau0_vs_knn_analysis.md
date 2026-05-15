# K=128 Dropout 0.0 Tau 0 vs kNN Diagnostic Analysis

Selection note: this was motivated by the K=64 lower-bound tau result and uses diagnostic test metrics. It should be validated with a held-out validation selector before final paper claims.

## Paired Per-Seed Results

|     seed |   lambda_tau |   araf_latent_dim |   araf_dropout |   auc_amortized |   auc_knn |   rmse_amortized |   rmse_knn |   selected_knn_k |   auc_diff_araf_minus_knn |   rmse_diff_araf_minus_knn |
|---------:|-------------:|------------------:|---------------:|----------------:|----------:|-----------------:|-----------:|-----------------:|--------------------------:|---------------------------:|
| 0.000000 |     0.000000 |        128.000000 |       0.000000 |        0.827499 |  0.821105 |         0.398673 |   0.405230 |        10.000000 |                  0.006394 |                  -0.006557 |
| 1.000000 |     0.000000 |        128.000000 |       0.000000 |        0.824041 |  0.821672 |         0.401114 |   0.405161 |        10.000000 |                  0.002370 |                  -0.004047 |
| 2.000000 |     0.000000 |        128.000000 |       0.000000 |        0.823829 |  0.818480 |         0.399939 |   0.406019 |        10.000000 |                  0.005350 |                  -0.006080 |


## Paired Difference Tests

| metric   |   araf_mean |   knn_mean |   n |      mean |      std |       min |       max |   paired_t_stat |   paired_t_p_two_sided |   sign_test_p_two_sided |
|:---------|------------:|-----------:|----:|----------:|---------:|----------:|----------:|----------------:|-----------------------:|------------------------:|
| auc      |    0.825123 |   0.820419 |   3 |  0.004704 | 0.002088 |  0.002370 |  0.006394 |        3.901805 |               0.059849 |                0.250000 |
| rmse     |    0.399909 |   0.405470 |   3 | -0.005561 | 0.001333 | -0.006557 | -0.004047 |       -7.226554 |               0.018616 |                0.250000 |


Interpretation: K=128/dropout=0.0/tau=0 beats kNN on AUC for all three seeds. With n=3, paired t-test is borderline (p≈0.060) and exact sign test is p=0.25, so the direction is consistent but more split seeds or validation-selected confirmation are needed for a strong significance claim.
