# K=128/dropout=0/tau=0 ARAF vs RAW kNN, 20 Seeds

Fixed ARAF candidate: `araf_latent_dim=128`, `araf_dropout=0.0`, `lambda_tau=0.0`, RAW item embeddings. kNN uses the same RAW corpus/splits and validation-selected `k` from `5,10,20,50`; selected `k=10` for every seed in this set.

## Main result, seeds 0-19

- AUC: ARAF 0.824640 vs kNN 0.822717; mean diff +0.001922; wins 16/20; paired t p=0.017502; Wilcoxon p=0.017181; sign p=0.005909.
- RMSE: ARAF 0.400153 vs kNN 0.404006; mean diff -0.003853; wins 20/20; paired t p=2.19094e-08; Wilcoxon p=1.90735e-06; sign p=9.53674e-07.

## Out-of-selection result, seeds 3-19

This excludes seeds 0-2, which were used during the exploratory K/tau search.

- AUC: ARAF 0.824554 vs kNN 0.823123; mean diff +0.001432; wins 13/17; paired t p=0.090164; Wilcoxon p=0.079681; sign p=0.024521.
- RMSE: ARAF 0.400196 vs kNN 0.403748; mean diff -0.003552; wins 17/17; paired t p=5.58174e-07; Wilcoxon p=1.52588e-05; sign p=7.62939e-06.

## New confirmation block, seeds 10-19

- AUC: ARAF 0.823528 vs kNN 0.822396; mean diff +0.001132; wins 7/10; paired t p=0.365786; Wilcoxon p=0.275391; sign p=0.171875.
- RMSE: ARAF 0.400620 vs kNN 0.404267; mean diff -0.003648; wins 10/10; paired t p=0.00040147; Wilcoxon p=0.00195312; sign p=0.000976562.

## Interpretation

The final 20-seed result confirms a statistically significant AUC and RMSE advantage for the fixed K=128 ARAF candidate over validation-selected RAW kNN. The stronger publication-style check is the out-of-selection set, seeds 3-19; it remains directionally positive on AUC with 13/17 wins, but paired t/Wilcoxon p-values are above 0.05, while the one-sided sign test is below 0.05. RMSE remains clearly significant across all tests after excluding the three exploratory seeds.

Caveat: the fixed ARAF candidate was chosen from an exploratory sweep over K/dropout/tau. Report the candidate-selection process explicitly, or add a validation-based ARAF hyperparameter-selection layer if the claim needs fully symmetric model selection.

## Artifacts

- Paired rows: `k128_dropout0_tau0_vs_knn_20seed_paired.csv`
- Significance table: `k128_dropout0_tau0_vs_knn_20seed_significance.csv`
