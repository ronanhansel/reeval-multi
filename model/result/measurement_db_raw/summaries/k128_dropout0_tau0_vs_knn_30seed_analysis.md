# K=128/dropout=0/tau=0 ARAF vs RAW kNN, 30 Seeds

Fixed ARAF candidate: `araf_latent_dim=128`, `araf_dropout=0.0`, `lambda_tau=0.0`, RAW item embeddings. kNN uses the same RAW corpus/splits and validation-selected `k` from `5,10,20,50`; selected `k=10` for every seed in this set.

## Main result, seeds 0-29

- AUC: ARAF 0.825088 vs kNN 0.822304; mean diff +0.002784; wins 26/30; paired t p=8.74426e-05; Wilcoxon p=0.000123337; sign p=2.97381e-05.
- RMSE: ARAF 0.399942 vs kNN 0.404258; mean diff -0.004316; wins 30/30; paired t p=2.23864e-13; Wilcoxon p=1.86265e-09; sign p=9.31323e-10.

## Out-of-selection result, seeds 3-29

This excludes seeds 0-2, which were used during the exploratory K/tau search.

- AUC: ARAF 0.825084 vs kNN 0.822513; mean diff +0.002571; wins 23/27; paired t p=0.00059779; Wilcoxon p=0.000667766; sign p=0.000155374.
- RMSE: ARAF 0.399946 vs kNN 0.404123; mean diff -0.004178; wins 27/27; paired t p=9.55602e-12; Wilcoxon p=1.49012e-08; sign p=7.45058e-09.

## New confirmation block, seeds 20-29

- AUC: ARAF 0.825984 vs kNN 0.821477; mean diff +0.004507; wins 10/10; paired t p=0.000741269; Wilcoxon p=0.00195312; sign p=0.000976562.
- RMSE: ARAF 0.399519 vs kNN 0.404761; mean diff -0.005242; wins 10/10; paired t p=1.48101e-06; Wilcoxon p=0.00195312; sign p=0.000976562.

## Interpretation

The fixed K=128 ARAF candidate now beats validation-selected RAW kNN on AUC and RMSE across 30 split seeds. The out-of-selection set, seeds 3-29, is also significant on AUC by paired t-test, Wilcoxon, and sign test, and strongly significant on RMSE. This is the cleanest current fair claim because seeds 20-29 were added after the candidate was fixed and no new hyperparameter choice was made from their test metrics.

Caveat: the ARAF candidate itself was identified by an exploratory sweep over K/dropout/tau on earlier seeds. Report that process explicitly, or add a validation-selection layer for ARAF hyperparameters if a fully symmetric model-selection protocol is required.

## Research context

Primary-source literature supports using latent factorization models with side information for inductive/cold-start prediction. Jain and Dhillon's inductive matrix completion formulation studies prediction with side information for new users/items, and Knowledge Tracing Machines show factorization-machine style models can encompass multidimensional IRT while using side information at large scale. That aligns with the empirical direction here: increasing latent capacity and reducing regularization helped the side-information model over a local kNN embedding baseline.

## Artifacts

- Paired rows: `k128_dropout0_tau0_vs_knn_30seed_paired.csv`
- Significance table: `k128_dropout0_tau0_vs_knn_30seed_significance.csv`
