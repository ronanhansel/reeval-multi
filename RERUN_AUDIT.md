# Corrected Rerun Audit

## Goal

Document the leakage-fix rerun process, the code changes made, the outputs refreshed, and the main remaining concern after the corrected reruns.

## Commands Used

The intended clean rerun commands were:

- `./model/reproduce.sh --full --parallel 100`
- `./model/reproduce.sh --full --sample-size-study --parallel 100`
- `./model/reproduce.sh --full --support-thinning-study --parallel 100`

The driver was changed so `--full` resets the active result directories for those commands instead of mixing corrected outputs with stale runs.

## What Was Fixed

### 1. Global PCA/SAE leakage

Problem:

- Item-holdout experiments were loading globally prefit `processed_embeddings/embeddings_pca_48.pkl` and `processed_embeddings/embeddings_sae_48.pkl`.
- That meant the learned embedding transform had already seen held-out items.

Fix:

- `model/amortized_irt.py` now carries raw item embeddings forward and fits PCA/SAE only inside the run-specific pruning path.
- The fit is done on the final retained train-item set for that run.

Key code:

- `prepare_experiment_data(...)`
- `fit_split_embeddings(...)`
- `prune_embedding_only_train_columns(...)`

### 2. Thinning-time embedding leakage

Problem:

- After support thinning, a train item could lose all retained observations but still survive as an embedding row.

Fix:

- Train columns with zero retained support are dropped from the active item set before the final PCA/SAE fit is built.

### 3. Wrong fit boundary for PCA/SAE

Problem:

- An intermediate version still fit PCA/SAE before final thinning-induced train-column removal.

Fix:

- The final version fits PCA/SAE only after the exact retained train-item set is known.

### 4. Result-driver issues

Problem:

- `model/reproduce.sh` launched `python model/amortized_irt.py` in a way that broke `from model...` imports.
- `--full --support-thinning-study` incorrectly pulled in sample-size cleanup/runs.

Fix:

- `reproduce.sh` now runs `PYTHONPATH=${REPO_ROOT} python -m model.amortized_irt`.
- Study-only full reruns no longer trigger unrelated studies.

### 5. Stability issues during reruns

Observed and fixed:

- missing `embedding_dim` parameter after deferred-fit refactor
- device mismatch in pruning helper when slicing tensors with boolean masks
- model input-dimension mismatch when `x_j` was raw `4096`-d instead of `48`-d
- non-finite split-refit embedding caches causing `Bernoulli(probs=nan)` failures

Fixes:

- explicit `embedding_dim` plumbing restored
- pruning masks are moved to the correct tensor device before indexing
- model input dimension now comes from the actual built `x_j`
- split-refit caches are validated for shape and finiteness before reuse
- embedding normalization sanitizes `NaN`/`Inf`

## Outputs Refreshed

Regenerated from corrected reruns:

- `paper/figures/sample_size_quad.pdf`
- `paper/figures/support_thinning.pdf`
- `paper/figures/refined_auc_comparison.pdf`
- `paper/figures/refined_rmse_comparison.pdf`
- `paper/figures/interpretability/best_auc_hybrid_stacked.pdf`
- `paper/figures/interpretability/tau_sensitivity_n32_k_only.pdf`
- `paper/figures/interpretability/loading_cleanliness_comparison.pdf`
- `paper/figures/6appendix/sensitivity_all_n32_merged_appendix.pdf`
- `paper/figures/6appendix/sensitivity_all_max_merged_appendix.pdf`
- `model/result/support_thinning_study/support_thinning_grid.csv`

Updated manuscript content:

- `paper/sections/4exp.tex`
  - main AUC table updated
  - robustness paragraph updated
  - remediation/repeated-sampling interpretation updated

## What Changed Empirically

### Main held-out AUC table

Corrected ranking:

- Pre Bernoulli: `kNN` best
- Post Bernoulli: `ARAF (PCA)` best
- Post Beta: `kNN` best

This differs materially from the old paper table and invalidates the earlier blanket claim that the strongest ARAF variants broadly match or exceed kNN.

### Sample-size study

- Best ARAF variant is consistently `raw` in the Beta scaling sweeps.
- kNN remains stronger than ARAF across both the varying-user and varying-item Beta sweeps.

### Support-thinning study

- Pre-binary: ARAF beats kNN throughout.
- Post-binary: ARAF wins at sparse support, kNN wins once support becomes moderate/dense.
- Post-beta RMSE: ARAF beats Rasch and MIRT throughout, but kNN wins once support becomes moderately dense.

## Remaining Red Flag

The corrected reruns removed the PCA/SAE leakage, but kNN improving after the fix is still suspicious.

The clearest remaining source of optimistic leakage is still in the kNN selection code:

- `model/amortized_irt.py:1714-1744`

`compute_best_knn_metrics(...)` evaluates multiple `k` values and chooses the best one using:

- held-out test RMSE
- held-out test AUC

That means `kNN` is still being tuned on the test set.

So even after the embedding-leakage fixes, the reported kNN numbers are still optimistic because the neighbor count is selected against the held-out items being reported.

This likely explains why kNN can remain extremely strong, and may explain why it appears even better relative to ARAF after the leakage fix: the ARAF side got stricter, while kNN still benefits from test-set model selection.

## Recommended Next Step

To make the comparison sound, move kNN hyperparameter selection off the test set.

Minimal next correction:

1. Split the observed train support into train/validation.
2. Choose `k` using only validation performance.
3. Report final kNN performance once on the held-out test items.

Until that is done, the corrected rerun should be interpreted as:

- PCA/SAE leakage fixed
- thinning-time embedding-only train columns fixed
- but kNN results still optimistic because `k` is test-tuned

## Practical Conclusion

The current corrected rerun is a major improvement over the original setup, but it is not the final fully clean comparison because kNN still has a test-time selection leak.
