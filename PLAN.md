# Current Plan

## Goal

Correct embedding leakage in the amortized IRT rerun pipeline, remove stale affected outputs, and make the three full rerun commands work cleanly:

- `./model/reproduce.sh --full`
- `./model/reproduce.sh --full --sample-size-study`
- `./model/reproduce.sh --full --support-thinning-study`

## Tasks

- [x] Audit current leakage points in the item-holdout and embedding pipeline.
- [x] Decide the corrected protocol: refit PCA/SAE on train items only for each item split, cache transformed embeddings by split, and keep raw embeddings as frozen external features.
- [x] Patch `model/amortized_irt.py` to build leakage-safe split-specific embeddings and save enough metadata for downstream interpretability.
- [x] Patch plotting/utilities that assume global PCA/SAE embeddings so they stay aligned with the rerun outputs.
- [x] Patch `model/reproduce.sh` so the three full rerun commands automatically clear the affected result directories instead of mixing deprecated runs with corrected ones.
- [x] Verify the updated commands and plotting paths for likely runtime errors.
- [x] Drop zero-support train columns from the active item set during thinning-time runs so their embeddings are not available after retention masking.
- [x] Delete stale affected result outputs in `model/result/main`, `model/result/sample_size_study`, and `model/result/support_thinning_study` to force clean reruns.
- [x] Move PCA/SAE fitting to the final post-thinning active train-item set so the learned transform is not influenced by items later dropped for that run.
- [x] Re-verify caching and runtime stability after the deferred-fit refactor.
- [x] Fix reproduce entrypoint/import handling and study selection so study-only full reruns execute from repo root without triggering unrelated studies.
- [x] Harden split-refit embedding generation and cache validation against non-finite PCA/SAE outputs observed during the sample-size rerun.
- [x] Regenerate all paper figures and numbers affected by the corrected main, sample-size, and support-thinning reruns.
- [x] Compare updated sample-size/support-thinning results against the paper narrative and summarize the substantive differences.

## Leakage Notes

- Current leakage comes from using globally prefit `processed_embeddings/embeddings_pca_48.pkl` and `processed_embeddings/embeddings_sae_48.pkl` for item-holdout experiments.
- The corrected protocol should fit PCA/SAE on train-item raw embeddings only, then transform both train and held-out items with that train-only fit.
- Raw embeddings remain frozen side information, so they do not require refitting.
- The sample-size plotting code also had stale filename and baseline-cache assumptions; those were updated to read the dedicated study outputs produced by `reproduce.sh`.
- For support thinning, train columns whose retained support becomes empty should be removed from the active item set for that run rather than remaining as embedding-only columns.
- The stronger protocol should fit PCA/SAE only after the final active train-item set is known, not merely prune zero-support columns after a broader fit.
- In this workspace, the sample-size and support-thinning rerun outputs were available and replotted, but `model/result/main` is still missing the corrected rerun CSVs needed to refresh the main comparison table and interpretability figures.

## Follow-up Investigation

- [x] Explain why `kNN` improved while ARAF decreased after the held-out embedding leakage fix, and verify whether any remaining evaluation leakage or protocol asymmetry is still present.

## kNN Selection Fix

- [x] Patch `model/amortized_irt.py` so `kNN` selects `k` on a validation split carved from observed training support rather than the held-out test fold.
- [x] Identify and delete cached/result files affected by the old test-tuned `kNN` baseline so reruns regenerate them cleanly.
- [x] Verify the updated pipeline paths and summarize which reruns are now required.

## Result Recheck

- [x] Recheck the latest pulled rerun outputs and compare updated `kNN` versus `ARAF` behavior after the validation-only `k` selection fix.

## Summary Repair

- [x] Restore or regenerate the missing support-thinning summary outputs from the latest pulled results.
- [x] Patch the main summary/export layer so `kNN` appears in `comprehensive_results` alongside the amortized models and baselines.
- [x] Verify the repaired summaries against the underlying raw result CSVs.

## Reproduce Resume

- [x] Patch `model/reproduce.sh` so a continued full rerun auto-detects incomplete pulled support-thinning results and backfills the missing outputs.
- [x] Verify the updated resume detection logic matches the current missing-artifact pattern.
