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

## Leakage Notes

- Current leakage comes from using globally prefit `processed_embeddings/embeddings_pca_48.pkl` and `processed_embeddings/embeddings_sae_48.pkl` for item-holdout experiments.
- The corrected protocol should fit PCA/SAE on train-item raw embeddings only, then transform both train and held-out items with that train-only fit.
- Raw embeddings remain frozen side information, so they do not require refitting.
- The sample-size plotting code also had stale filename and baseline-cache assumptions; those were updated to read the dedicated study outputs produced by `reproduce.sh`.
- For support thinning, train columns whose retained support becomes empty should be removed from the active item set for that run rather than remaining as embedding-only columns.
