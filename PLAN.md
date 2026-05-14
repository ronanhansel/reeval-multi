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

## Deep kNN vs ARAF Re-Audit

- [ ] Re-audit item split, embedding transform, cache keying, and baseline selection paths for remaining leakage.
- [ ] Recompute/summarize current raw outputs to locate where `kNN` beats ARAF and whether comparison is like-for-like.
- [ ] Stress-test assumptions with focused verification commands and independent subagent checks.
- [ ] Patch code or summaries only if concrete bug found.

## Measurement-DB RAW-Only Large Evaluation

Goal: build a larger measurement-db evaluation path that runs RAW item embeddings only for `kNN` and ARAF, with flat reusable config-keyed artifacts and enough structured metadata for later aggregation and plotting.

Disk check:

- [x] Confirm available workspace storage before planning large data outputs.
- Workspace path resolves to `/Data/home/v-qizhengli/workspace/reeval-multi`.
- `findmnt -T /Data/home/v-qizhengli/workspace/reeval-multi` shows the active filesystem is `/Data` on `/dev/nvme0n1p1` (`ext4`), not a smaller mounted drive.
- Usable filesystem for this repo: `/Data` has 13.9T total, 5.3T used, 7.9T available, about 38-41% used depending on `df`/`findmnt` rounding.
- `df -hT` reports the same `/Data` filesystem for `/Data/home/v-qizhengli/workspace/reeval-multi`, `/Data/home/v-qizhengli/workspace`, `/Data/home/v-qizhengli`, `/Data`, and `/home`.
- `quota -s` returned no quota output, so no user quota was detected by the standard quota command.
- Full repo `du` walk was stopped because generated artifacts made it slow; free filesystem space is sufficient for the planned large-db cache/results.

Implementation plan:

- [x] Start implementation of commandline-ready RAW-only measurement-db path.
- [x] Add stable result root `model/result/measurement_db_raw/` with `data_cache/`, `embeddings/`, `baselines/`, `araf/`, `summaries/`, and `logs/`.
- [x] Add `model/reproduce_large_db.sh` for the large-db RAW-only run.
- [x] Default large run settings: `conda activate hal`, `--parallel 100`, quiet mode, and full stdout/stderr redirected to log files under `model/result/measurement_db_raw/logs/`.
- [x] Support all source-buildable ready per-item measurement-db datasets, while recording skipped/failed datasets with reasons.
- [x] Generate RAW item embeddings from item contents; do not implement PCA or SAE for this path.
- [x] Run `kNN` as the baseline using RAW embeddings, `--baseline-profile knn_only`, and validation-only `k` selection over `5,10,20,50`.
- [x] Run ARAF with `--embedding-type raw` against the same corpus, split seed, test mask, and RAW embeddings used by kNN.
- [x] Use readable deterministic config slugs in filenames instead of hashes.
- [x] Save exact sidecar config JSON next to every corpus, embedding, kNN, ARAF, summary, and log artifact.
- [x] Skip repeated configs when artifact exists, sidecar config matches exactly, expected metric columns are present, and required metrics are non-null.
- [x] Mark partial or failed outputs with `.incomplete` or `.failed`; do not delete old artifacts automatically.

Aggregation and plotting requirements:

- [x] Write `summaries/result_manifest.csv` as the plot-ready index of completed artifacts.
- [x] Write `summaries/metrics_long.csv` with one row per method/config.
- [x] Write `summaries/metrics_wide.csv` joining kNN and ARAF on shared corpus, embedding, split, model type, and training settings.
- [x] Write `summaries/dataset_inventory_{corpus_slug}.csv` with dataset inclusion status, skip reason, matrix dimensions, density, response metadata, modality, and domain.
- [x] Write `summaries/split_inventory_{corpus_slug}_seed{seed}_test{test_size}.csv` with item ID, dataset, split, embedding availability, and observed count.
- [x] Ensure every result row includes `data_source`, `dataset_selector`, `dataset_names`, `corpus_slug`, `embedding_slug`, `method`, `embedding_type`, `model_type`, `seed`, `test_size`, `train_retention`, `n_samples`, `lambda_tau`, `epochs`, `knn_k_grid`, `selected_knn_k`, validation metrics when available, test metrics, corpus dimensions, split counts, artifact path, config path, log path, and status.

Acceptance checks:

- [ ] Smoke run: `bash model/reproduce_large_db.sh --datasets hle,bfcl --parallel 2 --epochs 5`.
- [ ] Rerun the same smoke command and verify corpus, embeddings, kNN, ARAF, and summaries skip or refresh only when needed.
- [x] Full RAW run: `bash model/reproduce_large_db.sh --all-ready --parallel 100 --quiet`.
- [ ] Verify no PCA/SAE files are generated by this path.
- [ ] Verify kNN never selects `k` on held-out test items.
- [ ] Verify RAW kNN and RAW ARAF share the same corpus, embeddings, split seed, and test mask.

Validation notes:

- [x] Full RAW run completed with kNN and ARAF results.
- [x] kNN AUC=0.823439, RMSE=0.402876, selected k=10.
- [x] ARAF AUC=0.788884, RMSE=0.418016.
- [x] Rerun properly skipped heavy work after config match.
- [x] Baseline manifest and mirt sweep manifest now materialized at config-keyed paths.
- [ ] Next: smoke run optional for verification.
- [x] `python -m py_compile model/measurement_db_raw.py model/amortized_irt.py` passes in `hal`.
- [x] `bash -n model/reproduce_large_db.sh` passes.
- [x] Synthetic measurement-db RAW corpus smoke test produced one kNN baseline row and one ARAF row through `--data-source measurement_db_raw`.
- [x] Synthetic aggregation smoke test produced `metrics_long.csv` with `kNN` and `ARAF` rows and `metrics_wide.csv` with joined `araf_auc/araf_rmse/knn_auc/knn_rmse` columns.

## Measurement-DB RAW Data Preparation Run

Goal: start the data preparation and RAW embedding phase only, so later runs can execute fitting without rebuilding corpus or embeddings.

- [x] Confirm no existing `model/result/measurement_db_raw` artifacts were present before starting.
- [x] Reconfirm usable storage for this repo: `/Data` has about 7.9T available.
- [x] Start HF-published measurement-db all-ready corpus preparation and RAW embedding generation only.
- [x] Fix HF metadata-array handling that initially caused datasets to be skipped.
- [x] Restart embedding with chunked/resumable parts after Qwen batch-size 8 OOM; final successful settings were `batch_size=1`, `chunk_size=128`, `max_chars=20000`.
- [x] Verify corpus parquet, item contents, dataset inventory, embedding pickle, config sidecars, and logs.
- [x] Provide the fitting-only command that consumes prepared `CORPUS_PATH` and `EMBEDDING_PATH`.

Prepared artifact summary:

- Corpus: `model/result/measurement_db_raw/data_cache/corpus_src-hf_all-ready_mins4_mini50_bounded01_mdb-raw-matrix-v1.parquet`
- Item contents: `model/result/measurement_db_raw/data_cache/item_contents_src-hf_all-ready_mins4_mini50_bounded01_mdb-raw-matrix-v1.csv`
- Dataset inventory: `model/result/measurement_db_raw/summaries/dataset_inventory_src-hf_all-ready_mins4_mini50_bounded01_mdb-raw-matrix-v1.csv`
- RAW embeddings: `model/result/measurement_db_raw/embeddings/rawemb-src-hf_all-ready_mins4_mini50_bounded01_mdb-raw-matrix-v1_model-qwen3-8b_maxchars20000_mdb-raw-embedding-v1.pkl`
- Split inventory: `model/result/measurement_db_raw/summaries/split_inventory_src-hf_all-ready_mins4_mini50_bounded01_mdb-raw-matrix-v1_seed42_test0-1.csv`
- Corpus shape: 858 subjects by 39,735 items, 3,429,086 observed responses.
- Embeddings: 39,735 rows, 4,096 dimensions, no missing or extra item IDs versus corpus columns.
- Split seed 42: 35,762 train items, 3,973 test items, 3,089,224 train observations, 339,862 test observations.
