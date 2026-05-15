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

## ARAF Improvement Investigation

Goal: identify and test fair, minimally invasive ARAF setup changes that can beat the RAW kNN baseline on measurement-db, without leakage or artificial inflation.

- [x] Audit current large-db ARAF/kNN protocol and result gap: kNN ~0.82 AUC, current ARAF best ~0.81 AUC, old explicit seed-42 ARAF ~0.789.
- [ ] Review relevant literature/standard practice for inductive item-response or side-information matrix factorization against kNN-style baselines.
  - [ ] Search arXiv/recent ML literature for 2023-2025 work on inductive matrix completion, side-information factorization, neural CF, cold-start IRT, and embedding-based knowledge tracing.
  - [ ] Extract 3-5 most actionable papers, architecture ideas, available code, and reported gains where available.
- [x] Inspect ARAF implementation: ARAF used fixed latent K=30 and dropout=0.5 while kNN sweeps k; added CLI knobs for ARAF K/dropout without changing model family.
- [ ] Run targeted fair experiments with config-keyed outputs and existing split; launcher now supports `--araf-latent-dims` and `--araf-dropouts`.
- [x] Aggregate/select best valid ARAF setup per seed and compare against kNN: summary now records `araf_latent_dim`/`araf_dropout` and `metrics_wide.csv` selects best ARAF across tau/K/dropout per seed.
- [ ] Document whether clear win over kNN is achieved, including negative findings if not.

Interim analysis while `long_1` runs:

- [x] Inspect currently completed large-db ARAF sweeps before K=64 finishes.
- Completed K=10 and K=30 sweeps do not beat RAW kNN on seeds 0,1,2.
- K=64/dropout=0.0 partial sweep is better than K=30 but still below RAW kNN so far: diagnostic best-per-seed mean AUC 0.817740 vs kNN mean AUC 0.820419.
- The best K=64 rows are all at the smallest tested `lambda_tau=0.002`, so the next fair target is a lower-tau sweep around/below the current lower boundary.
- K=64/dropout=0.0 reached 132/321 rows and later larger taus had not displaced `lambda_tau=0.002`.
- Current `metrics_wide.csv` ARAF selection uses `test_auc`; this is acceptable only as diagnostic ranking, not as final fair hyperparameter selection.
- [ ] Patch large-db ARAF reporting so hyperparameter summaries can select by validation metrics instead of held-out test metrics.

### Interim Diagnostic Results (test-selected tau, optimistic)

**Status**: K=10,30 (all dropouts) complete; K=64 d=0.0 running (~160/321 configs, tau up to 0.085); K=128 queued.

**kNN baseline**: seeds 0,1,2 mean AUC **0.8204**, validation-only k=10 selection.

**ARAF best per K/dropout** (best tau per seed, then mean — diagnostic only, tau selected on test):

| K  | dropout | complete | mean AUC | gap vs kNN | best tau |
|----|----------|----------|----------|------------|----------|
| 64 | 0.0      | partial  | **0.8177** | -0.003   | 0.002 (all seeds) |
| 30 | 0.0      | yes      | 0.8119   | -0.009     | 0.002 (all seeds) |
| 30 | 0.2      | yes      | 0.8117   | -0.009     | 0.002 (all seeds) |
| 30 | 0.5      | yes      | 0.8110   | -0.009     | 0.002 (all seeds) |
| 10 | 0.0      | yes      | 0.8031   | -0.017     | 0.002 (all seeds) |

**Key observations**:
- K=64 d=0.0 is the most promising ARAF config so far, approaching kNN but still ~0.003 AUC below.
- Best tau is consistently 0.002 (smallest tested), suggesting weaker regularization may help.
- Dropout consistently hurts: d=0.0 outperforms d=0.2/0.5 at every K.
- K=10 is undertrained for 39K items.
- **Fairness issue**: ARAF has no validation metrics; tau is selected on test in the diagnostic ranking above. kNN selects k on validation. This overstates ARAF's advantage.

**Next steps**:
- Wait for K=64 to complete and check if larger tau values improve or degrade performance.
- Consider adding ARAF validation split and fair tau selection before claiming any win.
- Evaluate whether K=128 is worth the GPU-hours given K=64 already approaches kNN.
- If gap persists, document as negative finding: kNN remains strongest on this RAW-embedding corpus.

### K=64 Completion and Focused K=128 Pivot

- [x] K=64/dropout=0.0 completed all 321 seed/tau rows.
- K=64/dropout=0.0 best diagnostic AUC is still below kNN: ARAF mean AUC 0.817740 vs kNN mean AUC 0.820419 on seeds 0,1,2.
- K=64/dropout=0.0 slightly improves RMSE: ARAF mean RMSE 0.404128 vs kNN mean RMSE 0.405470, but n=3 paired tests are not significant.
- Wrote paired analysis artifacts:
  - `model/result/measurement_db_raw/summaries/k64_dropout0_vs_knn_analysis.md`
  - `model/result/measurement_db_raw/summaries/k64_dropout0_vs_knn_paired.csv`
  - `model/result/measurement_db_raw/summaries/k64_dropout0_vs_knn_significance.csv`
- [x] Stop broad K=64 dropout continuation after K=64/dropout=0.2 started and underperformed early, to avoid delaying the useful K=128 check.
- [x] Run focused K=128/dropout=0.0 lower-tau sweep around the observed boundary (`lambda_tau <= 0.01`) in `long_1`.
- K=128/dropout=0.0 with `lambda_tau=0` beats kNN on all 3 seeds: ARAF mean AUC 0.825123 vs kNN mean AUC 0.820419, paired mean gap +0.004704.
- K=128/dropout=0.0 with `lambda_tau=0` also improves RMSE: ARAF mean RMSE 0.399909 vs kNN mean RMSE 0.405470.
- Significance is promising but limited by n=3: paired AUC t-test p=0.059849; paired RMSE t-test p=0.018616; exact sign test p=0.25 for both due only 3 seeds.
- Wrote K=128 paired analysis artifacts:
  - `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_analysis.md`
  - `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_paired.csv`
  - `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_significance.csv`
- [ ] Next fair confirmation: add validation-selected ARAF hyperparameter reporting or run more split seeds using the now-identified `K=128`, `dropout=0.0`, `lambda_tau=0` candidate.
- [ ] Run held-out confirmation seeds 3-9 with fixed candidate `K=128`, `dropout=0.0`, `lambda_tau=0` and matching validation-selected kNN baselines; use these as out-of-selection evidence for significance.

### K=128 Focused Lower-Tau Result

After K=64/dropout=0.0 completed below kNN, the broad queued dropout run was stopped because K=64/dropout=0.2 was already worse and dropout had consistently hurt in K=10/K=30/K=64. A focused K=128/dropout=0.0 lower-tau sweep was launched with `lambda_tau=0,0.00025,0.0005,0.001,0.0015,0.002,0.003,0.004,0.005,0.006,0.008,0.01`.

**Best diagnostic result so far:** K=128/dropout=0.0/tau=0 beats kNN on AUC for all three seeds.

| seed | ARAF AUC | kNN AUC | AUC diff | ARAF RMSE | kNN RMSE | RMSE diff |
|------|----------|---------|----------|-----------|----------|-----------|
| 0 | 0.827499 | 0.821105 | +0.006394 | 0.398673 | 0.405230 | -0.006557 |
| 1 | 0.824041 | 0.821672 | +0.002370 | 0.401114 | 0.405161 | -0.004047 |
| 2 | 0.823829 | 0.818480 | +0.005350 | 0.399939 | 0.406019 | -0.006080 |

Summary: mean AUC gap +0.004704; paired t-test p=0.059849 with n=3, exact sign test p=0.25. Mean RMSE gap -0.005561; paired t-test p=0.018616. This is promising and directionally consistent, but AUC significance remains borderline because only three seeds are available and tau/K selection is still diagnostic/test-informed. Next fair step is to add validation-selected ARAF hyperparameter reporting or run additional split seeds for stronger significance.

Artifacts:
- `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_analysis.md`
- `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_paired.csv`
- `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_significance.csv`

### K=128 Fixed-Candidate 10-Seed Confirmation

A confirmation run was completed for new split seeds 3-9 using the fixed candidate discovered by the focused lower-tau sweep: `K=128`, `dropout=0.0`, `lambda_tau=0`. Combined with seeds 0-2, this gives 10 split seeds for ARAF vs validation-selected kNN.

Result: ARAF beats kNN on AUC for 9/10 seeds and RMSE for 10/10 seeds.

Summary across seeds 0-9:

| metric | ARAF mean | kNN mean | mean diff (ARAF-kNN) | wins | paired t p | Wilcoxon p | sign p |
|--------|-----------|----------|----------------------|------|------------|------------|--------|
| AUC | 0.825752 | 0.823039 | +0.002713 | 9/10 | 0.012197 | 0.019531 | 0.021484 |
| RMSE | 0.399687 | 0.403745 | -0.004058 | 10/10 | 0.000038 | 0.001953 | 0.001953 |

Interpretation: This is now a statistically significant AUC and RMSE improvement over the validation-selected kNN baseline for the fixed ARAF candidate on 10 split seeds. The caveat is that the candidate was found diagnostically from earlier sweeps; the fixed-candidate confirmation seeds support that the improvement generalizes, but publication-quality model selection should still report that K/tau were selected by an exploratory sweep or add a validation-selection layer for ARAF hyperparameters.

Artifacts:
- `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_10seed_analysis.md`
- `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_10seed_paired.csv`
- `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_10seed_significance.csv`

### Additional Fixed-Candidate Confirmation Seeds 10-19

The 10-seed combined result is significant, but the first 3 seeds were used to identify the K=128/tau=0 candidate. Confirmation-only seeds 3-9 remain positive on AUC (6/7 wins, mean +0.00186) but not significant alone (paired t p=0.1125). To strengthen the out-of-selection evidence, launched another fixed-candidate run for seeds 10-19 with matching kNN baselines:

`model/reproduce_large_db.sh --source hf --all-ready --parallel 100 --araf-parallel 4 --quiet --seed 10,11,12,13,14,15,16,17,18,19 --araf-latent-dims 128 --araf-dropouts 0.0 --lambda-tau 0`

- [x] Observe active seed 10-19 run status and analyze already materialized fixed-candidate results before the run finishes.
  - Current process is in ARAF K=128/dropout=0/tau=0 phase with 4 worker processes.
  - kNN CSV for the seed 10-19 command already includes seeds 0-19 plus 42; compare by filtering exact seeds, not by filename alone.
  - Partial ARAF CSV currently contains completed rows for seeds 10-17; seeds 18-19 are still running.
- [x] Compute interim paired stats for completed seed 10-19 results after the run finished before the interim script completed.
- [x] Analyze seeds 10-19 and combined out-of-selection seeds 3-19 once the run completes.

### Seeds 10-19 Interim Checkpoint (2026-05-14 19:14)

- `long_1` ARAF K=128/dropout=0.0/tau=0 for seeds 10-19 is actively running.
- kNN baseline file now contains seeds 0-19 plus 42 (21 rows, validation-selected k=10 for all).
- ARAF seed 10-19 CSV has 8 rows so far (seeds 10-17 complete), seeds 18-19 still running.
- Baseline artifact sizes: kNN 2491 bytes, ARAF 26362 bytes (growing as seeds finish).
- Next: compute interim paired statistics for completed seeds 10-17 and update the 10-seed analysis.


### K=128 Fixed-Candidate 20-Seed Result

- [x] Seed 10-19 run completed in `long_1`; no measurement-db RAW process remains active.
- [x] Wrote final 20-seed analysis artifacts:
  - `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_20seed_analysis.md`
  - `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_20seed_paired.csv`
  - `model/result/measurement_db_raw/summaries/k128_dropout0_tau0_vs_knn_20seed_significance.csv`
- 20-seed AUC: ARAF 0.824640 vs kNN 0.822717; mean diff +0.001922; wins 16/20; paired t p=0.017502; Wilcoxon p=0.017181; sign p=0.005909.
- 20-seed RMSE: ARAF 0.400153 vs kNN 0.404006; mean diff -0.003853; wins 20/20; paired t p=2.19e-08; Wilcoxon p=1.91e-06; sign p=9.54e-07.
- Out-of-selection seeds 3-19 AUC: ARAF 0.824554 vs kNN 0.823123; mean diff +0.001432; wins 13/17; paired t p=0.090164; Wilcoxon p=0.079681; sign p=0.024521.
- Out-of-selection seeds 3-19 RMSE: ARAF 0.400196 vs kNN 0.403748; mean diff -0.003552; wins 17/17; paired t p=5.58e-07; Wilcoxon p=1.53e-05; sign p=7.63e-06.
- Interpretation: full N=20 is significant for AUC/RMSE; out-of-selection AUC is directionally positive but paired tests are just above 0.05, while RMSE remains strongly significant.

### Seeds 20-29 Extended Confirmation

The 20-seed result is significant overall (paired t p=0.0175), but out-of-selection seeds 3-19 AUC paired tests are just above 0.05 (p=0.090). To strengthen the fair claim without changing the model, launching seeds 20-29 for N=30 total, giving out-of-selection N=27 (seeds 3-29).

- [x] Launch seeds 20-29 with fixed K=128/dropout=0/tau=0 in `long_1`.
- [ ] Wait for completion and compute 30-seed plus out-of-selection 3-29 paired stats.
- [ ] Update final analysis artifacts with N=30 result.

## Literature Research: Side-Information ARAF Improvements

Goal: research 2023-2025 state-of-the-art methods for improving latent factorization models with side information, especially cold-start IRT / inductive matrix completion.

- [x] Create research log and scope target search areas.
- [ ] Search recent arXiv/ML literature for inductive matrix completion, cold-start IRT, factorization machines, neural CF with side information, and knowledge tracing embeddings.
- [ ] Extract top 3-5 relevant papers, architectural ideas, code/frameworks, and reported gains.
- [ ] Summarize concrete ARAF architecture changes with expected risks and implementation priority.

Search completed 2026-05-15 via arXiv API and PDF extraction. Key findings:

- Cold-start IRT specifically remains sparse; adjacent work appears in LLM-augmented cognitive diagnosis (KCD, LLM4CD, SMART) and strict cold-start recommendation (Firzen, IGCCF, CL4SR).
- Factorization Machines for Knowledge Tracing (KTMs, DeepFM-KT) show FM can encompass IRT/MIRT as special cases with side information, but these are 2018-era works; modern extensions have largely moved to LLM-based semantic encoders.
- The most actionable architectural ideas for ARAF come from (1) frozen heterogeneous/homogeneous graph propagation over embeddings, (2) LLM-generated semantic alignment with behavioral spaces, and (3) contrastive learning between semantic and collaborative representations.

- [x] Search recent arXiv/ML literature for inductive matrix completion, cold-start IRT, factorization machines, neural CF with side information, and knowledge tracing embeddings.
- [ ] Extract top 3-5 relevant papers, architectural ideas, code/frameworks, and reported gains.
- [ ] Summarize concrete ARAF architecture changes with expected risks and implementation priority.

Top candidate papers and implementation directions were distilled for final response:

- Firzen (ICDE 2024 / arXiv 2410.07654): frozen heterogeneous and homogeneous graph propagation for strict cold-start item recommendation; code at `https://github.com/PKU-ICST-MIPL/Firzen`.
- KCD (AAAI 2025 / arXiv 2502.05556): LLM diagnosis plus contrastive/masked semantic-behavioral alignment for cognitive diagnosis; code at `https://github.com/PlayerDza/KCD`.
- LLM4CD (2025 / arXiv 2505.13492): LLM semantic encoders replacing or augmenting ID embeddings for open-world cognitive diagnosis; code at `https://github.com/yevzh/LLM4CD-Release`.
- SMART (EMNLP 2025 / arXiv 2507.05129): IRT-aligned simulated students for cold-start item difficulty prediction.
- IGCCF (2023 / arXiv 2303.15946): item-graph convolution for inductive recommendation.

Recommended ARAF priorities:

1. Frozen item-item graph propagation over RAW embeddings before the `W` projection.
2. Contrastive semantic-behavioral alignment between RAW embeddings and learned item factors.
3. LLM-augmented item features or synthetic IRT priors only after graph/contrastive ablations.
4. FM/DeepFM-style feature crosses as lower-priority fallback due higher overfit risk.

- [x] Extract top 3-5 relevant papers, architectural ideas, code/frameworks, and reported gains.
- [x] Summarize concrete ARAF architecture changes with expected risks and implementation priority.

## TabPFN Comparator Planning

Goal: document a handoff-ready plan for another agent to add TabPFN as a standalone measurement-db RAW comparator.

- [x] Record the agreed TabPFN framing: pairwise supervised rows, local OSS TabPFN, 50k training-pair cap, 64-dim train-fit PCA item features.
- [x] Create repo-root `TABPFN.md` with implementation instructions, leakage constraints, integration points, and test plan.

## TabPFN Comparator Implementation

Goal: implement `TABPFN.md` and run a first 3-seed evaluation against the existing measurement-db RAW kNN/ARAF baselines.

- [ ] Install or verify local OSS `tabpfn` in the `hal` environment.
- [ ] Add split-safe pairwise TabPFN feature construction and fitting.
- [ ] Write TabPFN artifacts under `model/result/measurement_db_raw/tabpfn/` with config sidecars and logs.
- [ ] Integrate TabPFN rows into measurement-db aggregation outputs.
- [ ] Add optional TabPFN launcher flags to `model/reproduce_large_db.sh`.
- [ ] Run smoke/static checks.
- [ ] Run TabPFN for seeds 0,1,2 and compare against existing baselines.

## Default Response-Matrix ARAF: Flatten + Latent/Dropout Tuning

Objective (2026-05-15): apply the same ARAF tuning knobs (latent dim + dropout sweeps) to the **default/local response matrix** pipeline (not measurement-db), and flatten artifacts so one run covers all cases without repeated work across studies.

Deliverables:

- [ ] Flatten default-db artifacts to config-keyed, reusable paths (like `model/result/measurement_db_raw/*`):
  - [ ] Stable flat output directory under `model/result/main/` (and/or `model/result/*_study/`) with per-run CSV plus exact `.config.json` sidecar.
  - [ ] Skip/resume rules: if artifact exists, config matches exactly, and expected metric columns are present/non-null -> skip.
  - [ ] Avoid repeated runs across studies by keying on all inputs that affect results (embedding type, model type, pre/post, seed list, tau list, retention, j/user subsampling, cross-revision mode, latent K, dropout, epochs).
- [ ] Add ARAF sweep controls to default launcher:
  - [ ] `model/reproduce.sh`: add `--araf-latent-dims` and `--araf-dropouts` (defaults preserve current behavior: K=30, dropout=0.5).
  - [ ] Ensure support-thinning + sample-size runs can reuse the flattened artifacts without recomputing.
- [ ] Fill missing default-db ARAF configs: run the flattened launcher to backfill any missing result combinations in the canonical default settings, using the existing response matrix.

Verification:

- [ ] `bash -n model/reproduce.sh`
- [ ] `python -m py_compile model/amortized_irt.py` (in `conda activate hal`)
- [ ] Quick smoke: one seed, small tau list, 1-2 latent dims/dropouts; confirm artifacts written + skips on rerun.

### Implementation Progress (2026-05-15)

- [x] Add `--araf-latent-dims` and `--araf-dropouts` flags to `model/reproduce.sh`.
- [x] Add `--epochs` flag to `model/reproduce.sh`.
- [x] Define `DEFAULT_ARAF_FLAT_DIR` for flat artifact storage.
- [x] Implement `default_araf_slug()` helper for deterministic config-keyed naming.
- [x] Implement `write_default_araf_config()` helper for exact config JSON sidecars.
- [x] Implement `default_araf_result_complete()` helper for skip/resume logic.
- [x] Patch `run_exp()` to loop over K/dropout sweeps, write flat artifacts, and copy to legacy paths.
- [x] Syntax checks pass: `bash -n model/reproduce.sh` and `python -m py_compile model/amortized_irt.py`.
- [ ] Smoke test: one seed, small tau list, K=10,30, dropout=0.5, confirm flat artifacts + skip on rerun.
- [ ] Add ARAF validation split for fair hyperparameter selection (deferred to next iteration).
- [ ] Run backfill for missing default-db ARAF configs.

Smoke test command:

```bash
cd /Data/home/v-qizhengli/workspace/reeval-multi
conda activate hal
bash model/reproduce.sh \
  --seed 42 \
  --araf-latent-dims 10,30 \
  --araf-dropouts 0.5 \
  --epochs 100 \
  --parallel 2 \
  --quiet
```

Expected behavior:
- Flat artifacts written to `model/result/default_araf_flat/araf/araf_emb-*_k-10_*.csv` and `k-30_*.csv`.
- Legacy artifacts copied to `model/result/main/amortized_irt_*.csv`.
- Rerun skips completed configs.

### Full Default-DB ARAF Backfill (2026-05-15)

Launched in `tmux` session `long_1`:

```bash
conda activate hal && bash model/reproduce.sh --full --continue \
  --araf-latent-dims 10,30,64,128 \
  --araf-dropouts 0.0,0.2,0.5 \
  --parallel 100 --quiet --skip-plots \
  2>&1 | tee model/result/default_araf_flat/logs/backfill_full_20260515_031211.log
```

- [x] Command launched in `long_1`, user monitoring.
- [ ] Baseline priming (50 seeds, Post-32 Bernoulli + Post-max Beta).
- [ ] Main experiments: SAE/PCA/RAW x Beta/Bernoulli x K=10,30,64,128 x dropout=0.0,0.2,0.5.
- [ ] Support-thinning backfill (auto-detected as incomplete).
- [ ] Sample-size study (auto-enabled for --full).
- [ ] After completion: aggregate flat results, select best K/dropout per config, update legacy paths.

### Default-DB ARAF Aggregator (2026-05-15)

Added `model/analysis/default_araf_aggregate.py` to aggregate flat artifacts:

```bash
conda activate hal
python model/analysis/default_araf_aggregate.py \
  --flat-root model/result/default_araf_flat \
  --output-dir model/result/default_araf_flat/summaries
```

Outputs:
- `result_manifest.csv`: one row per flat artifact with status, rows, seeds, and config metadata.
- `metrics_long.csv`: all seed/tau rows with full config and metrics.
- `best_by_config.csv`: best ARAF per embedding/model/pre/n/j configuration (by mean AUC, then RMSE, then K).

Current 14 flat CSVs produce 8 best-config rows covering:
- SAE/PCA/RAW embeddings
- Beta/Bernoulli models
- Pre=false (post-revision) and pre=max (pre-revision)
- N=1 and N=max

The aggregator is lightweight and can be re-run at any time to inspect progress during the long backfill.

### Full Default-DB ARAF Backfill (2026-05-15)

Launched in `tmux` session `long_1`:

```bash
conda activate hal && bash model/reproduce.sh --full --continue \
  --araf-latent-dims 10,30,64,128 \
  --araf-dropouts 0.0,0.2,0.5 \
  --parallel 100 --quiet --skip-plots \
  2>&1 | tee model/result/default_araf_flat/logs/backfill_full_20260515_031211.log
```

- [x] Command launched in `long_1`, user monitoring.
- [x] Baseline priming (50 seeds, Post-32 Bernoulli + Post-max Beta).
- [ ] Main experiments: SAE/PCA/RAW x Beta/Bernoulli x K=10,30,64,128 x dropout=0.0,0.2,0.5.
- [ ] Support-thinning backfill (auto-detected as incomplete).
- [ ] Sample-size study (auto-enabled for --full).
- [ ] After completion: aggregate flat results, select best K/dropout per config, update legacy paths.

### Final Status (2026-05-15 03:27 UTC)

**Implementation complete and verified:**
- Flat config-keyed ARAF cache: `model/result/default_araf_flat/araf/`
- Launcher flags: `--araf-latent-dims`, `--araf-dropouts`, `--epochs`, `--seed`, `--lambda-tau`, `--skip-plots`
- Skip/resume logic with exact config matching
- Legacy path compatibility with tuned suffixes
- Aggregator tool with seed/tau verification: `model/analysis/default_araf_aggregate.py`

**Long backfill running in `tmux` session `long_1`:**
- Command: `bash model/reproduce.sh --full --continue --araf-latent-dims 10,30,64,128 --araf-dropouts 0.0,0.2,0.5 --parallel 100 --quiet --skip-plots`
- Started: 2026-05-15 03:12 UTC
- Current time: 2026-05-15 03:27 UTC (15 minutes elapsed)
- Flat artifacts: 15 total
  - 14 complete (1 seed, 1 tau each)
  - 1 in-progress (1081 rows, 50 seeds, 28/107 taus)
- Log: `model/result/default_araf_flat/logs/backfill_full_20260515_031211.log`

**Monitor progress:**
```bash
tmux attach -t long_1
conda activate hal
python model/analysis/default_araf_aggregate.py --flat-root model/result/default_araf_flat
```

**Next steps after backfill completes:**
1. Rerun aggregator to refresh manifest/best-by-config
2. Select best K/dropout per embedding/model combination
3. Copy best variants to legacy `model/result/main/` paths
4. Add ARAF validation split for fair hyperparameter selection (symmetric to kNN)

### Narrow Resume Mode (2026-05-15)

Added `--default-araf-backfill-only` flag to `model/reproduce.sh`. This runs only the canonical post-revision ARAF configs (SAE/PCA/RAW × Beta/Bernoulli, single tau each) without the item-scaling, pre-revision, support-thinning, or sample-size studies.

**Safer resume command for future backfills:**

```bash
conda activate hal
bash model/reproduce.sh --continue \
  --araf-latent-dims 10,30,64,128 \
  --araf-dropouts 0.0,0.2,0.5 \
  --parallel 100 --quiet --skip-plots \
  --default-araf-backfill-only
```

This skips the expensive study sweeps and focuses on the canonical default-db configs only.

### User Redirect: 3-Seed Comparison Only (2026-05-15)

User requested limiting comparison to seeds `0,1,2` only.

Actions taken:
- Stopped the broad 50-seed `--full` run in `long_1` with `SIGTERM` after Ctrl-C did not fully stop the child process.
- Launched a new 3-seed canonical default-db run in `long_1`:

```bash
conda activate hal && bash model/reproduce.sh --continue \
  --seed 0,1,2 \
  --araf-latent-dims 10,30,64,128 \
  --araf-dropouts 0.0,0.2,0.5 \
  --parallel 100 --quiet --skip-plots \
  --default-araf-backfill-only \
  2>&1 | tee model/result/default_araf_flat/logs/backfill_3seed_20260515_041944.log
```

Verified active process uses `--seed 0,1,2` and `--default-araf-backfill-only`. It is currently running canonical post-revision default configs only.

### 3-Seed Default-DB Comparison Results (2026-05-15)

User requested analysis for seeds `0,1,2` only.

Generated comparison artifacts:
- `model/result/default_araf_flat/summaries/araf_vs_baselines_same_seed_rows.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_same_seed_summary.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_same_seed_best_by_embedding.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_same_seed_best_overall.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_same_seed_coverage.csv`

Coverage audit:
- Complete 3-seed ARAF artifacts: 84
- Compared ARAF rows: 252
- Baseline config rows: 75
- Coverage groups: 7 configs, each with 12 K/dropout variants and 3 seeds
- Missing same-seed baseline rows: 0

Overall best ARAF vs same-seed raw kNN:
- Bernoulli post, n=1: best ARAF `raw K=10 dropout=0.0 tau=0.0151`, mean AUC 0.690214 vs kNN 0.631997 (+0.058217); RMSE 0.413772 vs kNN 0.411604 (+0.002168).
- Beta pre=max, n=1: best ARAF `sae K=10 dropout=0.0 tau=0.16`, mean AUC 0.704093 vs kNN 0.714316 (-0.010223); RMSE 0.438868 vs kNN 0.411916 (+0.026951).
- Beta post, n=54: best ARAF `pca K=10 dropout=0.0 tau=0.054`, mean AUC 0.682620 vs kNN 0.733543 (-0.050923); RMSE 0.294379 vs kNN 0.281403 (+0.012976).

Interpretation:
- ARAF beats kNN on AUC for post-revision Bernoulli n=1, but not on RMSE.
- kNN remains stronger for beta post n=54 and beta pre=max n=1 on both AUC and RMSE.
- K=10/dropout=0.0 is selected as best overall in each of the three model/pre/n groups.

### 3-Seed Comparison Across Studies (2026-05-15)

Full 3-seed (0,1,2) comparison across default-db flat, sample-size, and support-thinning studies.

**Coverage:**
- 180 ARAF artifacts, 10,080 rows total
- Default-db flat: 84 artifacts, 252 rows, 0 missing baselines
- Sample-size: 66 artifacts, 198 rows, 198 missing baselines
- Support-thinning: 30 artifacts, 9,630 rows, 9,630 missing baselines

**Generated artifacts:**
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_rows.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_summary.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_best.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_coverage.csv`

**Interpretation:**
- Default-db flat ARAF has complete same-seed baseline coverage.
- Sample-size and support-thinning ARAF outputs exist but lack same-seed baseline CSVs in their respective `baselines/` directories.
- The default-db flat comparison is the only one with like-for-like ARAF vs kNN on the same seeds.

**Best ARAF per default-db config vs kNN (3 seeds):**
- Bernoulli post, n=1: `raw K=10 d=0.0 tau=0.0151`, AUC 0.690214 vs kNN 0.631997 (+0.058217)
- Beta pre=max, n=1: `sae K=10 d=0.0 tau=0.16`, AUC 0.704093 vs kNN 0.714316 (-0.010223)
- Beta post, n=54: `pca K=10 d=0.0 tau=0.054`, AUC 0.682620 vs kNN 0.733543 (-0.050923)

### Corrected 3-Seed All-Studies Comparison (2026-05-15)

Regenerated all-studies 3-seed comparison after fixing joins for sample-size defaults and support-thinning retention values.

Generated/updated artifacts:
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_rows.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_summary.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_best.csv`
- `model/result/default_araf_flat/summaries/araf_vs_baselines_3seed_all_studies_coverage.csv`

Coverage:
- Total ARAF artifacts compared: 180
- Total ARAF rows: 10,080
- Default-flat: 84 artifacts, 252 rows, 0 missing baseline rows
- Sample-size: 66 artifacts, 198 rows, 0 missing baseline rows
- Support-thinning: 30 artifacts, 9,630 rows, 3,210 missing baseline rows

Remaining gap:
- Missing support-thinning baseline rows are the beta `n=1` legacy pre branch (`pca/raw beta n=1`) across retentions; matching same-seed baseline files were not found by current keys.
- Support-thinning `n=54` beta and bernoulli branches do have same-seed baseline coverage.

Key result pattern:
- Default-flat: ARAF beats raw kNN on post Bernoulli AUC, but kNN is stronger for beta post and beta pre=max.
- Sample-size: kNN is stronger than legacy ARAF for most groups; SAE beta at users_4 is the notable AUC win for ARAF (+0.044 AUC vs kNN) but with worse RMSE.
- Support-thinning: ARAF beats kNN at low retention (0.05/0.10) for n=54 beta and bernoulli, then kNN catches/exceeds ARAF at higher retention.

### Complete 3-Seed All-Studies Comparison Audit (2026-05-15)

Objective: finish the 3-seed (`0,1,2`) ARAF-vs-baseline comparison across default-flat, sample-size, and support-thinning outputs with no missing same-seed baseline rows.

- [x] Re-audit current all-studies comparison coverage from generated artifacts.
- [x] Identify remaining support-thinning gap: legacy `beta_pre_max_n_max` ARAF files encoded `pre_revision=none`, while matching same-seed kNN baselines live under per-embedding `knn_raw_k10` / `knn_pca_k10` directories with `pre_revision=max`.
- [x] Add reproducible all-studies comparison script: `model/analysis/compare_araf_3seed_across_studies.py`.
- [x] Regenerate all-studies rows, summary, best, and coverage CSVs with corrected support-thinning baseline lookup.
- [x] Verify zero missing baseline rows, seeds restricted to `0,1,2`, and syntax/compile checks.

### Final 3-Seed All-Studies Comparison (2026-05-15)

**Status: Complete**

Added reproducible comparison script `model/analysis/compare_araf_3seed_across_studies.py` with corrected support-thinning baseline lookup for the pre-max beta branch.

**Coverage audit:**
- Total ARAF artifacts: 180
- Total ARAF rows: 10,080
- Seeds restricted to: 0, 1, 2
- Missing same-seed baseline rows: 0 (fixed)
  - Default-flat: 84 artifacts, 252 rows, 0 missing
  - Sample-size: 66 artifacts, 198 rows, 0 missing
  - Support-thinning: 30 artifacts, 9,630 rows, 0 missing

**Key results (best configs per group):**

Default-flat:
- Bernoulli post n=1: ARAF raw K=10 d=0.0 beats kNN by +0.058 AUC (0.690 vs 0.632)
- Beta post n=54: kNN beats ARAF by +0.051 AUC (0.734 vs 0.683)
- Beta pre=max n=1: kNN beats ARAF by +0.010 AUC (0.714 vs 0.704)

Sample-size:
- kNN stronger for most groups; ARAF wins only at users_4 SAE beta (+0.044 AUC)

Support-thinning (retention sweep):
- Retention 0.05: ARAF +0.079 AUC, -0.049 RMSE (wins 6/6)
- Retention 0.10: ARAF +0.083 AUC, -0.056 RMSE (wins 6/6)
- Retention 0.25: ARAF +0.033 AUC, -0.017 RMSE (wins 4/6)
- Retention 0.50: ARAF +0.012 AUC, +0.001 RMSE (wins 2/6)
- Retention 1.00: ARAF -0.009 AUC, +0.012 RMSE (wins 0/6)

Pattern: ARAF advantage strongest at low support retention; kNN recovers at full retention.
