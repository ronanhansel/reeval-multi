# TabPFN Measurement-DB Comparator Plan

## Summary

Add TabPFN as a **new comparator method** for the measurement-db RAW evaluation pipeline, alongside kNN and ARAF. TabPFN should be evaluated on the same held-out item split seeds, using a pairwise supervised table where each observed `(user, item)` response becomes a row with split-safe features.

Use local OSS `tabpfn` under `conda activate hal`, with a conservative first configuration of **50k training rows per seed** and **64 PCA item-embedding dimensions**.

## Decisions Already Made

- Method role: standalone comparator, not an ARAF ensemble or replacement.
- Framing: pairwise supervised meta-model.
- Package target: local OSS `tabpfn`, not an external API.
- Training-row cap: 50k observed train pairs per seed.
- Item embedding reduction: 64 train-fit PCA dimensions.
- Evaluation: same held-out item splits and metrics as kNN/ARAF.

## Modeling Design

Each training row represents one observed response pair:

- Label: binary response `y[user, item]`.
- Item features: train-fit PCA projection of RAW item embeddings, 64 dims.
- User features: split-safe user summary statistics computed from train items only, such as user mean, observed count, response variance, and optionally coarse quantiles.
- Item support features: for train items only, item mean, observed count, and variance computed from train observations.
- Held-out test items: use embedding-derived features and global fallback values only; do not use held-out item response aggregates.

Fit TabPFN on a deterministic stratified sample of up to 50k observed train pairs per seed. Predict all observed held-out test pairs in batches and compute AUC/RMSE using the same `evaluate_auc` and `compute_rmse` utilities as kNN/ARAF.

## Leakage Rules

- Do not use held-out item labels in features.
- Do not use held-out item response means, counts, variances, or other response aggregates.
- Do not use test-mask information as a model feature.
- Fit item PCA on train items only.
- Compute user statistics from train-item observations only.
- Sample TabPFN training rows only from observed train-item pairs.

## Implementation Changes

- Add a TabPFN result path under `model/result/measurement_db_raw/tabpfn/`, with config sidecars, logs, and skip/resume behavior matching the existing `araf/` and `baselines/` paths.
- Implement TabPFN either by extending `model/measurement_db_raw.py` with a `tabpfn` run mode or by adding a focused helper module such as `model/measurement_db_tabpfn.py` if that keeps the integration cleaner.
- Extend aggregation so `metrics_long.csv`, `result_manifest.csv`, and `metrics_wide.csv` include `method=tabpfn`.
- Add TabPFN metadata columns:
  - `tabpfn_train_rows`
  - `tabpfn_item_pca_dim`
  - `tabpfn_feature_set`
  - `tabpfn_package_version`
  - `test_auc`
  - `test_rmse`
  - `artifact_path`
  - `config_path`
  - `log_path`
- Extend `model/reproduce_large_db.sh` with optional flags:
  - `--tabpfn`
  - `--tabpfn-train-rows 50000`
  - `--tabpfn-item-pca-dim 64`
  - `--tabpfn-seeds 0,1,2`

## Suggested Artifact Naming

- Result CSV: `model/result/measurement_db_raw/tabpfn/tabpfn_raw_corpus-{corpus_slug}-seed-{seed}-rows-{train_rows}-pca-{pca_dim}.csv`
- Config sidecar: same path with `.config.json`
- Log: `model/result/measurement_db_raw/logs/tabpfn_corpus-{corpus_slug}-seed-{seed}.log`

The result CSV should include:

- `seed`
- `model_type`
- `test_size`
- `train_retention`
- `n_samples`
- `tabpfn_train_rows`
- `tabpfn_item_pca_dim`
- `tabpfn_feature_set`
- `tabpfn_package_version`
- `test_auc`
- `test_rmse`
- `train_item_count`
- `test_item_count`
- `train_observed_count`
- `test_observed_count`

## Dependency Check

Run all Python commands through the `hal` environment:

```bash
conda run -n hal python -c "import tabpfn; print(tabpfn.__version__)"
```

If missing, install in `hal` and record the version:

```bash
conda activate hal
pip install tabpfn
```

## Test Plan

Smoke test:

- Create or reuse a tiny synthetic measurement-db RAW corpus.
- Run TabPFN on one seed with `--tabpfn-train-rows 1000 --tabpfn-item-pca-dim 16`.
- Verify the result CSV has non-null `test_auc` and `test_rmse`.

Leakage checks:

- Confirm item PCA is fit on train items only.
- Confirm held-out item response aggregates are not used as test features.
- Confirm TabPFN train rows come only from train-item observations.
- Inspect the feature matrix construction for any test-mask leakage.

Integration checks:

```bash
conda run -n hal python -m py_compile model/measurement_db_raw.py
bash -n model/reproduce_large_db.sh
```

- Run aggregation and verify `metrics_long.csv` contains `method=tabpfn`.
- Verify config sidecars are written correctly.
- Verify rerun skip/resume behavior works when the same TabPFN config is already complete.

Evaluation:

- Initial run: seeds `0,1,2` against existing kNN/ARAF.
- If promising, run fixed TabPFN config on seeds `3-29`.
- Write paired significance artifacts analogous to `k128_dropout0_tau0_vs_knn_30seed_analysis.md`.
- Compare `tabpfn` vs `knn` and `tabpfn` vs `araf`; do not claim ARAF improvement from TabPFN unless an explicit ensemble plan is later chosen.

## Success Criteria

- TabPFN results integrate cleanly into the existing measurement-db summary pipeline.
- All TabPFN features are split-safe and use train data only.
- Results are reproducible with fixed seeds.
- TabPFN is evaluated as a standalone comparator.
- Empirical outcome is documented whether TabPFN beats kNN/ARAF or not.
