# HAL: Hierarchical ARD for LLM Evaluation

## Overview

This project trains a Hierarchical Automatic Relevance Determination (ARD) model to analyze LLM performance across multiple benchmarks and behavioral dimensions.

### Key Data Structure Issue

**⚠️ IMPORTANT**: Task IDs are **NOT globally unique**. Different benchmarks use the same sequential IDs (1-100, etc.), causing:
- **156 unique task_ids** across all benchmarks
- **191 total items** when accounting for (task_id, benchmark) pairs
- **35 task_ids** appear in multiple benchmarks (e.g., task_id='11' in both `taubench` and `scicode`)

**Solution**: Always use `(task_id, benchmark)` as the key for embeddings, not just `task_id`.

---

## Data Files

### Input Data
```
data/
├── resmat_binary_success_rate.pkl      # 46×191 binary success matrix (y_data)
├── resmat_environmentalbarrier.label.pkl
├── resmat_instructionfollowing.label.pkl
├── resmat_selfcorrection.label.pkl
├── resmat_tooluse.label.pkl
├── resmat_verification.label.pkl       # 5 behavior matrices (z_data)
├── all_benchmarks_inputs.pkl           # 1479 rows with task_id, benchmark_id, task_input
└── task_bench_to_embedding.pkl         # Generated: (task_id, benchmark) → embedding mapping

result/
└── all_benchmarks_with_embeddings.pkl  # 1479 rows with embeddings column (2560-dim)
```

### Data Dimensions
- **N = 46**: Number of models
- **J = 191**: Number of items (unique task_id + benchmark combinations)
- **M = 5**: Number of behavioral dimensions
- **d_features = 2560**: Embedding dimension (Qwen3-Embedding-4B)
- **K = 25**: Number of latent factors (configurable)

---

## Quick Start

### 1. Generate Embeddings (if needed)

**Check if embeddings exist and are complete:**
```bash
conda run -n reeval python embed.py --check data/resmat_binary_success_rate.pkl
```

**Generate embeddings for all inputs:**
```bash
# Full embedding (takes ~hours on GPU)
conda run -n reeval python embed.py

# Or with resmat checking (only embeds what's needed)
conda run -n reeval python embed.py --check data/resmat_binary_success_rate.pkl
```

**Features:**
- ✅ Automatic resume if interrupted (saves batches to `temp/`)
- ✅ Batch size validation on resume
- ✅ OOM protection (truncates long inputs to 20k chars)
- ✅ GPU memory management with `torch.cuda.empty_cache()`

### 2. Create Embedding Mapping

```bash
conda run -n reeval python create_embeddings_mapping.py
```

**What it does:**
- Loads embeddings from `result/all_benchmarks_with_embeddings.pkl`
- Creates `(task_id, benchmark)` keyed dictionary
- Handles missing item `('73', 'scicode')` with zero vector
- Saves to `data/task_bench_to_embedding.pkl`
- **Verifies 191/191 items matched**

### 3. Train the Model

```bash
conda run -n reeval python train.py
```

**Expected output:**
```
Loading actual data from pickles...
Loaded embeddings for 191 (task_id, benchmark) pairs
Data loaded: N=46 (models), J=191 (items), M=5 (behaviors), d_features=2560
Matched 191/191 embeddings
...
Ep 1000 | Loss 3.36e+04 | Tau: [4.269 3.9 3.729 ...]
Effective Dimension: 25
```

---

## Key Files

### Core Scripts

| File | Purpose |
|------|---------|
| `embed.py` | Generate embeddings using Qwen3-Embedding-4B |
| `create_embeddings_mapping.py` | Map (task_id, benchmark) → embeddings |
| `train.py` | Train Hierarchical ARD model |

### Diagnostic Scripts

| File | Purpose |
|------|---------|
| `verify_embeddings_coverage.py` | Check if all resmat items have embeddings |
| `find_repeated_task_ids.py` | Identify task_id collisions across benchmarks |
| `find_missing_task.py` | Find details about missing embeddings |

### Notebooks

| File | Purpose |
|------|---------|
| `inspect.ipynb` | Data exploration and visualization |
| `train.ipynb` | Interactive training (legacy, use `train.py`) |

---

## Model Architecture

### RobustARDModel

```python
# Amortized item loadings via embeddings
a_j = x_j @ W.T * tau

# Overall prediction
logits_y = θ @ a_j.T + δ_j

# Subskill prediction with gates
logits_z[m] = θ @ (a_j * g_m).T + δ_zm
```

**Key components:**
- `x_j`: Item embeddings (2560-dim, normalized)
- `W`: Weight matrix (K × 2560)
- `tau`: ARD scales (K-dim, ReLU ensures sparsity)
- `g_m`: Behavioral gates (M × K, sigmoid)
- `θ`: Model abilities (N × K)

**Hyperparameters:**
- `K_MODEL = 25`: Number of latent factors
- `lambda_tau = 5.0`: L1 penalty on tau (controls sparsity)
- Learning rates: 0.005 (tau), 0.01 (others)

---

## Troubleshooting

### Issue: "Matched 0/191 embeddings"

**Cause**: Using `task_id` only instead of `(task_id, benchmark)`.

**Fix**: Regenerate mapping with `create_embeddings_mapping.py`.

### Issue: "Missing embeddings for item X"

**Cause**: Embedding file doesn't have that (task_id, benchmark) pair.

**Fix**: 
```bash
# Check what's missing
python verify_embeddings_coverage.py

# Re-embed with checking
python embed.py --check data/resmat_binary_success_rate.pkl
```

### Issue: OOM during embedding

**Cause**: Extremely long task_input texts.

**Fix**: `embed.py` automatically truncates to 20k chars. Reduce `batch_size` if needed (line 30).

### Issue: All tau → 0 during training

**Cause**: `lambda_tau` too high or learning rate imbalance.

**Fix**: Adjust hyperparameters in `train.py` (lines 165-170):
- Decrease `lambda_tau` (try 2.0, 1.0)
- Increase tau learning rate (try 0.01)

---

## File Structure Explained

### MultiIndex Columns in Resmat

```python
resmat.columns = MultiIndex([
    ('task_id', 'text_input', 'benchmark'),
    ...
])

# Example columns:
('11', 'You are a user...', 'taubench')
('11', 'You are a world expert...', 'scicode')  # Same task_id, different benchmark!
```

### Embeddings DataFrame

```python
all_benchmarks_with_embeddings.pkl:
    task_id | benchmark_id | task_input | embeddings
    --------|-------------|------------|------------
    hash1   | taubench    | "text..."  | array(2560,)
    11      | scicode     | "text..."  | array(2560,)
    11      | taubench    | "text..."  | array(2560,)
```

**Note**: Same (task_id, benchmark) can appear multiple times if different text_inputs are used.

---

## Common Commands

```bash
# Check embeddings coverage
conda run -n reeval python verify_embeddings_coverage.py

# Find which task_ids appear multiple times
conda run -n reeval python find_repeated_task_ids.py

# Run full pipeline
conda run -n reeval python embed.py --check data/resmat_binary_success_rate.pkl
conda run -n reeval python create_embeddings_mapping.py
conda run -n reeval python train.py

# Run training for longer
# Edit train.py line 173: for e in range(2001)
conda run -n reeval python train.py > train_output.log 2>&1
```

---

## Expected Results

**Good training run:**
- ✅ Loss decreases from ~37k to ~34k
- ✅ 20-25 effective dimensions (tau > 0)
- ✅ Top tau values > 3.0
- ✅ All 191/191 embeddings matched

**Bad training run:**
- ❌ 0 matched embeddings
- ❌ All tau → 0 (no effective dimensions)
- ❌ Loss stays high or increases

---

## Environment

```bash
conda activate reeval
# PyTorch 2.7.1
# sentence-transformers (for Qwen3-Embedding-4B)
# pandas, numpy, tqdm
```

**GPU**: Requires CUDA-capable GPU for embedding generation.

---

## Notes

- **Missing item**: `('73', 'scicode')` has empty text, uses zero vector
- **Embedding model**: Qwen3-Embedding-4B (2560-dim output)
- **Cache**: Embeddings saved to `result/`, temporary batches in `temp/`
- **Resume**: `embed.py` automatically resumes from interrupted runs

---

## Next Steps

1. **Tune hyperparameters**: Adjust `lambda_tau`, `K_MODEL`, learning rates
2. **Extend training**: Increase epochs for better convergence
3. **Analyze gates**: Examine which dimensions activate for each behavior
4. **Save model**: Add checkpointing to `train.py`
5. **Evaluation**: Add held-out test set evaluation

---

## Authors & License

Part of the `reeval-multi` project for multi-dimensional LLM evaluation.
