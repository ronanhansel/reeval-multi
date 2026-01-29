# Generalisability Experiments

Reproducible evaluation of Amortized IRT models on HELM and ColBench benchmarks.

## Structure

```
generalisability/
├── embeddings.py     # Embedding generation (Raw/Qwen, PCA, SAE)
├── models.py         # IRT models (Bernoulli and Beta versions)
├── aggregate.py      # N-holdout survey across response matrices
├── plotting.py       # Unified plotting with tueplots icml2024
├── reproduce.sh      # Main reproducibility script
└── result/           # Output directory (CSVs and PDFs)
```

## Quick Start

```bash
# Interactive mode
./reproduce.sh

# Or specify directly
./reproduce.sh helm      # HELM benchmark only
./reproduce.sh colbench  # ColBench only
./reproduce.sh both      # Both benchmarks
```

## Components

### embeddings.py

Handles three embedding types:
- **Raw (Qwen/LLaMA)**: 4096-dimensional language model embeddings
- **PCA**: Dimensionality reduction via Principal Component Analysis
- **SAE**: Sparse Autoencoder features (requires `hypothesaes`)

```bash
python embeddings.py --type pca --dim 48
python embeddings.py --type sae --dim 48 --k 4
```

### models.py

Two IRT model families:
- **Bernoulli IRT**: For binary response data (HELM)
- **Beta IRT**: For continuous [0,1] responses (ColBench aggregated)

Both support:
- Amortized item parameters via embeddings
- Automatic Relevance Determination (ARD) for dimension discovery
- Baseline comparisons (Global Mean, Rasch-IRT)

```bash
python models.py --benchmark helm --model bernoulli
python models.py --benchmark colbench --model beta
```

### aggregate.py

N-holdout survey: evaluates model performance across varying numbers of response matrices.
Runs both PCA and SAE Amortised IRT models for comprehensive comparison.

```bash
python aggregate.py                      # Run all n values (1 to max)
python aggregate.py --n-samples 1,22     # Run only n=1 and n=22
python aggregate.py --n-samples 1-5,22   # Run n=1 through 5, plus 22
python aggregate.py --model beta         # Use Beta IRT (default)
```

### plotting.py

Generates all plots with consistent ICML 2024 styling:
- Uses `tueplots` icml2024 bundle
- Font size 15 throughout
- Consistent color palette

```bash
python plotting.py --plot all      # All available plots
python plotting.py --plot helm     # HELM plots only
python plotting.py --plot colbench # ColBench plots only
```

## Output

Results are saved to `result/`:

| File | Description |
|------|-------------|
| `helm_results.csv` | HELM model comparison (Model, AUC) |
| `colbench_results.csv` | ColBench comparison with PCA and SAE models (Model, RMSE, AUC) |
| `convergence_results.csv` | N-holdout results with PCA/SAE metrics (n_samples, RMSE, AUC per model) |
| `auc_comparison_helm.pdf` | HELM AUC bar chart |
| `rmse_comparison_colbench.pdf` | ColBench RMSE bar chart |
| `auc_comparison_colbench.pdf` | ColBench AUC bar chart |
| `rmse_convergence.pdf` | RMSE vs n_samples line plot (includes PCA and SAE) |
| `auc_convergence.pdf` | AUC vs n_samples line plot (includes PCA and SAE) |
| `rmse_comparison.pdf` | N=1 vs N=max RMSE comparison |
| `auc_comparison.pdf` | N=1 vs N=max AUC comparison |

## Data Dependencies

Data is automatically downloaded from HuggingFace (`ronanhansel/data-reeval-multi`) if not present locally.

All data loaded from `../data-reeval-multi/`:
- `resmat.pkl` - HELM response matrix (183 models x 78,712 items)
- `embed_meta-llama_Llama-3.1-8B-Instruct.pkl` - HELM embeddings
- `colbench/resmat_moon*.csv` - ColBench response matrices
- `hal/all_benchmarks_embeddings_4096_8B.pkl` - ColBench embeddings

## Requirements

```
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
tueplots
hypothesaes  # optional, for SAE embeddings
```
