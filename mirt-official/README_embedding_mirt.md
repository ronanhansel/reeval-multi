# Embedding-based MIRT Implementation

This implementation combines multidimensional Item Response Theory (MIRT) with question embeddings to predict item parameters, inspired by the calibration approach in `calibration.ipynb` and adapted to the MIRT framework from `k-trials.py`.

## Overview

Traditional MIRT models require optimizing item parameters (discrimination `a` and difficulty `b`) for each item individually. This approach uses neural networks to predict these parameters from question embeddings, enabling:

1. **Amortized parameter estimation**: Train once, predict for new items
2. **Better generalization**: Leverages semantic similarity between questions
3. **Scalability**: Can handle new items without retraining the entire MIRT model
4. **Hybrid approach**: Combines traditional optimization with embedding-based prediction

## Files

### Core Implementation

- `embedding_mirt.py`: Main implementation with neural network architecture and training functions
- `compare_mirt_methods.py`: Comprehensive comparison between traditional and embedding-based MIRT
- `demo_embedding_mirt.py`: Demo script showing basic usage and examples

### Key Components

#### Neural Network Architecture (`MIRTParamPredictor`)

```python
class MIRTParamPredictor(nn.Module):
    def __init__(self, embedding_dim, k_factors):
        # Shared layers for feature extraction
        self.shared = nn.Sequential(...)

        # Separate heads for different parameter types
        self.a_head = nn.Sequential(...)  # Discrimination parameters (k-dimensional)
        self.b_head = nn.Sequential(...)  # Difficulty parameters (scalar)
```

**Design choices:**

- **Shared backbone**: Common feature extraction for both parameter types
- **Separate heads**: Different output layers for `a` and `b` parameters
- **Softplus activation**: Ensures positive discrimination parameters
- **Dropout**: Prevents overfitting in high-dimensional embedding space

#### Training Strategy

1. **Target Generation**: Train traditional MIRT to get target parameters
2. **Embedding Preparation**: Match questions with embeddings, handle missing cases
3. **Neural Network Training**: Predict parameters from embeddings using MSE loss
4. **Hybrid Integration**: Use NN predictions where embeddings exist, traditional parameters otherwise
5. **Person Parameter Refinement**: Fine-tune person abilities with predicted item parameters

## Usage Examples

### Basic Usage

```python
from embedding_mirt import load_embeddings_and_questions, train_embedding_predictor

# Load data
resmat, embeds, question_to_emb = load_embeddings_and_questions()

# Train model
k = 4  # number of factors
model = train_embedding_predictor(X_embeddings, a_targets, b_targets, k)

# Predict for new items
a_pred, b_pred = model(new_embeddings)
```

### Complete Comparison

```python
# Compare traditional vs embedding-based MIRT
python compare_mirt_methods.py
```

### Demo and Exploration

```python
# Run interactive demo
python demo_embedding_mirt.py
```

## Key Advantages

### 1. **Scalability**

- Once trained, can instantly predict parameters for new items with embeddings
- No need to retrain entire MIRT model for new items
- Particularly valuable for adaptive testing scenarios

### 2. **Generalization**

- Leverages semantic similarity between questions
- Can potentially predict parameters for items similar to training items
- Reduces noise in parameter estimation through regularization

### 3. **Flexibility**

- Hybrid approach uses best of both methods
- Falls back to traditional optimization for items without embeddings
- Can be easily extended to different embedding types

### 4. **Interpretability**

- Neural network learns meaningful representations of item characteristics
- Can analyze which embedding features predict difficulty vs discrimination
- Enables understanding of what makes items difficult or discriminating

## Performance Considerations

### Expected Performance Patterns

1. **High Embedding Coverage**: When most items have embeddings, embedding MIRT should perform similarly or better than traditional MIRT
2. **Low Embedding Coverage**: Performance depends on quality of traditional fallback parameters
3. **Factor Complexity**: Higher k values may benefit more from embedding regularization
4. **Data Size**: Larger datasets provide better NN training, improving embedding predictions

### Evaluation Metrics

The implementation tracks:

- **AUC-ROC**: Primary performance metric for binary predictions
- **Parameter Correlation**: How well predicted parameters match traditional targets
- **MSE**: Mean squared error for parameter predictions
- **Coverage**: Percentage of items with embeddings

## Relationship to Original Implementations

### From `calibration.ipynb`

- **Embedding loading and preprocessing pattern**
- **Neural network architecture inspiration (AbilityPredictor)**
- **Train/test split methodology for embeddings**
- **Hybrid approach (NN + traditional optimization)**

### From `k-trials.py`

- **MIRT parameter structure (theta, a, b)**
- **Training loop with early stopping**
- **Weighted loss for item imbalance**
- **Evaluation metrics and model comparison framework**

### Key Adaptations

- **Multi-output NN**: Predicts both `a` (k-dimensional) and `b` (scalar) simultaneously
- **Separate parameter heads**: Different network branches for different parameter types
- **Hybrid parameter integration**: Seamless combination of NN and traditional parameters
- **Person parameter refinement**: Fine-tuning after item parameter prediction

## Future Extensions

### Possible Improvements

1. **Advanced architectures**: Transformer-based embedding models
2. **Multi-task learning**: Jointly optimize response prediction and parameter estimation
3. **Bayesian approaches**: Uncertainty quantification for parameter predictions
4. **Cross-validation**: More robust evaluation of embedding effectiveness
5. **Domain adaptation**: Transfer learning across different test domains

### Research Directions

1. **Embedding quality analysis**: Which embedding types work best for IRT parameters?
2. **Parameter interpretation**: What semantic features predict difficulty vs discrimination?
3. **Active learning**: How to select which items need traditional vs embedding-based calibration?
4. **Robustness**: How sensitive are predictions to embedding quality and coverage?

## Requirements

- PyTorch >= 1.9
- pandas, numpy, scikit-learn
- tqdm for progress bars
- matplotlib, seaborn for visualization
- Pre-computed question embeddings (`embed_meta-llama_Llama-3.1-8B-Instruct.pkl`)
- Response matrix data (`resmat.pkl`)

## Citation and Attribution

This implementation builds on:

- Multidimensional IRT framework from Reckase (2009)
- Neural parameter prediction inspired by variational autoencoders for IRT
- Embedding approaches from transformer-based language models
- Implementation patterns from the reeval-multi calibration notebooks
