#!/usr/bin/env python3
"""
Generate Embeddings for Amortized IRT Experiments

Produces three types of embeddings:
  1. Raw LLM embeddings (4096-dim from Qwen3-Embedding-8B)
  2. PCA embeddings (reduced dimensionality)
  3. SAE embeddings (sparse autoencoder features)

Optionally interprets SAE features using an LLM (GPT-4o).

All embeddings are saved locally and can be pushed to HuggingFace.

Usage:
    # Generate PCA/SAE from existing raw embeddings
    python generate_embeddings.py

    # Full pipeline: generate raw embeddings from text, then PCA/SAE
    python generate_embeddings.py --from-text

    # Custom dimensions
    python generate_embeddings.py --from-text --pca-dim 48 --sae-features 48

    # Interpret SAE features with GPT-4o (requires OPENAI_API_KEY)
    python generate_embeddings.py --interpret

    # Push all embeddings to HuggingFace
    python generate_embeddings.py --from-text --push-to-hf
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download, HfApi
from sklearn.decomposition import PCA

try:
    from hypothesaes.quickstart import train_sae, interpret_sae
    HAS_SAE_LIB = True
except ImportError:
    HAS_SAE_LIB = False
    print("WARNING: 'hypothesaes' library not found. SAE embeddings will not be generated.")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

HF_REPO_ID = "ronanhansel/data-reeval-multi"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'processed_embeddings')

# Benchmarks for raw embedding generation
BENCHMARKS = [
    'scicode', 'gaia', 'taubench_airline', 'scienceagentbench', 'corebench_hard',
    'assistantbench', 'usaco', 'online_mind2web', 'swebench_verified_mini',
    'colbench_backend_programming'
]

# Raw embedding settings
DEFAULT_LLM_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BATCH_SIZE = 8
MAX_CHARS = 20000  # Truncate to prevent OOM

# Default hyperparameters for dimensionality reduction
DEFAULT_PCA_DIM = 48
DEFAULT_SAE_FEATURES = 48
DEFAULT_SAE_K_SPARSITY = 4
DEFAULT_SAE_EPOCHS = 100
DEFAULT_SAE_LR = 5e-4
DEFAULT_SAE_BATCH_SIZE = 512

RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Raw Embedding Generation (from text)
# ══════════════════════════════════════════════════════════════════════════════

def generate_raw_embeddings_from_text(data_dir, output_file, model_name=DEFAULT_LLM_MODEL,
                                       batch_size=DEFAULT_BATCH_SIZE):
    """
    Generate raw LLM embeddings from benchmark text inputs.

    Args:
        data_dir: Directory containing {benchmark}_inputs.csv files
        output_file: Path to save the output pickle file
        model_name: HuggingFace model ID for embeddings
        batch_size: Batch size for embedding generation

    Returns:
        Path to the saved embeddings file
    """
    import torch
    from sentence_transformers import SentenceTransformer

    # Setup cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
    os.environ['HF_HOME'] = CACHE_DIR

    # Load input data
    print("Loading input data from CSV files...")
    dfs = []

    for benchmark in BENCHMARKS:
        csv_file = os.path.join(data_dir, f"{benchmark}_inputs.csv")
        if os.path.exists(csv_file):
            temp_df = pd.read_csv(csv_file)
            temp_df['benchmark'] = benchmark
            dfs.append(temp_df)
            print(f"  Loaded {len(temp_df)} items from {benchmark}")
        else:
            print(f"  WARNING: {csv_file} not found")

    if not dfs:
        raise FileNotFoundError("No input CSV files found!")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df)} task inputs")

    if 'task_id' not in df.columns or 'text_input' not in df.columns:
        raise ValueError("CSV files must have 'task_id' and 'text_input' columns")

    # Pre-process text
    print(f"Pre-processing texts (truncating to {MAX_CHARS} chars)...")
    texts = df['text_input'].astype(str).tolist()
    truncated_texts = [t[:MAX_CHARS] for t in texts]

    # Load model and generate embeddings
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=CACHE_DIR, trust_remote_code=True)

    print(f"Generating embeddings (batch_size={batch_size})...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    embeddings = model.encode(
        truncated_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=False,
    )

    # Format and save
    df['embedding'] = list(embeddings)
    df['benchmark.task_id'] = df['benchmark'] + '.' + df['task_id'].astype(str)
    final_df = df[['benchmark.task_id', 'text_input', 'embedding']]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_pickle(output_file)

    print(f"Saved raw embeddings to: {output_file}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Embedding dim: {len(final_df.iloc[0]['embedding'])}")

    return output_file


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def ensure_data_downloaded():
    """Returns local paths for the main data directory and raw embeddings."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, 'item-editor', 'eval_response_matrix')
    emb_file = os.path.join(data_dir, 'all_benchmarks_embeddings_4096_8B.pkl')
    return data_dir, emb_file


def load_raw_embeddings(emb_file):
    """Load raw embeddings and texts from pickle file."""
    print(f"Loading raw embeddings from {emb_file}...")
    emb_df = pd.read_pickle(emb_file)

    # Extract task IDs, embeddings, and texts
    task_ids = []
    embeddings = []
    texts = []

    has_text = 'text_input' in emb_df.columns

    for _, row in emb_df.iterrows():
        task_id = str(row['benchmark.task_id'])
        emb = row['embedding']

        # Handle string-encoded embeddings
        if isinstance(emb, str):
            import ast
            emb = ast.literal_eval(emb)

        task_ids.append(task_id)
        embeddings.append(np.array(emb, dtype=np.float32))

        if has_text:
            texts.append(str(row['text_input']) if pd.notna(row['text_input']) else "")
        else:
            texts.append("")

    embeddings = np.stack(embeddings)
    print(f"Loaded {len(task_ids)} embeddings with shape {embeddings.shape}")
    if has_text:
        print(f"Loaded {sum(1 for t in texts if t)} texts for interpretation")

    return task_ids, embeddings, texts


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Generation (PCA / SAE)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_embeddings(embeddings):
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return embeddings / norms


def generate_pca_embeddings(embeddings, n_components=DEFAULT_PCA_DIM):
    """Generate PCA-reduced embeddings."""
    print(f"Generating PCA embeddings (n_components={n_components})...")

    # Normalize before PCA
    embeddings_norm = normalize_embeddings(embeddings)

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_embeddings = pca.fit_transform(embeddings_norm)

    # Normalize output
    pca_embeddings = normalize_embeddings(pca_embeddings)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.2%}")
    print(f"PCA embeddings shape: {pca_embeddings.shape}")

    return pca_embeddings, pca


def generate_sae_embeddings(embeddings, n_features=DEFAULT_SAE_FEATURES,
                            k_sparsity=DEFAULT_SAE_K_SPARSITY):
    """Generate SAE sparse feature embeddings."""
    if not HAS_SAE_LIB:
        print("Skipping SAE embeddings (hypothesaes library not available)")
        return None, None

    print(f"Generating SAE embeddings (M={n_features}, K={k_sparsity})...")

    # Normalize before SAE
    embeddings_norm = normalize_embeddings(embeddings)

    # Train SAE
    sae = train_sae(
        embeddings=embeddings_norm,
        M=n_features,
        K=k_sparsity,
        batch_size=DEFAULT_SAE_BATCH_SIZE,
        n_epochs=DEFAULT_SAE_EPOCHS,
        learning_rate=DEFAULT_SAE_LR,
        checkpoint_dir='/tmp/_generate_embeddings_sae_ckpt'
    )

    # Get activations
    sae_embeddings = sae.get_activations(embeddings_norm)

    # Normalize output
    sae_embeddings = normalize_embeddings(sae_embeddings)

    # Report sparsity
    avg_active = (np.abs(sae_embeddings) > 1e-6).sum(axis=1).mean()
    print(f"SAE avg active features per item: {avg_active:.2f} (target K={k_sparsity})")
    print(f"SAE embeddings shape: {sae_embeddings.shape}")

    return sae_embeddings, sae


def interpret_sae_features(sae, texts, embeddings, n_features, output_path,
                           interpreter_model="gpt-4o"):
    """
    Interpret SAE features using an LLM.

    Args:
        sae: Trained SAE model
        texts: List of text inputs corresponding to embeddings
        embeddings: Raw embeddings (will be normalized)
        n_features: Number of SAE features to interpret
        output_path: Path to save feature descriptions pickle
        interpreter_model: LLM model for interpretation (default: gpt-4o)

    Returns:
        DataFrame with feature interpretations
    """
    if not HAS_SAE_LIB:
        print("Skipping interpretation (hypothesaes library not available)")
        return None

    # Check for existing interpretations
    if os.path.exists(output_path):
        print(f"Loading existing feature descriptions from {output_path}")
        feature_df = pd.read_pickle(output_path)
        print(f"Loaded {len(feature_df)} feature interpretations")
        return feature_df

    # Filter out empty texts
    valid_texts = [t for t in texts if t and str(t).strip()]
    if not valid_texts:
        print("No valid texts found for interpretation")
        return None

    print(f"Interpreting {n_features} SAE features with {interpreter_model}...")
    print(f"Using {len(valid_texts)} texts for interpretation")

    try:
        embeddings_norm = normalize_embeddings(embeddings)

        feature_df = interpret_sae(
            texts=texts,
            embeddings=embeddings_norm,
            sae=sae,
            n_top_neurons=n_features,
            interpreter_model=interpreter_model
        )

        if len(feature_df) > 0:
            feature_df.to_pickle(output_path)
            print(f"Saved {len(feature_df)} feature interpretations to {output_path}")

            print("\nSample interpretations:")
            print(feature_df[['neuron_idx', 'interpretation']].head(10))
        else:
            print("No feature descriptions generated")

        return feature_df

    except Exception as e:
        print(f"Interpretation failed: {e}")
        print("Make sure OPENAI_API_KEY or OPENAI_KEY_SAE environment variable is set")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Save & Upload
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings(task_ids, raw_emb, pca_emb, sae_emb, output_dir, pca_dim, sae_features):
    """Save all embeddings to pickle files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save raw embeddings (reference copy with task_id alignment)
    raw_df = pd.DataFrame({
        'task_id': task_ids,
        'embedding': list(raw_emb)
    })
    raw_path = os.path.join(output_dir, 'embeddings_raw.pkl')
    raw_df.to_pickle(raw_path)
    print(f"Saved raw embeddings to {raw_path}")

    # Save PCA embeddings
    pca_df = pd.DataFrame({
        'task_id': task_ids,
        'embedding': list(pca_emb)
    })
    pca_path = os.path.join(output_dir, f'embeddings_pca_{pca_dim}.pkl')
    pca_df.to_pickle(pca_path)
    print(f"Saved PCA embeddings to {pca_path}")

    # Save SAE embeddings
    if sae_emb is not None:
        sae_df = pd.DataFrame({
            'task_id': task_ids,
            'embedding': list(sae_emb)
        })
        sae_path = os.path.join(output_dir, f'embeddings_sae_{sae_features}.pkl')
        sae_df.to_pickle(sae_path)
        print(f"Saved SAE embeddings to {sae_path}")

    return raw_path, pca_path


def push_to_huggingface(output_dir, repo_id=HF_REPO_ID):
    """Push generated embeddings to HuggingFace."""
    print(f"\nPushing embeddings to HuggingFace ({repo_id})...")

    api = HfApi()

    # Upload all embedding files
    for filename in os.listdir(output_dir):
        if filename.startswith('embeddings_') and filename.endswith('.pkl'):
            local_path = os.path.join(output_dir, filename)
            remote_path = f"hal/processed_embeddings/{filename}"

            print(f"  Uploading {filename} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
            )

    print("Upload complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for Amortized IRT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PCA/SAE from existing raw embeddings
  python generate_embeddings.py

  # Full pipeline: raw embeddings from text -> PCA -> SAE
  python generate_embeddings.py --from-text

  # Custom dimensions and push to HuggingFace
  python generate_embeddings.py --from-text --pca-dim 64 --sae-features 64 --push-to-hf
        """
    )

    # Raw embedding options
    parser.add_argument('--from-text', action='store_true',
                        help='Generate raw embeddings from text inputs (requires GPU)')
    parser.add_argument('--llm-model', type=str, default=DEFAULT_LLM_MODEL,
                        help=f'LLM model for raw embeddings (default: {DEFAULT_LLM_MODEL})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for LLM encoding (default: {DEFAULT_BATCH_SIZE})')

    # Dimensionality reduction options
    parser.add_argument('--pca-dim', type=int, default=DEFAULT_PCA_DIM,
                        help=f'PCA output dimensions (default: {DEFAULT_PCA_DIM})')
    parser.add_argument('--sae-features', type=int, default=DEFAULT_SAE_FEATURES,
                        help=f'SAE feature dimensions (default: {DEFAULT_SAE_FEATURES})')
    parser.add_argument('--sae-k', type=int, default=DEFAULT_SAE_K_SPARSITY,
                        help=f'SAE sparsity K (default: {DEFAULT_SAE_K_SPARSITY})')

    # Output options
    parser.add_argument('--output-dir', type=str,
                        default=OUTPUT_DIR,
                        help='Output directory for embeddings')
    parser.add_argument('--push-to-hf', action='store_true',
                        help='Push generated embeddings to HuggingFace')

    # Interpretation options
    parser.add_argument('--interpret', action='store_true',
                        help='Interpret SAE features using LLM (requires OPENAI_API_KEY)')
    parser.add_argument('--interpreter-model', type=str, default='gpt-4o',
                        help='LLM model for SAE interpretation (default: gpt-4o)')

    args = parser.parse_args()

    print("=" * 60)
    print("EMBEDDING GENERATION")
    print("=" * 60)
    print(f"From text: {args.from_text}")
    if args.from_text:
        print(f"LLM model: {args.llm_model}")
        print(f"Batch size: {args.batch_size}")
    print(f"PCA dimensions: {args.pca_dim}")
    print(f"SAE features: {args.sae_features} (K={args.sae_k})")
    print(f"Interpret SAE: {args.interpret}")
    if args.interpret:
        print(f"Interpreter model: {args.interpreter_model}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Step 1: Get raw embeddings
    data_dir, raw_emb_file = ensure_data_downloaded()

    if args.from_text:
        print("=" * 60)
        print("STEP 1: GENERATING RAW EMBEDDINGS FROM TEXT")
        print("=" * 60)
        input_dir = os.path.join(data_dir, 'hal')
        raw_emb_file = generate_raw_embeddings_from_text(
            data_dir=input_dir,
            output_file=raw_emb_file,
            model_name=args.llm_model,
            batch_size=args.batch_size
        )
        print()

    # Step 2: Load raw embeddings
    task_ids, raw_embeddings, texts = load_raw_embeddings(raw_emb_file)

    # Step 3: Generate PCA embeddings
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING PCA EMBEDDINGS")
    print("=" * 60)
    pca_embeddings, pca_model = generate_pca_embeddings(
        raw_embeddings, n_components=args.pca_dim
    )

    # Step 4: Generate SAE embeddings
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING SAE EMBEDDINGS")
    print("=" * 60)
    sae_embeddings, sae_model = generate_sae_embeddings(
        raw_embeddings, n_features=args.sae_features, k_sparsity=args.sae_k
    )

    # Step 4b: Interpret SAE features (optional)
    if args.interpret and sae_model is not None:
        print("\n" + "=" * 60)
        print("STEP 3b: INTERPRETING SAE FEATURES")
        print("=" * 60)
        interpret_output = os.path.join(args.output_dir, 'feature_descriptions_sae.pkl')
        interpret_sae_features(
            sae=sae_model,
            texts=texts,
            embeddings=raw_embeddings,
            n_features=args.sae_features,
            output_path=interpret_output,
            interpreter_model=args.interpreter_model
        )

    # Step 5: Save all embeddings
    print("\n" + "=" * 60)
    print("STEP 4: SAVING EMBEDDINGS")
    print("=" * 60)
    save_embeddings(
        task_ids, raw_embeddings, pca_embeddings, sae_embeddings,
        args.output_dir, args.pca_dim, args.sae_features
    )

    # Step 6: Push to HuggingFace if requested
    if args.push_to_hf:
        print("\n" + "=" * 60)
        print("STEP 5: PUSHING TO HUGGINGFACE")
        print("=" * 60)
        push_to_huggingface(args.output_dir)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
