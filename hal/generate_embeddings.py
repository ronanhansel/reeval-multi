#!/usr/bin/env python3
"""
Generate Embeddings for Amortized IRT Experiments

Produces three types of embeddings from raw Qwen3-Embedding-8B vectors:
  1. Raw embeddings (4096-dim)
  2. PCA embeddings (reduced dimensionality)
  3. SAE embeddings (sparse autoencoder features)

All embeddings are saved locally and can be pushed to HuggingFace.

Usage:
    python generate_embeddings.py                      # Generate all embedding types
    python generate_embeddings.py --push-to-hf         # Generate and push to HuggingFace
    python generate_embeddings.py --pca-dim 48         # Custom PCA dimensions
    python generate_embeddings.py --sae-features 48    # Custom SAE features
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download, HfApi
from sklearn.decomposition import PCA

try:
    from hypothesaes.quickstart import train_sae
    HAS_SAE_LIB = True
except ImportError:
    HAS_SAE_LIB = False
    print("WARNING: 'hypothesaes' library not found. SAE embeddings will not be generated.")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

HF_REPO_ID = "ronanhansel/data-reeval-multi"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data-reeval-multi')

# Default hyperparameters
DEFAULT_PCA_DIM = 48
DEFAULT_SAE_FEATURES = 48
DEFAULT_SAE_K_SPARSITY = 4
DEFAULT_SAE_EPOCHS = 100
DEFAULT_SAE_LR = 5e-4
DEFAULT_SAE_BATCH_SIZE = 512

RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def ensure_data_downloaded():
    """Download raw data from HuggingFace if not present."""
    emb_file = os.path.join(DATA_DIR, 'hal', 'all_benchmarks_embeddings_4096_8B.pkl')

    if not os.path.exists(emb_file):
        print(f"Raw embeddings not found. Downloading from HuggingFace ({HF_REPO_ID})...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print("Download complete.")

    return emb_file


def load_raw_embeddings(emb_file):
    """Load raw embeddings from pickle file."""
    print(f"Loading raw embeddings from {emb_file}...")
    emb_df = pd.read_pickle(emb_file)

    # Extract task IDs and embeddings
    task_ids = []
    embeddings = []

    for _, row in emb_df.iterrows():
        task_id = str(row['benchmark.task_id'])
        emb = row['embedding']

        # Handle string-encoded embeddings
        if isinstance(emb, str):
            import ast
            emb = ast.literal_eval(emb)

        task_ids.append(task_id)
        embeddings.append(np.array(emb, dtype=np.float32))

    embeddings = np.stack(embeddings)
    print(f"Loaded {len(task_ids)} embeddings with shape {embeddings.shape}")

    return task_ids, embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Generation
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
    parser = argparse.ArgumentParser(description='Generate embeddings for Amortized IRT')
    parser.add_argument('--pca-dim', type=int, default=DEFAULT_PCA_DIM,
                        help=f'PCA output dimensions (default: {DEFAULT_PCA_DIM})')
    parser.add_argument('--sae-features', type=int, default=DEFAULT_SAE_FEATURES,
                        help=f'SAE feature dimensions (default: {DEFAULT_SAE_FEATURES})')
    parser.add_argument('--sae-k', type=int, default=DEFAULT_SAE_K_SPARSITY,
                        help=f'SAE sparsity K (default: {DEFAULT_SAE_K_SPARSITY})')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(DATA_DIR, 'hal', 'processed_embeddings'),
                        help='Output directory for embeddings')
    parser.add_argument('--push-to-hf', action='store_true',
                        help='Push generated embeddings to HuggingFace')
    args = parser.parse_args()

    print("=" * 60)
    print("EMBEDDING GENERATION")
    print("=" * 60)
    print(f"PCA dimensions: {args.pca_dim}")
    print(f"SAE features: {args.sae_features} (K={args.sae_k})")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load raw embeddings
    emb_file = ensure_data_downloaded()
    task_ids, raw_embeddings = load_raw_embeddings(emb_file)

    # Generate PCA embeddings
    pca_embeddings, pca_model = generate_pca_embeddings(
        raw_embeddings, n_components=args.pca_dim
    )

    # Generate SAE embeddings
    sae_embeddings, sae_model = generate_sae_embeddings(
        raw_embeddings, n_features=args.sae_features, k_sparsity=args.sae_k
    )

    # Save all embeddings
    print("\n" + "=" * 60)
    print("SAVING EMBEDDINGS")
    print("=" * 60)
    save_embeddings(
        task_ids, raw_embeddings, pca_embeddings, sae_embeddings,
        args.output_dir, args.pca_dim, args.sae_features
    )

    # Push to HuggingFace if requested
    if args.push_to_hf:
        push_to_huggingface(args.output_dir)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
