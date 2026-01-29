"""
Embedding Generation and Processing Module

Supports three embedding types:
1. Raw (Qwen/LLaMA) - 4096-dimensional embeddings from language models
2. PCA - Dimensionality reduction via Principal Component Analysis
3. SAE - Sparse Autoencoder features

Usage:
    python embeddings.py --type pca --dim 48
    python embeddings.py --type sae --dim 48 --k 4
    python embeddings.py --type raw
"""

import argparse
import ast
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from huggingface_hub import snapshot_download

warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data-reeval-multi')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'embeddings_cache')

# HuggingFace dataset
HF_REPO_ID = "ronanhansel/data-reeval-multi"


def ensure_data_downloaded():
    """Download data from HuggingFace if not present locally."""
    if not os.path.exists(DATA_DIR):
        print(f"Data not found locally. Downloading from HuggingFace ({HF_REPO_ID})...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print("Download complete.")

# Default SAE settings
SAE_EPOCHS = 100
SAE_LR = 5e-4
SAE_BATCH_SIZE = 512


def load_raw_embeddings(benchmark='helm'):
    """
    Load raw embeddings from pre-computed files.

    Args:
        benchmark: 'helm' for HELM benchmark, 'colbench' for ColBench

    Returns:
        embeddings: numpy array of shape (n_items, 4096)
        task_ids: list of task identifiers
    """
    # Ensure data is downloaded
    ensure_data_downloaded()

    if benchmark == 'helm':
        emb_file = os.path.join(DATA_DIR, 'embed_meta-llama_Llama-3.1-8B-Instruct.pkl')
        id_col = 'question'
    else:
        emb_file = os.path.join(DATA_DIR, 'hal', 'all_benchmarks_embeddings_4096_8B.pkl')
        id_col = 'benchmark.task_id'

    print(f"Loading raw embeddings from {emb_file}...")
    emb_df = pd.read_pickle(emb_file)

    embeddings = []
    task_ids = []

    for _, row in emb_df.iterrows():
        task_id = str(row[id_col])
        emb = row['embedding']
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)
        embeddings.append(np.array(emb, dtype=np.float32))
        task_ids.append(task_id)

    embeddings = np.stack(embeddings)

    # L2 normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    print(f"Loaded {len(task_ids)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings, task_ids


def compute_pca_embeddings(raw_embeddings, n_components=48):
    """
    Apply PCA dimensionality reduction to embeddings.

    Args:
        raw_embeddings: numpy array of shape (n_items, raw_dim)
        n_components: target dimensionality

    Returns:
        pca_embeddings: numpy array of shape (n_items, n_components)
        pca_model: fitted PCA model
    """
    print(f"Computing PCA embeddings (dim={n_components})...")

    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(raw_embeddings)

    # L2 normalize
    pca_embeddings = pca_embeddings / (np.linalg.norm(pca_embeddings, axis=1, keepdims=True) + 1e-8)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.4f}")
    print(f"Output shape: {pca_embeddings.shape}")

    return pca_embeddings, pca


def compute_sae_embeddings(raw_embeddings, m_features=48, k_sparsity=4,
                           epochs=SAE_EPOCHS, checkpoint_dir=None):
    """
    Train Sparse Autoencoder and compute sparse features.

    Args:
        raw_embeddings: numpy array of shape (n_items, raw_dim)
        m_features: number of SAE features (expansion dimension)
        k_sparsity: top-k sparsity constraint
        epochs: training epochs
        checkpoint_dir: directory to save/load SAE checkpoints

    Returns:
        sae_embeddings: numpy array of shape (n_items, m_features)
        sae_model: trained SAE model
    """
    try:
        from hypothesaes.quickstart import train_sae
    except ImportError:
        raise ImportError("hypothesaes library required for SAE. Install with: pip install hypothesaes")

    if checkpoint_dir is None:
        checkpoint_dir = '/tmp/_sae_embeddings_ckpt'

    print(f"Computing SAE embeddings (M={m_features}, K={k_sparsity})...")

    sae = train_sae(
        embeddings=raw_embeddings,
        M=m_features,
        K=k_sparsity,
        batch_size=SAE_BATCH_SIZE,
        n_epochs=epochs,
        learning_rate=SAE_LR,
        checkpoint_dir=checkpoint_dir
    )

    sae_embeddings = sae.get_activations(raw_embeddings)

    # L2 normalize
    sae_embeddings = sae_embeddings / (np.linalg.norm(sae_embeddings, axis=1, keepdims=True) + 1e-8)

    avg_active = (np.abs(sae_embeddings) > 1e-6).sum(axis=1).mean()
    print(f"Average active features per item: {avg_active:.2f}")
    print(f"Output shape: {sae_embeddings.shape}")

    return sae_embeddings, sae


def get_embeddings(embedding_type='pca', dim=48, k_sparsity=4, benchmark='colbench',
                   force_recompute=False):
    """
    Get embeddings of specified type, computing if necessary.

    Args:
        embedding_type: 'raw', 'pca', or 'sae'
        dim: embedding dimension (for pca/sae)
        k_sparsity: sparsity parameter (for sae only)
        benchmark: 'helm' or 'colbench'
        force_recompute: if True, recompute even if cached

    Returns:
        embeddings: numpy array
        task_ids: list of task identifiers
        metadata: dict with embedding info
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for cached embeddings
    if embedding_type == 'raw':
        cache_file = os.path.join(OUTPUT_DIR, f'{benchmark}_raw.pkl')
    elif embedding_type == 'pca':
        cache_file = os.path.join(OUTPUT_DIR, f'{benchmark}_pca_{dim}.pkl')
    elif embedding_type == 'sae':
        cache_file = os.path.join(OUTPUT_DIR, f'{benchmark}_sae_{dim}_k{k_sparsity}.pkl')
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['embeddings'], cached['task_ids'], cached['metadata']

    # Load raw embeddings
    raw_embeddings, task_ids = load_raw_embeddings(benchmark)

    # Compute requested embedding type
    if embedding_type == 'raw':
        embeddings = raw_embeddings
        model = None
        metadata = {'type': 'raw', 'dim': raw_embeddings.shape[1]}

    elif embedding_type == 'pca':
        embeddings, model = compute_pca_embeddings(raw_embeddings, n_components=dim)
        metadata = {
            'type': 'pca',
            'dim': dim,
            'explained_variance': model.explained_variance_ratio_.sum()
        }

    elif embedding_type == 'sae':
        embeddings, model = compute_sae_embeddings(
            raw_embeddings,
            m_features=dim,
            k_sparsity=k_sparsity
        )
        metadata = {'type': 'sae', 'dim': dim, 'k_sparsity': k_sparsity}

    # Cache results
    cached = {
        'embeddings': embeddings,
        'task_ids': task_ids,
        'metadata': metadata
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cached, f)
    print(f"Cached embeddings to {cache_file}")

    return embeddings, task_ids, metadata


def align_embeddings_to_tasks(embeddings, task_ids, target_task_ids, benchmark='colbench'):
    """
    Align embeddings to a specific set of task IDs (e.g., from response matrix columns).

    Args:
        embeddings: numpy array of shape (n_items, dim)
        task_ids: list of task IDs corresponding to embeddings
        target_task_ids: list of task IDs to align to
        benchmark: benchmark name for handling naming variations

    Returns:
        aligned_embeddings: numpy array aligned to target_task_ids
    """
    # Build lookup map
    emb_map = {tid: emb for tid, emb in zip(task_ids, embeddings)}

    # Handle ColBench naming variations
    if benchmark == 'colbench':
        for tid, emb in list(emb_map.items()):
            if tid.startswith('colbench_backend_programming'):
                suffix = tid.split('.')[-1]
                emb_map[f'colbench.{suffix}'] = emb

    # Align to target
    aligned = []
    dim = embeddings.shape[1]
    missing_count = 0

    for target_id in target_task_ids:
        emb = emb_map.get(str(target_id))
        if emb is None and target_id.startswith('colbench.'):
            number = target_id.split('.')[-1]
            emb = emb_map.get(f'colbench_backend_programming.{number}')
        if emb is None:
            emb = np.zeros(dim, dtype=np.float32)
            missing_count += 1
        aligned.append(emb)

    if missing_count > 0:
        print(f"Warning: {missing_count}/{len(target_task_ids)} tasks have missing embeddings (using zeros)")

    return np.stack(aligned)


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings')
    parser.add_argument('--type', type=str, default='pca', choices=['raw', 'pca', 'sae'],
                        help='Embedding type (default: pca)')
    parser.add_argument('--dim', type=int, default=48,
                        help='Embedding dimension for pca/sae (default: 48)')
    parser.add_argument('--k', type=int, default=4,
                        help='SAE sparsity parameter (default: 4)')
    parser.add_argument('--benchmark', type=str, default='colbench', choices=['helm', 'colbench'],
                        help='Benchmark dataset (default: colbench)')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation even if cached')
    args = parser.parse_args()

    embeddings, task_ids, metadata = get_embeddings(
        embedding_type=args.type,
        dim=args.dim,
        k_sparsity=args.k,
        benchmark=args.benchmark,
        force_recompute=args.force
    )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Type: {metadata['type']}")
    print(f"Shape: {embeddings.shape}")
    print(f"Tasks: {len(task_ids)}")


if __name__ == '__main__':
    main()
