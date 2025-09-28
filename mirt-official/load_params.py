import torch
import numpy as np
from pathlib import Path
from factor_analyzer.rotator import Rotator
import pickle
import os
import re

def _get_cache_paths(model_path, rotation):
    """Generate cache file paths based on model path and rotation method."""
    model_name = Path(model_path).stem  # e.g., "mirt_model_k19"
    cache_dir = Path('./output')
    cache_dir.mkdir(exist_ok=True)
    
    rotation_str = "none" if rotation is None else rotation
    cache_prefix = f"{model_name}_{rotation_str}"
    
    return {
        'theta': cache_dir / f"{cache_prefix}_theta.npy",
        'a': cache_dir / f"{cache_prefix}_a.npy", 
        'b': cache_dir / f"{cache_prefix}_b.npy"
    }

def _cache_exists(cache_paths):
    """Check if all cache files exist."""
    return all(path.exists() for path in cache_paths.values())

def _load_from_cache(cache_paths):
    """Load cached results."""
    print("--- Loading from Cache ---")
    theta = np.load(cache_paths['theta'])
    a = np.load(cache_paths['a'])
    b = np.load(cache_paths['b'])
    print(f"Loaded cached theta shape: {theta.shape}")
    print(f"Loaded cached a shape: {a.shape}")
    print(f"Loaded cached b shape: {b.shape}\n")
    return theta, a, b

def _save_to_cache(cache_paths, theta, a, b):
    """Save results to cache."""
    print("--- Saving to Cache ---")
    np.save(cache_paths['theta'], theta)
    np.save(cache_paths['a'], a) 
    np.save(cache_paths['b'], b)
    print(f"Cached results saved to {cache_paths['theta'].parent}\n")

def _extract_model_info(model_path):
    """Extract K and repetition number from model path."""
    model_name = Path(model_path).stem
    # Extract K and repetition from filename like "mirt_model_k2_rep87"
    k_match = re.search(r'k(\d+)', model_name)
    rep_match = re.search(r'rep(\d+)', model_name)
    
    k = int(k_match.group(1)) if k_match else None
    rep = int(rep_match.group(1)) if rep_match else None
    
    return k, rep

def _load_procrustes_rotation(model_path):
    """Load the Procrustes rotation matrix for this specific model."""
    k, rep = _extract_model_info(model_path)
    
    if k is None or rep is None:
        print(f"⚠️ Could not extract K and rep from model path: {model_path}")
        return None
    
    # Try to load the rotation matrices for this K
    rotation_file = f"./output/rotation_matrices_k{k}.npz"
    
    if not os.path.exists(rotation_file):
        print(f"⚠️ Rotation matrices file not found: {rotation_file}")
        return None
    
    try:
        rotation_data = np.load(rotation_file, allow_pickle=True)
        rotation_matrices = rotation_data['rotation_matrices']
        reference_a = rotation_data['reference_a']
        
        if rep >= len(rotation_matrices):
            print(f"⚠️ Repetition {rep} not found in rotation matrices (max: {len(rotation_matrices)-1})")
            return None
        
        R = rotation_matrices[rep]
        print(f"✅ Loaded Procrustes rotation matrix for K={k}, rep={rep}")
        return R, reference_a
        
    except Exception as e:
        print(f"⚠️ Error loading rotation matrices: {e}")
        return None

def load_and_rotate(model_path='./output/mirt_model_k19_auc89.pt', rotation='procrustes'):
    """
    Rotate the item parameters (a) and person parameters (theta).
    Returns the rotated parameters (theta, a, b) WITHOUT z-scoring.
    Uses caching to avoid recomputation for the same model and rotation method.
    
    Default rotation method is 'procrustes' which uses the exported rotation matrices
    from dimensional analysis for consistency.
    """
    # Check cache first
    cache_paths = _get_cache_paths(model_path, rotation)
    if _cache_exists(cache_paths):
        return _load_from_cache(cache_paths)
    
    print("--- Cache not found, computing rotation ---")
    
    # 1) Load parameters
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    theta = model_data['theta']          # shape: (n_people, k)
    a = model_data['a']                  # shape: (n_items,  k)
    b = model_data['b']

    print("--- Initial Loaded Data ---")
    print(f"Original theta shape: {theta.shape}")
    print(f"Original 'a' matrix shape: {a.shape}\n")

    a_numpy = a.detach().cpu().numpy()
    theta_numpy = theta.detach().cpu().numpy()
    b_numpy = b.detach().cpu().numpy()

    # 2) Apply rotation
    if rotation == 'procrustes':
        # Use the exported Procrustes rotation matrices from dimensional analysis
        rotation_result = _load_procrustes_rotation(model_path)
        
        if rotation_result is not None:
            R, reference_a = rotation_result
            a_rotated_numpy = a_numpy @ R
            theta_transformed_numpy = theta_numpy @ R
            
            print("--- After Procrustes Rotation (from dimensional analysis) ---")
            print(f"Rotated 'a' matrix shape: {a_rotated_numpy.shape}")
            print(f"Transformed theta shape: {theta_transformed_numpy.shape}\n")
        else:
            # Fallback to no rotation if Procrustes matrices not found
            print("⚠️ Procrustes rotation matrices not found, using original parameters")
            a_rotated_numpy = a_numpy.copy()
            theta_transformed_numpy = theta_numpy.copy()
            
    elif rotation is not None:
        # Use factor analyzer for other rotation methods
        rotator = Rotator(method=rotation) 
        a_rotated_numpy = rotator.fit_transform(a_numpy)
        print("--- After Rotation of 'a' ---")
        print(f"Rotated 'a' matrix shape: {a_rotated_numpy.shape}\n")

        # 3) Transform theta using the SAME rotation matrix from Rotator
        T = rotator.rotation_
        theta_transformed_numpy = theta_numpy @ T

        print("--- After Transformation of 'theta' ---")
        print(f"Transformed theta shape: {theta_transformed_numpy.shape}\n")
    else:
        # No rotation - use original parameters
        a_rotated_numpy = a_numpy.copy()
        theta_transformed_numpy = theta_numpy.copy()
        print("--- No Rotation Applied ---")
        print(f"Using original 'a' matrix shape: {a_rotated_numpy.shape}")
        print(f"Using original theta shape: {theta_transformed_numpy.shape}\n")

    # 3) Keep raw rotated values (no z-scoring)
    theta_final = theta_transformed_numpy
    a_final = a_rotated_numpy
    b_final = b_numpy

    # Save results to cache for future use
    _save_to_cache(cache_paths, theta_final, a_final, b_final)
    
    return theta_final, a_final, b_final
