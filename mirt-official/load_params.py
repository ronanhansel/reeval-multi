import torch
import numpy as np
from pathlib import Path
from factor_analyzer.rotator import Rotator

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

def load_and_rotate(model_path='./output/mirt_model_k19.pt', rotation='varimax'):
    """
    Rotate the item parameters (a) and person parameters (theta).
    Returns the rotated parameters (theta, a, b) WITHOUT z-scoring.
    Uses caching to avoid recomputation for the same model and rotation method.
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

    # 2) Rotate 'a' (if rotation is specified)
    if rotation is not None:
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
