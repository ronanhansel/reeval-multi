import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from factor_analyzer.rotator import Rotator

def _get_cache_paths(model_path, rotation_method, top_k):
    """Generate cache file paths based on model path and rotation parameters."""
    model_name = Path(model_path).stem  # e.g., "mirt_model_k19"
    cache_dir = Path('./output/targeted_rotation')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    rotation_str = "none" if rotation_method is None else rotation_method
    cache_prefix = f"{model_name}_pc1_{rotation_str}_top{top_k}"
    
    return {
        'theta': cache_dir / f"{cache_prefix}_theta.pkl",
        'a': cache_dir / f"{cache_prefix}_a.pkl",
        'top_items': cache_dir / f"{cache_prefix}_top_items.pkl",
        'Q_final': cache_dir / f"{cache_prefix}_Q_final.npy"
    }

def _cache_exists(cache_paths):
    """Check if all cache files exist."""
    return all(path.exists() for path in cache_paths.values())

def _load_from_cache(cache_paths):
    """Load cached results."""
    print("--- Loading from Cache ---")
    theta_df = pd.read_pickle(cache_paths['theta'])
    a_df = pd.read_pickle(cache_paths['a'])
    top_items_df = pd.read_pickle(cache_paths['top_items'])
    Q_final = np.load(cache_paths['Q_final'])
    print(f"Loaded cached theta shape: {theta_df.shape}")
    print(f"Loaded cached a shape: {a_df.shape}")
    print(f"Loaded cached Q_final shape: {Q_final.shape}\n")
    return theta_df, a_df, top_items_df, Q_final

def _save_to_cache(cache_paths, theta_df, a_df, top_items_df, Q_final):
    """Save results to cache."""
    print("--- Saving to Cache ---")
    theta_df.to_pickle(cache_paths['theta'])
    a_df.to_pickle(cache_paths['a'])
    top_items_df.to_pickle(cache_paths['top_items'])
    np.save(cache_paths['Q_final'], Q_final)
    print(f"Cached results saved to {cache_paths['theta'].parent}\n")

def load_and_rotate_pc1_targeted(model_path='./output/mirt_model_k19.pt', 
                                rotation_method='oblimax', 
                                top_k=20,
                                model_names=None,
                                item_names=None):
    """
    Perform targeted rotation aligning first axis with PC1 from theta, 
    optionally rotating remaining axes for interpretability.
    Uses caching to avoid recomputation.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved MIRT model
    rotation_method : str or None
        Method for rotating remaining axes ('oblimax', 'varimax', etc.)
        Set to None to skip subspace rotation
    top_k : int
        Number of top items to extract per factor
    model_names : list, optional
        Names for the models/persons (will use indices if None)
    item_names : list, optional
        Names for the items (will use indices if None)
        
    Returns:
    --------
    theta_df : pd.DataFrame
        Rotated person parameters with PC1 as first factor
    a_df : pd.DataFrame  
        Rotated item parameters with PC1 as first factor
    top_items_df : pd.DataFrame
        Top items per factor for interpretation
    Q_final : np.ndarray
        Final rotation matrix applied
    """
    # Check cache first
    cache_paths = _get_cache_paths(model_path, rotation_method, top_k)
    if _cache_exists(cache_paths):
        return _load_from_cache(cache_paths)
    
    print("--- Cache not found, computing targeted rotation ---")
    
    # 1) Load parameters
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    theta = model_data['theta'].detach().cpu().numpy()  # shape: (n_people, k)
    a = model_data['a'].detach().cpu().numpy()          # shape: (n_items,  k)
    
    n_items, D = a.shape
    n_persons = theta.shape[0]
    
    print("--- Initial Loaded Data ---")
    print(f"Original theta shape: {theta.shape}")
    print(f"Original 'a' matrix shape: {a.shape}\n")
    
    # Generate default names if not provided
    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_persons)]
    if item_names is None:
        item_names = [f"item_{i}" for i in range(n_items)]
    
    # 2) Get PCA PC1 from theta
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(theta)[:, 0]
    pc1_vec = pca.components_[0]
    pc1_unit = pc1_vec / (np.linalg.norm(pc1_vec) + 1e-12)
    
    # 3) Build orthonormal basis Q with first column = pc1_unit
    rng = np.random.RandomState(12345)
    random_mat = rng.normal(size=(D, D-1))
    M = np.column_stack([pc1_unit.reshape(-1, 1), random_mat])
    Q, R_q = np.linalg.qr(M)
    
    # Fix sign if needed
    if np.dot(Q[:, 0], pc1_unit) < 0:
        Q[:, 0] *= -1.0
    
    orth_err = np.max(np.abs(Q.T @ Q - np.eye(D)))
    print(f"Orthonormality error: {orth_err:.2e}")
    
    # 4) Rotate a and theta with Q to align axis 0 with PC1
    a_q = a @ Q
    theta_q = theta @ Q
    
    # Verify alignment
    corr_pc1 = np.corrcoef(theta_q[:, 0], pc1_scores)[0, 1]
    print(f"Correlation between theta_q[:,0] and PCA PC1 scores: {corr_pc1:.6f}")
    
    # 5) Optionally rotate remaining D-1 axes for interpretability
    if rotation_method is not None:
        print(f"Applying {rotation_method} rotation to remaining axes...")
        A_sub = a_q[:, 1:]  # items x (D-1)
        rot = Rotator(method=rotation_method)
        A_sub_rot = rot.fit_transform(A_sub)
        R_sub = rot.rotation_
        
        # Build final rotation matrix
        Q_block = np.eye(D)
        Q_block[1:, 1:] = R_sub
        Q_final = Q @ Q_block
        
        a_final = a @ Q_final
        theta_final = theta @ Q_final
        print(f"Applied {rotation_method} on the orthogonal complement.")
    else:
        a_final = a_q.copy()
        theta_final = theta_q.copy()
        Q_final = Q.copy()
        print("No subspace rotation applied.")
    
    # 6) Flip factor signs for interpretability
    # This logic orients each factor based on its own loadings.
    # A common heuristic is to ensure the sum of loadings is positive.
    
    # We will modify a_final and theta_final in place
    a_used = a_final.copy()
    theta_used = theta_final.copy()
    
    print("\nOrienting factors for interpretability...")
    for d in range(D):
        # Calculate the sum of all item loadings on this factor.
        loading_sum = np.sum(a_used[:, d])
        
        # If the sum is negative, the factor is likely reversed.
        if loading_sum < 0:
            # Flip the signs for both the loadings and the scores to correct it.
            a_used[:, d] *= -1.0
            theta_used[:, d] *= -1.0
            print(f"Flipped Factor {d+1} (sum of loadings was {loading_sum:.2f})")
    
    # Now a_used and theta_used contain the final, interpretable results.
    
    # 7) Create DataFrames and extract top items
    factor_names = [f"F{d+1}" for d in range(D)]
    theta_df = pd.DataFrame(theta_used, index=model_names, columns=factor_names)
    a_df = pd.DataFrame(a_used, index=item_names, columns=factor_names)
    
    # Top items per factor
    top_items = {}
    for d in range(D):
        order = np.argsort(-np.abs(a_used[:, d]))[:top_k]
        top_items[f"F{d+1}"] = [(item_names[i], float(a_used[i, d])) for i in order]
    
    top_items_df = pd.DataFrame([
        {"factor": k, "top_items": "; ".join([f"{name}({load:.3f})" for name, load in v])}
        for k, v in top_items.items()
    ])
    
    # Save to cache
    _save_to_cache(cache_paths, theta_df, a_df, top_items_df, Q_final)
    
    print("Done. Results computed and cached.")
    return theta_df, a_df, top_items_df, Q_final

def load_and_rotate_pc1_reckase(model_path='./output/mirt_model_k19.pt', 
                               top_k=20,
                               model_names=None,
                               item_names=None):
    """
    Perform Reckase-style targeted rotation:
    Align the first axis with PC1 and apply the same rotation 
    to the entire factor space. No subspace rotation.
    """
    # Load model
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    theta = model_data['theta'].detach().cpu().numpy()
    a = model_data['a'].detach().cpu().numpy()
    
    n_items, D = a.shape
    n_persons = theta.shape[0]
    
    # PCA PC1
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(theta)[:, 0]
    pc1_vec = pca.components_[0]
    pc1_unit = pc1_vec / (np.linalg.norm(pc1_vec) + 1e-12)
    
    # Construct rotation matrix to align F1 with PC1
    rng = np.random.RandomState(12345)
    random_mat = rng.normal(size=(D, D-1))
    M = np.column_stack([pc1_unit.reshape(-1, 1), random_mat])
    Q, _ = np.linalg.qr(M)
    
    # Fix sign
    if np.dot(Q[:, 0], pc1_unit) < 0:
        Q[:, 0] *= -1.0
    
    # Apply rotation to all axes
    a_rot = a @ Q
    theta_rot = theta @ Q
    
    # Orient factors for interpretability
    for d in range(D):
        if np.sum(a_rot[:, d]) < 0:
            a_rot[:, d] *= -1
            theta_rot[:, d] *= -1
    
    # Build DataFrames
    factor_names = [f"F{d+1}" for d in range(D)]
    theta_df = pd.DataFrame(theta_rot, index=model_names, columns=factor_names)
    a_df = pd.DataFrame(a_rot, index=item_names, columns=factor_names)
    
    # Extract top items
    top_items = {}
    for d in range(D):
        order = np.argsort(-np.abs(a_rot[:, d]))[:top_k]
        top_items[f"F{d+1}"] = [(item_names[i], float(a_rot[i, d])) for i in order]
    
    top_items_df = pd.DataFrame([
        {"factor": k, "top_items": "; ".join([f"{name}({load:.3f})" for name, load in v])}
        for k, v in top_items.items()
    ])
    
    return theta_df, a_df, top_items_df, Q
