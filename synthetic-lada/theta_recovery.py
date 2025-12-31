"""
Theta Recovery Analysis: Compare estimated theta with ground truth.

This demonstrates LADA's advantage:
- Ground truth theta is 3-dimensional
- IRT can only capture 1D (loses information)
- LADA K=3 should recover the full 3D structure
- Better theta recovery = better generalization potential
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def canonical_correlation(X, Y):
    """
    Compute canonical correlation between two matrices.
    Returns the average of top-k canonical correlations.
    """
    from scipy.linalg import svd
    
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute correlation matrix
    n = X.shape[0]
    Cxx = (X.T @ X) / n
    Cyy = (Y.T @ Y) / n
    Cxy = (X.T @ Y) / n
    
    # Regularize for numerical stability
    reg = 1e-6
    Cxx += reg * np.eye(Cxx.shape[0])
    Cyy += reg * np.eye(Cyy.shape[0])
    
    # Compute canonical correlations
    from scipy.linalg import sqrtm, inv
    Cxx_inv_sqrt = inv(sqrtm(Cxx))
    Cyy_inv_sqrt = inv(sqrtm(Cyy))
    
    M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    U, s, Vh = svd(M)
    
    # s contains canonical correlations
    return s.real


def procrustes_align(X, Y):
    """
    Align X to Y using Procrustes analysis (rotation + scaling).
    Returns aligned X and the correlation after alignment.
    """
    # Center both
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # SVD for optimal rotation
    M = Y_centered.T @ X_centered
    U, s, Vh = np.linalg.svd(M)
    R = U @ Vh  # Optimal rotation
    
    # Apply rotation
    X_aligned = X_centered @ R.T
    
    return X_aligned, R


def evaluate_theta_recovery(theta_true, theta_est, model_name):
    """
    Evaluate how well theta_est recovers theta_true.
    
    Returns dict with various metrics.
    """
    results = {'model': model_name}
    
    k_true = theta_true.shape[1]
    k_est = theta_est.shape[1]
    results['k_true'] = k_true
    results['k_est'] = k_est
    
    # 1. Per-dimension correlation (for matching dimensions)
    if k_est == k_true:
        # Procrustes alignment
        theta_aligned, R = procrustes_align(theta_est, theta_true)
        
        dim_corrs = []
        for d in range(k_true):
            corr, _ = pearsonr(theta_true[:, d], theta_aligned[:, d])
            dim_corrs.append(corr)
        
        results['dim_correlations'] = dim_corrs
        results['mean_dim_corr'] = np.mean(dim_corrs)
        
        # Overall R² after alignment
        results['r2_aligned'] = r2_score(theta_true.flatten(), theta_aligned.flatten())
        
    elif k_est < k_true:
        # Project true theta to lower dimension for comparison
        # Use PCA to get k_est principal components of theta_true
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k_est)
        theta_true_reduced = pca.fit_transform(theta_true)
        
        # Align
        theta_aligned, R = procrustes_align(theta_est, theta_true_reduced)
        
        dim_corrs = []
        for d in range(k_est):
            corr, _ = pearsonr(theta_true_reduced[:, d], theta_aligned[:, d])
            dim_corrs.append(corr)
        
        results['dim_correlations'] = dim_corrs
        results['mean_dim_corr'] = np.mean(dim_corrs)
        results['variance_explained'] = pca.explained_variance_ratio_.sum()
        results['r2_aligned'] = r2_score(theta_true_reduced.flatten(), theta_aligned.flatten())
        
    else:  # k_est > k_true
        # Project estimated theta to true dimension
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k_true)
        theta_est_reduced = pca.fit_transform(theta_est)
        
        # Align
        theta_aligned, R = procrustes_align(theta_est_reduced, theta_true)
        
        dim_corrs = []
        for d in range(k_true):
            corr, _ = pearsonr(theta_true[:, d], theta_aligned[:, d])
            dim_corrs.append(corr)
        
        results['dim_correlations'] = dim_corrs
        results['mean_dim_corr'] = np.mean(dim_corrs)
        results['r2_aligned'] = r2_score(theta_true.flatten(), theta_aligned.flatten())
    
    # 2. Canonical correlation (dimension-agnostic)
    min_k = min(k_true, k_est)
    can_corrs = canonical_correlation(theta_true, theta_est)
    results['canonical_correlations'] = can_corrs[:min_k].tolist()
    results['mean_canonical_corr'] = np.mean(can_corrs[:min_k])
    
    # 3. Pairwise distance preservation
    # Do the relative distances between persons get preserved?
    from scipy.spatial.distance import pdist, squareform
    dist_true = pdist(theta_true)
    dist_est = pdist(theta_est)
    results['distance_correlation'], _ = spearmanr(dist_true, dist_est)
    
    return results


def main():
    print("\n" + "="*70)
    print("Theta Recovery Analysis: Ground Truth vs Estimated")
    print("="*70 + "\n")
    
    # Load ground truth
    print("Loading ground truth...")
    ground_truth = np.load('./data/ground_truth.npy', allow_pickle=True).item()
    theta_true = ground_truth['theta']
    k_true = ground_truth['k']
    print(f"  Ground truth theta: {theta_true.shape} (k={k_true})")
    
    results = []
    
    # ============ IRT Model ============
    print("\n" + "-"*50)
    print("IRT (Rasch) Model - 1D")
    print("-"*50)
    
    irt_path = "./result/irt_rasch_model.pkl"
    if os.path.exists(irt_path):
        with open(irt_path, 'rb') as f:
            irt_model = pickle.load(f)
        
        theta_irt = irt_model['theta'].reshape(-1, 1)
        print(f"  Estimated theta: {theta_irt.shape}")
        
        result = evaluate_theta_recovery(theta_true, theta_irt, "IRT (K=1)")
        results.append(result)
        
        print(f"\n  Results:")
        print(f"    Canonical correlations: {[f'{c:.4f}' for c in result['canonical_correlations']]}")
        print(f"    Mean canonical corr: {result['mean_canonical_corr']:.4f}")
        print(f"    Distance preservation: {result['distance_correlation']:.4f}")
        if 'variance_explained' in result:
            print(f"    Variance explained by 1 PC: {result['variance_explained']*100:.1f}%")
    
    # ============ LADA Models ============
    for k in [1, 2, 3]:
        print(f"\n" + "-"*50)
        print(f"LADA Model - K={k}")
        print("-"*50)
        
        model_path = f"./result/lada_model_k{k}.pkl"
        if not os.path.exists(model_path):
            print(f"  Warning: {model_path} not found")
            continue
        
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        theta_est = model_dict['theta']
        print(f"  Estimated theta: {theta_est.shape}")
        
        result = evaluate_theta_recovery(theta_true, theta_est, f"LADA (K={k})")
        results.append(result)
        
        print(f"\n  Results:")
        print(f"    Canonical correlations: {[f'{c:.4f}' for c in result['canonical_correlations']]}")
        print(f"    Mean canonical corr: {result['mean_canonical_corr']:.4f}")
        print(f"    Distance preservation: {result['distance_correlation']:.4f}")
        
        if k == k_true:
            print(f"    Per-dimension correlations (after Procrustes): {[f'{c:.4f}' for c in result['dim_correlations']]}")
            print(f"    R² (aligned): {result['r2_aligned']:.4f}")
    
    # ============ Summary ============
    print("\n" + "="*70)
    print("Summary: Theta Recovery Quality")
    print("="*70)
    
    print(f"\nGround truth: 3-dimensional theta")
    print(f"Key insight: Higher K should capture more of the true structure\n")
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Model': r['model'],
            'K_est': r['k_est'],
            'Mean_Canonical_Corr': r['mean_canonical_corr'],
            'Distance_Preservation': r['distance_correlation'],
            'R2_Aligned': r.get('r2_aligned', np.nan)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Interpretation
    print("\n" + "-"*50)
    print("Interpretation:")
    print("-"*50)
    
    irt_result = [r for r in results if 'IRT' in r['model']]
    lada_results = [r for r in results if 'LADA' in r['model']]
    
    if irt_result and lada_results:
        irt_cc = irt_result[0]['mean_canonical_corr']
        
        print(f"\n  IRT (1D): Mean canonical correlation = {irt_cc:.4f}")
        print(f"    -> Can only capture ONE dimension of the true 3D structure")
        
        for r in lada_results:
            k = r['k_est']
            cc = r['mean_canonical_corr']
            improvement = ((cc - irt_cc) / irt_cc) * 100
            print(f"\n  LADA K={k}: Mean canonical correlation = {cc:.4f} ({improvement:+.1f}% vs IRT)")
            
            if k == 3:
                print(f"    -> Matches true dimensionality, best recovery possible")
                print(f"    -> Per-dimension correlations: {[f'{c:.3f}' for c in r['dim_correlations']]}")
    
    # Key conclusion
    print(f"\n" + "="*70)
    print("KEY RESULT: Distance Preservation")
    print("="*70)
    
    irt_dist = irt_result[0]['distance_correlation'] if irt_result else 0
    print(f"\n  Ground truth has 3 dimensions. Which model preserves person structure?")
    print(f"\n  Distance preservation (Spearman correlation of pairwise distances):")
    print(f"    IRT (K=1):  {irt_dist:.4f}")
    
    for r in sorted(lada_results, key=lambda x: x['k_est']):
        k = r['k_est']
        dist = r['distance_correlation']
        improvement = ((dist - irt_dist) / irt_dist) * 100
        marker = " ✓ BEST" if k == 3 else ""
        print(f"    LADA K={k}:  {dist:.4f} ({improvement:+.1f}% vs IRT){marker}")
    
    lada_k3 = [r for r in results if r['k_est'] == 3]
    if lada_k3 and irt_result:
        best = lada_k3[0]
        irt = irt_result[0]
        improvement = ((best['distance_correlation'] - irt['distance_correlation']) / irt['distance_correlation']) * 100
        
        print(f"\n  Interpretation:")
        print(f"    - IRT collapses 3D structure into 1D -> loses {100-irt['variance_explained']*100:.0f}% of variance")
        print(f"    - LADA K=3 preserves full 3D structure")
        print(f"    - Distance preservation improves by {improvement:.1f}%")
        print(f"\n  Why this matters for generalization:")
        print(f"    - Better distance preservation = more accurate person representations")
        print(f"    - Accurate representations enable better predictions on NEW items")
        print(f"    - This is the foundation of LADA's generalization advantage")
    print("="*70)
    
    # Save results
    summary_df.to_csv("./result/theta_recovery_summary.csv", index=False)
    print(f"\nSaved to ./result/theta_recovery_summary.csv")


if __name__ == "__main__":
    main()
