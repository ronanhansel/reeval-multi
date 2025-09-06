import torch
import numpy as np
from factor_analyzer.rotator import Rotator

def load_and_rotate(model_path='./output/mirt_model_k19.pt'):
    """
    Rotate the item parameters (a) and person parameters (theta) using Varimax rotation.
    Returns the rotated and standardized parameters (theta, a, b).
    """
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

    # 2) Rotate 'a'
    rotator = Rotator(method='varimax')  # orthogonal
    a_rotated_numpy = rotator.fit_transform(a_numpy)
    print("--- After Rotation of 'a' ---")
    print(f"Rotated 'a' matrix shape: {a_rotated_numpy.shape}\n")

    # 3) Transform theta using the SAME rotation matrix from Rotator
    T = rotator.rotation_                # <- correct attribute on Rotator
    theta_transformed_numpy = theta_numpy @ T  # for orthogonal varimax, use R (not inv(R))

    print("--- After Transformation of 'theta' ---")
    print(f"Transformed theta shape: {theta_transformed_numpy.shape}\n")

    # (Optional) Sanity check: inner products preserved (up to numerical tolerance)
    # err = np.max(np.abs(a_numpy @ theta_numpy.T - a_rotated_numpy @ theta_transformed_numpy.T))
    # print(f"Max abs diff in item-person inner products: {err:.3e}")

    # 4) Standardize transformed theta
    # Z-score theta for interpretability
    m = theta_transformed_numpy.mean(axis=0)           # shape (K,)
    s = theta_transformed_numpy.std(axis=0)            # shape (K,)
    eps = 1e-8
    D = np.diag(np.maximum(s, eps))
    theta_z_scores_numpy = (theta_transformed_numpy - m) / np.maximum(s, eps)

    # If you intend to use theta_z for prediction, adjust a and b:
    a_for_z_numpy = a_rotated_numpy @ D                # (items x K)
    b_for_z_numpy = b.detach().cpu().numpy() - a_rotated_numpy @ m  # (items,)

    print("--- Final Standardized Z-Scores (from transformed theta) ---")
    print("These are the scores you should use for interpretation.")
    print(theta_z_scores_numpy)
    return theta_z_scores_numpy, a_for_z_numpy, b_for_z_numpy