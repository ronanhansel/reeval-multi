import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import softmax

def compute_eap_map_sd_lada(Y, w, d, n_points=21):
    """
    Compute 2D EAP, MAP, and posterior SD estimates using Gauss-Hermite quadrature for LADA models.
    
    In LADA:
    - w (weights) replaces A (discriminations) from MIRT
    - d (difficulties) replaces b from MIRT
    - The probability function is: P(Y=1|theta) = sum_k [ w_k * sigmoid(theta_k - d_k) ]
    
    Parameters
    ----------
    Y : array (n_persons, n_items)
        Binary response matrix.
    w : array (n_items, k)
        Weight matrix (softmax probabilities for each latent dimension per item).
    d : array (n_items, k)
        Difficulty parameters for each dimension.
    n_points : int
        Number of quadrature nodes per dimension.
    
    Returns
    -------
    eap_scores : array (n_persons, k)
        EAP ability estimates.
    map_scores : array (n_persons, k)
        MAP ability estimates.
    sd_scores : array (n_persons, k)
        Posterior standard deviation (uncertainty) estimates.
    """
    
    n_persons, n_items = Y.shape
    k = w.shape[1]
    assert k == 2, "This function is written for K=2."
    
    # 1. Gauss–Hermite quadrature nodes & weights
    nodes_1d, weights_1d = hermgauss(n_points)
    nodes_1d = nodes_1d * np.sqrt(2)       # rescale for N(0,1)
    weights_1d = weights_1d / np.sqrt(np.pi)
    
    # 2. Tensor grid for k=2
    grid = np.array(np.meshgrid(nodes_1d, nodes_1d)).T.reshape(-1, k)    # (n_points^2, 2)
    weights = np.outer(weights_1d, weights_1d).ravel()                   # (n_points^2,)
    
    # 3. Precompute item probabilities at all grid points
    # For LADA: P(correct) = sum_k [ w_ik * sigmoid(theta_k - d_ik) ]
    # grid: (n_points^2, k=2)
    # For each item i, compute across all grid points
    
    P = np.zeros((len(grid), n_items))  # (n_points^2, n_items)
    
    for i in range(n_items):
        # For item i, w[i] is (k,) and d[i] is (k,)
        # grid is (n_points^2, k)
        # theta_k - d_ik for each dimension
        logits = grid - d[i]  # (n_points^2, k)
        probs = 1 / (1 + np.exp(-logits))  # sigmoid, (n_points^2, k)
        # Weighted sum across dimensions
        P[:, i] = np.sum(w[i] * probs, axis=1)  # (n_points^2,)
    
    # Allocate storage
    eap_scores = np.zeros((n_persons, k))
    map_scores = np.zeros((n_persons, k))
    sd_scores  = np.zeros((n_persons, k))
    
    # 4. Loop over persons
    for i in range(n_persons):
        y = Y[i]  # (n_items,)
        
        # Log-likelihood for each grid point
        loglike = np.sum(
            y * np.log(P + 1e-12) + (1 - y) * np.log(1 - P + 1e-12),
            axis=1
        )
        
        # Posterior (log-sum-exp for stability)
        post = np.exp(loglike - loglike.max()) * weights
        if np.sum(post) == 0:  # safeguard for numerical underflow
            post = np.ones_like(post) / len(post)
        else:
            post /= np.sum(post)
        
        # --- EAP ---
        eap = np.sum(grid * post[:, None], axis=0)
        eap_scores[i] = eap
        
        # --- Posterior variance & SD ---
        second_moment = np.sum((grid**2) * post[:, None], axis=0)
        var = second_moment - eap**2
        sd_scores[i] = np.sqrt(np.maximum(var, 0.0))
        
        # --- MAP ---
        map_scores[i] = grid[np.argmax(post)]
    
    return eap_scores, map_scores, sd_scores


def lada_probability(theta, w, d):
    """
    Calculate probability of correct response for LADA model.
    
    P(correct) = sum_k [ w_k * sigmoid(theta_k - d_k) ]
    
    Parameters
    ----------
    theta : array (..., k)
        Ability parameters (can be single person or batch).
    w : array (k,) or (n_items, k)
        Weight vector for the item(s).
    d : array (k,) or (n_items, k)
        Difficulty vector for the item(s).
    
    Returns
    -------
    prob : array (...)
        Probability of correct response.
    """
    # theta: (..., k), w: (k,) or (n_items, k), d: (k,) or (n_items, k)
    logits = theta - d  # (..., k)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    return np.sum(w * probs, axis=-1)  # (...)


def lada_probability_vectorized(theta_grid, w_array, d_array):
    """
    Vectorized calculation of LADA probabilities for multiple items and grid points.
    
    P(correct) = sum_k [ w_k * sigmoid(theta_k - d_k) ]
    
    Parameters
    ----------
    theta_grid : array (n_grid_points, k)
        Grid of ability parameters.
    w_array : array (n_items, k)
        Weight matrix for all items.
    d_array : array (n_items, k)
        Difficulty matrix for all items.
    
    Returns
    -------
    prob : array (n_grid_points, n_items)
        Probability of correct response for each grid point and item.
    """
    # theta_grid: (n_grid, k), w_array: (n_items, k), d_array: (n_items, k)
    # Expand dimensions for broadcasting: theta_grid: (n_grid, 1, k), d_array: (1, n_items, k)
    theta_expanded = theta_grid[:, np.newaxis, :]  # (n_grid, 1, k)
    d_expanded = d_array[np.newaxis, :, :]  # (1, n_items, k)
    w_expanded = w_array[np.newaxis, :, :]  # (1, n_items, k)
    
    # Calculate logits: (n_grid, n_items, k)
    logits = theta_expanded - d_expanded
    
    # Apply sigmoid: (n_grid, n_items, k)
    probs = 1 / (1 + np.exp(-logits))
    
    # Weighted sum across k dimensions: (n_grid, n_items)
    result = np.sum(w_expanded * probs, axis=2)
    
    return result
