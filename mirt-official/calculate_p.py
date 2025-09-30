import numpy as np
from numpy.polynomial.hermite import hermgauss

def compute_eap_map_sd(Y, A, b, n_points=21):
    """
    Compute 2D EAP, MAP, and posterior SD estimates using Gauss-Hermite quadrature.

    Parameters
    ----------
    Y : array (n_persons, n_items)
        Binary response matrix.
    A : array (n_items, k)
        Discrimination/loadings matrix.
    b : array (n_items,)
        Difficulty parameters.
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
    k = A.shape[1]
    assert k == 2, "This function is written for K=2."

    # 1. Gauss–Hermite quadrature nodes & weights
    nodes_1d, weights_1d = hermgauss(n_points)
    nodes_1d = nodes_1d * np.sqrt(2)       # rescale for N(0,1)
    weights_1d = weights_1d / np.sqrt(np.pi)

    # 2. Tensor grid for k=2
    grid = np.array(np.meshgrid(nodes_1d, nodes_1d)).T.reshape(-1, k)    # (n_points^2, 2)
    weights = np.outer(weights_1d, weights_1d).ravel()                   # (n_points^2,)

    # 3. Precompute item probabilities at all grid points
    linpred = (A @ grid.T).T - b    # (n_points^2, n_items)
    P = 1 / (1 + np.exp(-linpred))  # logistic

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
