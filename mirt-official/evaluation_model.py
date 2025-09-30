import numpy as np
import math
import pandas as pd
import scipy.stats as st

# -------------------
# Rasch (1PL) helpers
# -------------------
def safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, 1 - eps))

def rasch_prob(theta, b):
    """Rasch model probability: logistic(theta - b)"""
    theta = np.asarray(theta).reshape(-1, 1)  # N x 1
    b = np.asarray(b).reshape(1, -1)          # 1 x J
    logits = theta - b
    return 1 / (1 + np.exp(-logits))

def loglik_rasch(Y, theta, b):
    """Log-likelihood for Rasch model."""
    P = rasch_prob(theta, b)
    mask = ~np.isnan(Y)
    ll = (Y[mask] * safe_log(P[mask]) + (1 - Y[mask]) * safe_log(1 - P[mask])).sum()
    return ll

def aic_bic_from_loglik(loglik, num_params, n_obs):
    """Compute -2LL, AIC, BIC."""
    neg2ll = -2 * loglik
    aic = neg2ll + 2 * num_params
    bic = neg2ll + num_params * math.log(n_obs)
    return {"loglik": loglik, "-2LL": neg2ll, "AIC": aic, "BIC": bic}

# -------------------
# General IRT helpers (MIRT any K)
# -------------------
def irt_prob(theta, a, b):
    """2PL probabilities for MIRT (N x J)."""
    logits = np.dot(theta, a.T) - b[np.newaxis, :]
    return 1 / (1 + np.exp(-logits))

def loglik_mirt(Y, theta, a, b):
    """Log-likelihood for MIRT (any K)."""
    P = irt_prob(theta, a, b)
    mask = ~np.isnan(Y)
    ll = (Y[mask] * safe_log(P[mask]) + (1 - Y[mask]) * safe_log(1 - P[mask])).sum()
    return ll

# -------------------
# Likelihood Ratio Test
# -------------------
def likelihood_ratio_test(ll_small, ll_large, df_small, df_large):
    """Compare nested models using -2LL difference test."""
    delta_neg2ll = -2 * (ll_small - ll_large)
    df = df_large - df_small
    pval = 1 - st.chi2.cdf(delta_neg2ll, df)
    return delta_neg2ll, df, pval

# -------------------
# Extra diagnostics
# -------------------
def per_response_ll(loglik, Y):
    """Log-likelihood per observed response (scalar)."""
    n_obs = Y[~np.isnan(Y)].size
    return loglik / n_obs

def pseudo_r2(ll_model, ll_null):
    """McFadden's pseudo-R² (relative to null)."""
    return 1 - (ll_model / ll_null)