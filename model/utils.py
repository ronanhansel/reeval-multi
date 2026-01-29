"""
Utility functions for IRT model evaluation.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


def compute_rmse(predictions, targets, mask):
    """Compute RMSE over masked entries."""
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))


def evaluate_auc(y_pred_tensor, y_true_tensor, mask_tensor):
    """Compute AUC over masked entries."""
    y_true_flat = y_true_tensor[mask_tensor].detach().cpu().numpy()
    y_pred_flat = y_pred_tensor[mask_tensor].detach().cpu().numpy()
    y_true_binary = (y_true_flat > 0.5).astype(int)
    if len(np.unique(y_true_binary)) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true_binary, y_pred_flat)
    except ValueError:
        return 0.5
