"""
Utility functions for IRT model evaluation.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import torch

def get_valid_item_mask(y_target_matrix):
    """
    Identify items (columns) that have non-zero variance.
    If an item has 100% failure (or 100% success), it does not differentiate capability
    and should be excluded from evaluation metrics to prevent variance collapse paradox.
    
    Args:
        y_target_matrix: Tensor or array of shape (N, M)
        
    Returns:
        valid_item_mask: Boolean tensor of shape (M,)
    """
    if isinstance(y_target_matrix, np.ndarray):
        y_tensor = torch.from_numpy(y_target_matrix)
    else:
        y_tensor = y_target_matrix
        
    # Valid if max is materially different from min
    diff = y_tensor.max(dim=0).values - y_tensor.min(dim=0).values
    return diff > 1e-5

def compute_rmse(predictions, targets, mask, item_mask=None):
    """Compute RMSE over masked entries. Optionally filter by valid items."""
    if item_mask is not None:
        if isinstance(item_mask, torch.Tensor):
            item_mask = item_mask.cpu().numpy()
        mask = mask.copy() if isinstance(mask, np.ndarray) else mask.clone()
        if isinstance(mask, np.ndarray):
            mask[:, ~item_mask] = False
        else:
            mask[:, ~torch.tensor(item_mask, device=mask.device)] = False

    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    if not valid.any():
        return 0.0
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

def evaluate_auc(y_pred_tensor, y_true_tensor, mask_tensor, item_mask=None):
    """Compute AUC over masked entries. Optionally filter by valid items."""
    if item_mask is not None:
        mask_tensor = mask_tensor.clone()
        if not isinstance(item_mask, torch.Tensor):
            item_mask = torch.tensor(item_mask, device=mask_tensor.device)
        mask_tensor[:, ~item_mask] = False

    valid_mask = mask_tensor.detach()
    if not valid_mask.any():
        return 0.5
        
    y_true_flat = y_true_tensor[valid_mask].detach().cpu().numpy()
    y_pred_flat = y_pred_tensor[valid_mask].detach().cpu().numpy()
    
    y_true_binary = (y_true_flat > 0.5).astype(int)
    if len(np.unique(y_true_binary)) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true_binary, y_pred_flat)
    except ValueError:
        return 0.5
