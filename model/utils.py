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
    if not isinstance(y_pred_tensor, torch.Tensor):
        y_pred_tensor = torch.as_tensor(y_pred_tensor)
    pred_device = y_pred_tensor.device

    if not isinstance(y_true_tensor, torch.Tensor):
        y_true_tensor = torch.as_tensor(y_true_tensor)
    y_true_tensor = y_true_tensor.to(pred_device)

    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.as_tensor(mask_tensor)
    mask_tensor = mask_tensor.to(pred_device).bool()

    if item_mask is not None:
        mask_tensor = mask_tensor.clone()
        if not isinstance(item_mask, torch.Tensor):
            item_mask = torch.as_tensor(item_mask, device=mask_tensor.device)
        else:
            item_mask = item_mask.to(mask_tensor.device)
        item_mask = item_mask.bool()
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

import os
import pandas as pd

def aggregate_remediation_resmats(benchmark_dir, benchmark_name):
    """
    Legacy function: aggregates all remediation runs into a single averaged matrix.
    """
    iters = get_benchmark_iterations(benchmark_dir, benchmark_name)
    if not iters:
        return None
        
    df_base = iters[0].copy()
    if len(iters) > 1:
        remediation_concat = pd.concat(iters[1:])
        remediation_avg = remediation_concat.groupby(remediation_concat.index).mean()
        df_base.update(remediation_avg)
    return df_base

def get_benchmark_iterations(benchmark_dir, benchmark_name):
    """
    Returns a list of dataframes where each is df_base.update(df_remediation_i).
    Iteration 0 is the base (index_0) with NO updates.
    Iterations 1..N are base updated with remediation 1..N.
    """
    files = [f for f in os.listdir(benchmark_dir) if f.startswith('resmat')]
    
    base_file = None
    for f in files:
        if (f.endswith('0.csv') and not f[-6].isdigit()) or 'original' in f:
            base_file = f
            break
            
    if base_file is None:
        return []
        
    df_base = pd.read_csv(os.path.join(benchmark_dir, base_file), index_col=0)
    
    # [CRITICAL DATA FIX]: Base dataset parsing occasionally loads aggregated columns from multiple datasets 
    # (e.g., SAB generation traces mistakenly embed SciCode tasks too due to logging intersections). 
    # To prevent item-size explosion across iterations, cleanly drop any non-benchmark columns.
    valid_cols = [c for c in df_base.columns if str(c).startswith(benchmark_name)]
    if not valid_cols:
        return []
    df_base = df_base[valid_cols]
    
    df_base.columns = [
        f"{benchmark_name}.{c}" 
        if not str(c).startswith(benchmark_name) and not str(c).startswith(benchmark_name.replace('_hard','')) 
        else c for c in df_base.columns
    ]
    
    results = [df_base.copy()]
    
    # Sort files to ensure deterministic order if needed, but not strictly required
    for f in sorted(files):
        if f != base_file:
            df_rem = pd.read_csv(os.path.join(benchmark_dir, f), index_col=0)
            valid_rem_cols = [c for c in df_rem.columns if str(c).startswith(benchmark_name)]
            if valid_rem_cols:
                df_rem = df_rem[valid_rem_cols]
                df_rem.columns = [
                    f"{benchmark_name}.{c}" 
                    if not str(c).startswith(benchmark_name) and not str(c).startswith(benchmark_name.replace('_hard','')) 
                    else c for c in df_rem.columns
                ]
                
                df_iter = df_base.copy()
                df_iter.update(df_rem)
                results.append(df_iter)
            
    return results

