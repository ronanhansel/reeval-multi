import sys, os, pandas as pd, torch
from model.amortized_irt import load_data, prepare_experiment_data, AmortizedIRTModel, train_amortized_irt
all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data('sae', 48)
data = prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map)
import numpy as np

N = len(global_shared_indices)

train_target_df = all_dfs[0].reindex(index=global_shared_indices)
all_columns = sorted(list(set().union(*[df.columns for df in all_dfs])))
train_target_df = train_target_df.reindex(columns=all_columns)

train_values = np.nan_to_num(train_target_df.values, nan=0.5)
y_train = torch.from_numpy(train_values.astype(np.float32)).to('cuda')

train_mask_current = np.zeros_like(train_target_df.values, dtype=bool)
train_mask_current[:, data['train_idx']] = ~np.isnan(train_target_df.values)[:, data['train_idx']]
train_mask_current_t = torch.from_numpy(train_mask_current).to('cuda')

model = AmortizedIRTModel(N, data['J'], 64, 48, data['x_j'], dropout=0.5).to('cuda')

import torch.optim as optim

optimizer = optim.AdamW([
    {'params': [model.theta, model.theta_bias], 'lr': 0.01, 'weight_decay': 0.0},
    {'params': [model.W, model.tau_raw, model.global_bias], 'lr': 0.002, 'weight_decay': 0.0},
    {'params': model.difficulty_proj.parameters(), 'lr': 0.002}
])

beta_phi = 10.0
eps = 1e-6
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    probs = model()
    p = probs[train_mask_current_t].clamp(eps, 1 - eps)
    y = y_train[train_mask_current_t].clamp(eps, 1 - eps)
    dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
    loss_fit = -dist.log_prob(y).mean()
    loss_fit.backward()
    print(f"Epoch {epoch} loss: {loss_fit.item()}")
    print(f"tau_raw mean grad: {model.tau_raw.grad.mean().item():.6f}, min grad: {model.tau_raw.grad.min().item():.6f}, max grad: {model.tau_raw.grad.max().item():.6f}")
    print(f"tau_raw mean val: {model.tau_raw.mean().item():.6f}")
    optimizer.step()
