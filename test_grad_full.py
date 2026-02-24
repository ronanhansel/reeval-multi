import sys, os, pandas as pd, torch, numpy as np
from model.amortized_irt import load_data, prepare_experiment_data, AmortizedIRTModel

all_dfs, global_shared_indices, raw_embs_map, actual_emb_type = load_data('sae', 48)
data = prepare_experiment_data(all_dfs, global_shared_indices, raw_embs_map)

n_files = 54
current_dfs = [all_dfs[i].reindex(index=global_shared_indices) for i in range(n_files)]
all_columns = sorted(list(set().union(*[df.columns for df in current_dfs])))
current_dfs = [df.reindex(columns=all_columns) for df in current_dfs]
current_stacked = np.array([df.values for df in current_dfs], dtype=float)
train_target_matrix = np.nanmean(current_stacked, axis=0)
train_target_df = pd.DataFrame(train_target_matrix, index=global_shared_indices, columns=all_columns)

train_values = np.nan_to_num(train_target_df.values, nan=0.5)
y_train = torch.from_numpy(train_values.astype(np.float32)).to('cuda')

train_mask_current = np.zeros_like(train_target_df.values, dtype=bool)
train_mask_current[:, data['train_idx']] = ~np.isnan(train_target_df.values)[:, data['train_idx']]
train_mask_current_t = torch.from_numpy(train_mask_current).to('cuda')

model = AmortizedIRTModel(len(global_shared_indices), data['J'], 64, 48, data['x_j'], dropout=0.5).to('cuda')

import torch.optim as optim
optimizer = optim.AdamW([
    {'params': [model.theta, model.theta_bias], 'lr': 0.01, 'weight_decay': 0.0},
    {'params': [model.W, model.global_bias], 'lr': 0.002, 'weight_decay': 0.0},
    {'params': model.difficulty_proj.parameters(), 'lr': 0.002}
])
optimizer_tau = optim.SGD([model.tau_raw], lr=0.5)

beta_phi = 10.0
eps = 1e-6
lambda_tau = 0.0001
model.tau_raw.data.fill_(0.5)

print("Starting training loop...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    optimizer_tau.zero_grad()
    
    probs = model()
    p = probs[train_mask_current_t].clamp(eps, 1 - eps)
    y = y_train[train_mask_current_t].clamp(eps, 1 - eps)
    dist = torch.distributions.Beta(p * beta_phi, (1 - p) * beta_phi)
    
    loss_fit = -dist.log_prob(y).mean()
    loss_sparsity = lambda_tau * torch.sum(model.get_tau())
    
    total_loss = loss_fit + loss_sparsity
    total_loss.backward()
    
    # Capture grads before optimizer step
    fit_grad = None
    if model.tau_raw.grad is not None:
         # To isolate fit_grad, we'd need a separate backward, but let's just see total_grad
         total_grad = model.tau_raw.grad.clone()
    
    optimizer.step()
    optimizer_tau.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}:")
        print(f"  tau_raw mean={model.tau_raw.mean().item():.5f}, dims={(model.get_tau()>0.001).sum().item()}")
        print(f"  total_grad mean={total_grad.mean().item():.8f}")
        print(f"  total_grad max={total_grad.max().item():.8f}, min={total_grad.min().item():.8f}")
        
