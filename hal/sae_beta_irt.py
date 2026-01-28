import matplotlib
matplotlib.use("Agg")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import ast
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, mean_squared_error

try:
    from hypothesaes.quickstart import train_sae, interpret_sae
    HAS_SAE_LIB = True
except ImportError:
    HAS_SAE_LIB = False
    print("WARNING: 'hypothesaes' library not found.")

# Data split settings
TEST_SIZE = 0.1
RANDOM_SEED = 42

# SAE settings
USE_SAE = True
SAE_FEATURES = 48
SAE_K_SPARSITY = 4
SAE_EPOCHS = 100
SAE_LR = 5e-4
SAE_BATCH_SIZE = 512

# Model settings
K_MODEL = 30

# Tau sparsity settings (adjusted for SAE)
LAMBDA_TAU = 0.0005
TAU_INIT = 0.5
TAU_WARMUP = 300
RAMP_EPOCHS = 300
SNAPPING_THRESHOLD = 0.001
DEAD_ZONE_VALUE = -0.1
TAU_THRESHOLD = 0.01

# Training settings
EPOCHS = 1500
EVAL_EVERY = 100

# Learning rates
LR_THETA = 0.02
LR_GLOBAL = 0.005
WD_THETA = 1e-3
WD_W = 1e-5

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

resmat_dir = '../data-reeval-multi/colbench'
all_files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
emb_file = '../data-reeval-multi/hal/all_benchmarks_embeddings_4096_8B.pkl'
raw_embs_map = {}

if os.path.exists(emb_file):
    print("Loading embeddings...")
    emb_df = pd.read_pickle(emb_file)
    for _, r in emb_df.iterrows():
        raw_embs_map[str(r['benchmark.task_id'])] = r['embedding']
        if str(r['benchmark.task_id']).startswith('colbench_backend_programming'):
            suffix = str(r['benchmark.task_id']).split('.')[-1]
            raw_embs_map[f'colbench.{suffix}'] = r['embedding']

print(f"Total matrices: {len(all_files)}")

all_dfs = []
for f in all_files:
    all_dfs.append(pd.read_csv(os.path.join(resmat_dir, f), index_col=0))

global_shared_indices = set(all_dfs[0].index)
for df in all_dfs[1:]:
    global_shared_indices = global_shared_indices.intersection(set(df.index))
global_shared_indices = sorted(list(global_shared_indices))

oracle_dfs_filtered = [df.loc[global_shared_indices] for df in all_dfs]
oracle_stacked = np.array([df.values for df in oracle_dfs_filtered])
oracle_matrix = np.nanmean(oracle_stacked, axis=0)
oracle_df = pd.DataFrame(oracle_matrix, index=global_shared_indices, columns=oracle_dfs_filtered[0].columns)

print(f"Oracle matrix: {oracle_df.shape} (N={len(global_shared_indices)} users, J={len(oracle_df.columns)} items)")

N, J = oracle_df.shape

y_vals = oracle_df.values.astype(np.float32)
J_indices = np.arange(J)
np.random.seed(RANDOM_SEED)
np.random.shuffle(J_indices)

n_test = int(TEST_SIZE * J)
test_idx = J_indices[:n_test]
train_idx = J_indices[n_test:]

train_mask = np.zeros_like(y_vals, dtype=bool)
train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]
test_mask = np.zeros_like(y_vals, dtype=bool)
test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.0)).to(device)
train_mask = torch.from_numpy(train_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)

print(f"Train entries: {train_mask.sum().item()}, Test entries: {test_mask.sum().item()}")

print("Preparing embeddings and training SAE...")
task_ids_oracle = oracle_df.columns.tolist()
oracle_raw_embs = []
text_map = {}

if 'text_input' in emb_df.columns:
    for _, r in emb_df.iterrows():
        text_map[str(r['benchmark.task_id'])] = r['text_input']

aligned_texts = []
for task_id in task_ids_oracle:
    emb = raw_embs_map.get(str(task_id))
    if emb is None and task_id.startswith('colbench.'):
        number = task_id.split('.')[-1]
        emb = raw_embs_map.get(f'colbench_backend_programming.{number}')
    if emb is None:
        emb = np.zeros(4096)
    elif isinstance(emb, str):
        emb = ast.literal_eval(emb)
    oracle_raw_embs.append(emb)

    text = text_map.get(str(task_id))
    if text is None and task_id.startswith('colbench.'):
        number = task_id.split('.')[-1]
        text = text_map.get(f'colbench_backend_programming.{number}')
    aligned_texts.append(text if text else "")

embeddings_np = np.array(oracle_raw_embs, dtype=np.float32)
embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

print(f"Matched embeddings: {len(oracle_raw_embs)} / {len(task_ids_oracle)}")

print(f"\nTraining SAE (M={SAE_FEATURES}, K={SAE_K_SPARSITY})...")
sae = train_sae(
    embeddings=embeddings_np,
    M=SAE_FEATURES,
    K=SAE_K_SPARSITY,
    batch_size=SAE_BATCH_SIZE,
    n_epochs=SAE_EPOCHS,
    learning_rate=SAE_LR,
    checkpoint_dir='/tmp/_reproduce_sae_ckpt2'
)

print("Transforming embeddings to SAE activations...")
sae_activations_np = sae.get_activations(embeddings_np)

sae_activations_np = sae_activations_np / (np.linalg.norm(sae_activations_np, axis=1, keepdims=True) + 1e-8)

x_j_input = torch.tensor(sae_activations_np, dtype=torch.float32).to(device)

d_features = SAE_FEATURES
print(f"SAE feature dimension: {d_features}")
print(f"SAE activations shape: {x_j_input.shape}")

with torch.no_grad():
    avg_active = (np.abs(sae_activations_np) > 1e-6).sum(axis=1).mean()
    print(f"Avg active features per item: {avg_active:.2f} (target K={SAE_K_SPARSITY})")

# Feature interpretation
feature_descriptions_path = 'feature_descriptions_sae.pkl'

if os.path.exists(feature_descriptions_path):
    print(f"Loading existing feature descriptions from {feature_descriptions_path}")
    feature_descriptions_df = pd.read_pickle(feature_descriptions_path)
    print(f"Loaded {len(feature_descriptions_df)} feature interpretations")
    print("\nSample interpretations:")
    print(feature_descriptions_df[['neuron_idx', 'interpretation']].head(10))
else:
    try:
        if any(aligned_texts):
            print(f"Interpreting {SAE_FEATURES} SAE features...")
            feature_descriptions_df = interpret_sae(
                texts=aligned_texts,
                embeddings=embeddings_np,
                sae=sae,
                n_top_neurons=SAE_FEATURES,
                interpreter_model="gpt-4o"
            )
            print(f"Interpretations complete: {len(feature_descriptions_df)} features")
            print("\nSample interpretations:")
            print(feature_descriptions_df[['neuron_idx', 'interpretation']].head(10))
            if len(feature_descriptions_df) > 0:
                output_path = 'feature_descriptions_sae.pkl'
                feature_descriptions_df.to_pickle(output_path)
                print(f"Feature descriptions saved to: {output_path}")
                print(f"Saved {len(feature_descriptions_df)} feature interpretations")
            else:
                print("No feature descriptions to save (DataFrame is empty)")
    except Exception as e:
        print(f"Skipping interpretation (error or missing API key): {e}")
        feature_descriptions_df = pd.DataFrame()


class ReluARDModel(nn.Module):
    def __init__(self, N, J, K, d, x_j_emb, dropout=0.0):
        super().__init__()
        self.register_buffer('x_j', x_j_emb)
        self.dropout = dropout
        self.theta = nn.Parameter(torch.randn(N, K) * 0.01)
        self.theta_bias = nn.Parameter(torch.zeros(N))
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(K, d) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K) * TAU_INIT)
        self.difficulty_proj = nn.Linear(d, 1)

    def get_tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        W_norm = F.normalize(self.W, dim=1)
        base_loadings = self.x_j @ W_norm.T
        tau = self.get_tau()
        a_j = base_loadings * tau.unsqueeze(0)
        if self.training and self.dropout > 0:
            a_j = F.dropout(a_j, p=self.dropout)
        diff = self.difficulty_proj(self.x_j).squeeze()
        logits = self.theta @ a_j.T + diff.unsqueeze(0) + self.theta_bias.unsqueeze(1) + self.global_bias
        return torch.sigmoid(logits)

def compute_rmse(predictions, targets, mask):
    valid = mask.astype(bool) if isinstance(mask, np.ndarray) else mask
    return np.sqrt(mean_squared_error(targets[valid], predictions[valid]))

model = ReluARDModel(N, J, K_MODEL, d_features, x_j_input, dropout=0.5).to(device)
print(f"Model initialized: N={N}, J={J}, K={K_MODEL}, d={d_features}")

# Training loop
optimizer = optim.AdamW([
    {'params': [model.theta, model.theta_bias], 'lr': LR_THETA, 'weight_decay': WD_THETA},
    {'params': [model.W, model.tau_raw, model.global_bias], 'lr': LR_GLOBAL, 'weight_decay': WD_W},
    {'params': model.difficulty_proj.parameters(), 'lr': LR_GLOBAL}
])

print("Starting training...")
best_rmse = float('inf')

for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    probs = model()

    loss_fit = F.binary_cross_entropy(probs[train_mask], y_data[train_mask])

    if epoch < TAU_WARMUP:
        current_lambda = 0.0
    elif epoch < TAU_WARMUP + RAMP_EPOCHS:
        progress = (epoch - TAU_WARMUP) / RAMP_EPOCHS
        current_lambda = LAMBDA_TAU * progress
    else:
        current_lambda = LAMBDA_TAU

    tau = model.get_tau()
    loss_sparsity = current_lambda * torch.sum(tau)
    (loss_fit + loss_sparsity).backward()
    optimizer.step()

    if epoch > TAU_WARMUP + 50 and epoch % 10 == 0:
        with torch.no_grad():
            active_mask = model.get_tau() > SNAPPING_THRESHOLD
            for k in range(K_MODEL):
                if not active_mask[k]:
                    model.tau_raw[k] = DEAD_ZONE_VALUE

    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            p_eval = model()
            curr_rmse = compute_rmse(p_eval.cpu().numpy(), y_data.cpu().numpy(), test_mask.cpu().numpy())
            best_rmse = min(best_rmse, curr_rmse)

            active_dims = (model.get_tau() > TAU_THRESHOLD).sum().item()
            print(f"Ep {epoch} | Loss: {loss_fit.item():.4f} | RMSE: {curr_rmse:.4f} | Active dims: {active_dims}")

print(f"\nTraining complete. Best RMSE: {best_rmse:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    probs = model()

    y_test_np = y_data.cpu().numpy()
    p_test_np = probs.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()

    rmse_test = compute_rmse(p_test_np, y_test_np, test_mask_np)
    rmse_train = compute_rmse(p_test_np, y_test_np, train_mask_np)

    y_test_flat = y_test_np[test_mask_np]
    p_test_flat = p_test_np[test_mask_np]
    y_train_flat = y_test_np[train_mask_np]
    p_train_flat = p_test_np[train_mask_np]

    y_test_binary = (y_test_flat > 0.5).astype(int)
    y_train_binary = (y_train_flat > 0.5).astype(int)

    test_auc = roc_auc_score(y_test_binary, p_test_flat) if len(np.unique(y_test_binary)) > 1 else 0.5
    train_auc = roc_auc_score(y_train_binary, p_train_flat) if len(np.unique(y_train_binary)) > 1 else 0.5

    test_acc = np.mean((p_test_flat > 0.5) == y_test_binary)
    train_acc = np.mean((p_train_flat > 0.5) == y_train_binary)

    print("="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"\nTest RMSE: {rmse_test:.4f} | Train RMSE: {rmse_train:.4f}")
    print(f"Test AUC:  {test_auc:.4f} | Train AUC:  {train_auc:.4f}")
    print(f"Test Acc:  {test_acc:.4f} | Train Acc:  {train_acc:.4f}")

    tau_values = model.get_tau().detach().cpu().numpy()
    active_indices = np.where(tau_values > TAU_THRESHOLD)[0]
    print(f"\nDiscovered {len(active_indices)} active latent dimensions: {active_indices[:20]}...")
    print(f"Tau values (top 10): {np.sort(tau_values)[::-1][:10]}")

print("\nINTERPRETING LATENT FACTORS")
print(f"Analyzing {len(active_indices)} active dimensions")

W_matrix = model.W.detach().cpu().numpy()
TOP = 3

for k in active_indices[:10]:
    print(f"\n--- Latent Factor (Skill) #{k} (tau={tau_values[k]:.3f}) ---")

    weights = W_matrix[k]
    top_feature_indices = np.argsort(np.abs(weights))[-TOP:][::-1]

    print("  Driven by SAE Features:")
    for f_idx in top_feature_indices:
        weight_val = weights[f_idx]

        try:
            if 'feature_descriptions_df' in dir() and len(feature_descriptions_df) > 0:
                desc = feature_descriptions_df.loc[feature_descriptions_df['neuron_idx'] == f_idx, 'interpretation'].values[0]
                if desc is None or str(desc) == 'None':
                    desc = f"SAE Feature {f_idx} (interpretation unavailable)"
            else:
                desc = f"SAE Feature {f_idx} (run interpretation cell to get descriptions)"
        except:
            desc = f"SAE Feature {f_idx} (no description)"

        print(f"    Neuron {f_idx} (w={weight_val:.3f}): {desc}")
