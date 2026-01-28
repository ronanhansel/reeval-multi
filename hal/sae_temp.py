import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import ast
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

try:
    from hypothesaes.quickstart import train_sae, interpret_sae
    HAS_SAE_LIB = True
except ImportError:
    HAS_SAE_LIB = False
    print("WARNING: 'hypothesaes' library not found.")

TEST_SIZE = 0.1
RANDOM_SEED = 42
USE_SAE = True
SAE_FEATURES = 64
SAE_K_SPARSITY = 4
SAE_EPOCHS = 100
SAE_LR = 5e-4
SAE_BATCH_SIZE = 512

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
print("\nINTERPRETING LATENT FACTORS")
print(f"Analyzing {len(active_indices)} active dimensions")

W_matrix = model.W.detach().cpu().numpy()
TOP = 3

for k in active_indices:
    print(f"\n--- Latent Factor (Skill) #{k} (tau={tau_values[k]:.3f}) ---")
    
    weights = W_matrix[k]
    top_feature_indices = np.argsort(weights)[-TOP:][::-1]
    
    print("  Driven by SAE Features:")
    for f_idx in top_feature_indices:
        weight_val = weights[f_idx]
        
        try:
            if len(feature_descriptions_df) > 0:
                desc = feature_descriptions_df.loc[feature_descriptions_df['neuron_idx'] == f_idx, 'interpretation'].values[0]
                if desc is None or str(desc) == 'None':
                    desc = f"SAE Feature {f_idx} (interpretation unavailable)"
            else:
                desc = f"SAE Feature {f_idx} (interpretations not generated)"
        except:
            desc = f"SAE Feature {f_idx} (no description)"
        
        print(f"    Neuron {f_idx} (w={weight_val:.3f}): {desc}")
model.eval()
with torch.no_grad():
    logits_y = model()
    probs = torch.sigmoid(logits_y)
    
    y_test = torch.masked_select(y_data, test_mask).cpu().numpy()
    p_test = torch.masked_select(probs, test_mask).cpu().numpy()
    y_train = torch.masked_select(y_data, train_mask).cpu().numpy()
    p_train = torch.masked_select(probs, train_mask).cpu().numpy()
    
    y_test_binary = (y_test > 0.5).astype(int)
    y_train_binary = (y_train > 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test_binary, p_test) if len(np.unique(y_test_binary)) > 1 else 0.0
    test_acc = np.mean((p_test > 0.5) == y_test_binary)
    train_auc = roc_auc_score(y_train_binary, p_train) if len(np.unique(y_train_binary)) > 1 else 0.0
    train_acc = np.mean((p_train > 0.5) == y_train_binary)
    
    print(f"\nTest AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f}")
    
    tau_values = model.tau.detach().cpu().numpy()
    active_indices = np.where(tau_values > 0.01)[0]
    print(f"\nDiscovered {len(active_indices)} active latent dimensions: {active_indices}")
lambda_tau = 25
lr_latent = 0.01
lr_proj = 0.005
lr_tau = 0.01
wd_latent = 1e-4
wd_proj = 1e-2
reg_theta = 0.5

opt_local = optim.Adam([{'params': [model.theta], 'lr': lr_latent, 'weight_decay': wd_latent}], lr=lr_latent)
opt_global = optim.Adam([
    {'params': model.tau_raw, 'lr': lr_tau, 'weight_decay': 0.0},
    {'params': [model.W], 'lr': lr_proj, 'weight_decay': wd_proj},
    {'params': list(model.difficulty_proj.parameters()), 'lr': lr_proj, 'weight_decay': wd_proj}
], lr=lr_proj)

print("Starting alternating optimization...")
n_epochs = 2001
early_stop_patience = 20
best_loss = float('inf')
epochs_no_improve = 0

for e in range(n_epochs):
    model.train()
    
    opt_local.zero_grad()
    logits_y = model()
    lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    loss_theta = reg_theta * torch.sum(model.theta**2)
    loss_local = lik_y + loss_theta
    loss_local.backward()
    opt_local.step()
    
    opt_global.zero_grad()
    logits_y = model()
    lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    reg_tau_loss = lambda_tau * torch.norm(model.tau, 1)
    loss_global = lik_y + reg_tau_loss
    loss_global.backward()
    opt_global.step()
    
    with torch.no_grad():
        model.tau_raw[model.tau < 0.01] = -0.1
    
    if loss_global.item() < best_loss:
        best_loss = loss_global.item()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping at epoch {e}")
        break
    
    if e % 100 == 0:
        active_dims = (model.tau > 0.01).sum().item()
        print(f"Ep {e} | Loss: {loss_global.item():.2e} | Active dims: {active_dims}")
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
class LinearRobustARD(nn.Module):
    def __init__(self, N, J, K_model, d_features, x_j_input):
        super().__init__()
        self.N, self.J, self.K = N, J, K_model
        self.register_buffer('x_j', x_j_input)
        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.1)
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)
        self.difficulty_proj = nn.Linear(d_features, 1)

    @property
    def tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        x_j = self.x_j
        pred_delta = self.difficulty_proj(x_j).squeeze().unsqueeze(0)
        W_norm = F.normalize(self.W, dim=1)
        a_j = (x_j @ W_norm.T) * self.tau.unsqueeze(0)
        logits_y = self.theta @ a_j.T + pred_delta
        return logits_y

K_MODEL = 100
model = LinearRobustARD(N, J, K_MODEL, d_features, x_j_input).to(device)
print(f"Model initialized: N={N}, J={J}, K={K_MODEL}, d={d_features}")
try:
    if any(aligned_texts):
        print(f"Interpreting {SAE_FEATURES} SAE features...")
        feature_descriptions_df = interpret_sae(
            texts=aligned_texts,
            embeddings=embeddings_np,
            sae=sae,
            n_top_neurons=SAE_FEATURES,
            interpreter_model="gpt-4"
        )
        print(f"Interpretations complete: {len(feature_descriptions_df)} features")
        print("\nSample interpretations:")
        print(feature_descriptions_df[['neuron_idx', 'interpretation']].head(10))
except Exception as e:
    print(f"Skipping interpretation (error or missing API key): {e}")
    feature_descriptions_df = pd.DataFrame()
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
    checkpoint_dir='checkpoints/hal_sae_temp'
)

print("Transforming embeddings to SAE activations...")
sae_activations_np = sae.get_activations(embeddings_np)
x_j_input = torch.tensor(sae_activations_np, dtype=torch.float32).to(device)

d_features = sae.m_total_neurons
print(f"SAE feature dimension: {d_features}")
print(f"SAE activations shape: {x_j_input.shape}")

with torch.no_grad():
    avg_active = (np.abs(sae_activations_np) > 1e-6).sum(axis=1).mean()
    print(f"Avg active features per item: {avg_active:.2f} (target K={SAE_K_SPARSITY})")
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
resmat_dir = 'resmats'
all_files = sorted([f for f in os.listdir(resmat_dir) if f.startswith('resmat')])
emb_file = 'result/all_benchmarks_embeddings_4096_8B.pkl'
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