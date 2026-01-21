# Auto-converted from train_hal.ipynb

# %%
# Ensure we're in the correct directory for imports
import os
import sys

# Get the directory containing this notebook
# Try multiple methods to find the notebook directory
if '__file__' in globals():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
else:
    notebook_dir = os.getcwd()

# Ensure we're in the hal directory
# Handle both local and Azure ML compute paths
if not notebook_dir.endswith('hal'):
    # Check if we're already in a subdirectory
    if 'reeval-multi/hal' in notebook_dir:
        # Extract the hal directory path
        parts = notebook_dir.split('reeval-multi')
        if len(parts) >= 2:
            notebook_dir = parts[0] + 'reeval-multi/hal'
    elif 'hal' not in notebook_dir:
        # Try to construct the path
        possible_paths = [
            '/home/azureuser/cloudfiles/code/reeval-multi/hal',
            '/mnt/batch/tasks/shared/LS_root/mounts/clusters/nvidiat4x1-16g/code/reeval-multi/hal',
            os.path.join(notebook_dir, 'hal')
        ]
        for path in possible_paths:
            if os.path.exists(path):
                notebook_dir = path
                break

# Change to notebook directory FIRST
if os.path.exists(notebook_dir):
    os.chdir(notebook_dir)
    # Add to Python path if not already there - use insert(0) to prioritize
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    print(f"✓ Working directory: {os.getcwd()}")
else:
    print(f"⚠ Warning: Could not find hal directory. Current dir: {os.getcwd()}")

# CRITICAL: Load .env BEFORE importing azure_hypothesaes_patch
from dotenv import load_dotenv
env_file = os.path.join(os.getcwd(), '.env')
env_loaded = load_dotenv(env_file, override=True)
print(f"✓ Loading env from: {env_file}")
print(f"✓ Environment variables loaded: {env_loaded}")

# Verify Azure credentials are set
azure_key = os.environ.get('AZURE_OPENAI_KEY')
azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
azure_deployment = os.environ.get('AZURE_OPENAI_DEPLOYMENT')

print(f"\n🔑 Azure Configuration Check:")
if azure_key and '<your-api-key>' not in azure_key and azure_key != '':
    print(f"  ✓ AZURE_OPENAI_KEY: Set (length: {len(azure_key)})")
else:
    print(f"  ✗ AZURE_OPENAI_KEY: NOT SET or placeholder")
    
if azure_endpoint:
    print(f"  ✓ AZURE_OPENAI_ENDPOINT: {azure_endpoint}")
else:
    print(f"  ✗ AZURE_OPENAI_ENDPOINT: NOT SET")
    
if azure_deployment:
    print(f"  ✓ AZURE_OPENAI_DEPLOYMENT: {azure_deployment}")
else:
    print(f"  ✗ AZURE_OPENAI_DEPLOYMENT: NOT SET")

# Now import the Azure patch
patch_file = os.path.join(os.getcwd(), 'azure_hypothesaes_patch.py')
if os.path.exists(patch_file):
    print(f"\n✓ Found azure_hypothesaes_patch.py")
    
    try:
        # Force reload if already imported
        if 'azure_hypothesaes_patch' in sys.modules:
            import importlib
            import azure_hypothesaes_patch
            importlib.reload(azure_hypothesaes_patch)
            print(f"✓ Reloaded azure_hypothesaes_patch with updated code")
        else:
            import azure_hypothesaes_patch
            print(f"✓ Successfully imported and patched hypothesaes for Azure OpenAI")
    except Exception as e:
        print(f"✗ ERROR importing azure_hypothesaes_patch: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ ERROR: azure_hypothesaes_patch.py not found at {patch_file}")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import ast
from sklearn.metrics import roc_auc_score
# IMPORTANT: Import Azure patch BEFORE hypothesaes
# Check if already imported from previous cell
if 'azure_hypothesaes_patch' not in sys.modules:
    import azure_hypothesaes_patch
    print("✓ Loaded Azure OpenAI patch")
else:
    print("✓ Azure OpenAI patch already loaded")

from hypothesaes.quickstart import train_sae, interpret_sae

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.manual_seed(42)
np.random.seed(42)

# Device Setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
y_df = pd.read_csv('data-hal/result_matrix_newrun.csv', index_col=0)
emb_df = pd.read_pickle('result/all_benchmarks_embeddings_4096_8B.pkl')

z_names = ['environmentalbarrier_newrun', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']
z_df_list = [pd.read_csv(f'data-hal/rubrics_matrix_{z_name}.csv', index_col=0) for z_name in z_names]
# benchmarks = ["assistantbench", "taubench_airline", "corebench", "scienceagentbench"]
benchmarks = None

# Filter Benchmarks
if benchmarks:
    print(f"Filtering by: {benchmarks}")
    cols = [c for c in y_df.columns if any(b in str(c) for b in benchmarks)]
    print(f"Selected {len(cols)} benchmarks after filtering.")
    y_df = y_df[cols]
    z_df_list = [df[cols] for df in z_df_list]

# %%

# # Filter Rows/Cols
# y_df = y_df[y_df.notna().any(axis=1)]
# for i in range(len(z_df_list)): z_df_list[i] = z_df_list[i].loc[y_df.index]

# Filter out columns with all NaN or all 0 values

valid_cols = []
for c in y_df.columns:
    valid_cols.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
y_df = y_df.loc[:, valid_cols]
for i in range(len(z_df_list)): z_df_list[i] = z_df_list[i].loc[:, valid_cols]

# ==========================================
# 2. ITEM-WISE SPLIT (COLD START)
# ==========================================
print("\nCreating ITEM-WISE Split (Cold Start)...")
y_vals = y_df.values.astype(np.float32)
N, J = y_vals.shape
J_indices = np.arange(J)
np.random.shuffle(J_indices)

n_test = int(0.1 * J)
test_idx = J_indices[:n_test]
train_idx = J_indices[n_test:]

train_mask = np.zeros_like(y_vals, dtype=bool)
train_mask[:, train_idx] = ~np.isnan(y_vals)[:, train_idx]

test_mask = np.zeros_like(y_vals, dtype=bool)
test_mask[:, test_idx] = ~np.isnan(y_vals)[:, test_idx]

y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.0)).to(device)
train_mask = torch.from_numpy(train_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)

# Z Data
z_data = torch.stack([torch.from_numpy(np.nan_to_num(df.values, nan=0.0)).float() for df in z_df_list], dim=2).to(device)
z_mask = torch.stack([torch.from_numpy((~np.isnan(df.values)).astype(bool)) for df in z_df_list], dim=2).to(device)

# Mask all z entries for testing
# M = len(z_names)
# z_mask = torch.zeros((N, J, M), dtype=torch.bool).to(device)  # All masked (no data)

print(f"Y Matrix Shape: {y_df.shape}")
print(f"Z Tensor Shape: {z_data.shape}")

# Filter out environmentalbarrier_newrun == 1 (treat as missing)
print("\nFiltering environmentalbarrier_newrun == 1...")
env_barrier_idx = z_names.index('environmentalbarrier_newrun')
env_barrier_mask = z_data[:, :, env_barrier_idx] == 1
print(f"Masking {env_barrier_mask.sum().item()} entries where environmentalbarrier_newrun == 1")

# Apply mask to training/test masks and z_mask
train_mask = train_mask & ~env_barrier_mask
test_mask = test_mask & ~env_barrier_mask
z_mask = z_mask & ~env_barrier_mask.unsqueeze(2)

# %%
# ==========================================
# 3. PREPARE EMBEDDINGS & TRAIN SAE
# ==========================================
print("\nPreparing Embeddings...")
emb_map = {str(r['benchmark.task_id']): r['embedding'] for _, r in emb_df.iterrows()}
# [ADDED] Extract texts for interpretation
texts_list = [] # We need to rebuild texts aligned with the columns of y_df
if 'text_input' in emb_df.columns:
    text_map = {str(r['benchmark.task_id']): r['text_input'] for _, r in emb_df.iterrows()}
else:
    text_map = {}

raw_embs = []
aligned_texts = [] 
for c in y_df.columns:
    e = emb_map.get(str(c), np.zeros(512))
    if isinstance(e, str): e = ast.literal_eval(e)
    raw_embs.append(e)
    aligned_texts.append(text_map.get(str(c), ""))

x_j_dense = torch.tensor(np.array(raw_embs), dtype=torch.float32)
x_j_dense = F.normalize(x_j_dense, p=2, dim=1).to(device)

print(f"Matched embeddings: {len(raw_embs)} / {y_df.shape[1]} ({len(raw_embs)/y_df.shape[1]:.2%})")

# --- [ADDED] SAE Training & Transformation Start ---
print("\nTraining/Loading SAE...")
embeddings_np = x_j_dense.cpu().numpy()

# Train the SAE
sae = train_sae(
    embeddings=embeddings_np,
    M=64,             # Total features
    K=4,               # Active features per item
    batch_size=512,
    n_epochs=100,
    learning_rate=5e-4,
    checkpoint_dir='checkpoints/hal_sae'
)

# Interpret features using Azure OpenAI
# Azure patch handles endpoint routing and per-neuron retries.
try:
    if any(aligned_texts):
        print("Generating SAE Interpretations with Azure OpenAI...")
        
        feature_descriptions_df = interpret_sae(
            texts=aligned_texts,
            embeddings=embeddings_np,
            sae=sae,
            n_top_neurons=64,
            interpreter_model="gpt-5.2"  # This will use your Azure deployment
        )
        
        print("\nTop 64 Interpreted Neurons:")
        print(feature_descriptions_df[['neuron_idx', 'interpretation']])
except Exception as e:
    print(f"Skipping interpretation (error or missing API key): {e}")
    import traceback
    traceback.print_exc()

# Transform dense embeddings to sparse activations
print("Transforming embeddings to SAE activations...")
sae_activations_np = sae.get_activations(embeddings_np)
x_j_input = torch.tensor(sae_activations_np, dtype=torch.float32).to(device)

# Update dimension for the Bayesian Model
d_features = sae.m_total_neurons 
print(f"New Feature Dimension (SAE): {d_features}")
# --- [ADDED] SAE Logic End ---

M = z_data.shape[2]

# %%
# ==========================================
# 4. ROBUST LINEAR MODEL (Paper Aligned)
# ==========================================
class LinearRobustARD(nn.Module):
    def __init__(self, N, J, M, K_model, d_features, x_j_input):
        super().__init__()
        self.N, self.J, self.M, self.K = N, J, M, K_model
        self.register_buffer('x_j', x_j_input)

        # [FEATURE 2] Latent Factors
        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)  # User Ability
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.1) # Feature Projection
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)    # Sparsity Scale
        
        # [FEATURE 3] Subskills
        self.u_logits = nn.Parameter(torch.ones(M, K_model) * 2.0)
        self.delta_m = nn.Parameter(torch.zeros(1, M)) # Global Subskill Bias (Parameter Efficient)

        # [FEATURE 4] Linear Amortized Difficulty 
        self.difficulty_proj = nn.Linear(d_features, 1)

    @property
    def tau(self):
        # Paper Eq (6): ReLU ensures exact zero sparsity
        return F.relu(self.tau_raw)

    def get_gates(self, t):
        # Paper Eq (5): Differentiable relaxation
        return torch.sigmoid(self.u_logits / t)

    def forward(self, temp=1.0):
        # 1. Corrupt Input
        x_j = self.x_j
        
        # 2. Linear Difficulty Projection
        # (J, d) @ (d, 1) -> (1, J)
        pred_delta = self.difficulty_proj(x_j).squeeze().unsqueeze(0)
        
        # 3. Linear Loading Projection (Normalized)
        # Paper Eq (7): Normalize W to resolve scale indeterminacy
        W_norm = F.normalize(self.W, dim=1)
        # Paper Eq (7): a_j = tau * (W @ x_j)
        a_j = (x_j @ W_norm.T) * self.tau.unsqueeze(0)

        # 4. Overall Prediction
        # Paper Eq (1): p = sigma(theta @ a_j + delta)
        logits_y = self.theta @ a_j.T + pred_delta
        
        # 5. Subskill Prediction (Gated)
        g_m = self.get_gates(temp)
        logits_z = []
        for m in range(self.M):
            # z depends on gated loading
            lz = self.theta @ (a_j * g_m[m].unsqueeze(0)).T + self.delta_m[:, m].unsqueeze(1)
            logits_z.append(lz.unsqueeze(2))
            
        return logits_y, torch.cat(logits_z, dim=2)

# %%
# ==========================================
# 5. OPTIMIZATION (CORRECTED)
# ==========================================
K_MODEL = 100
lambda_tau = 30.5
lr_latent = 0.01
lr_proj = 0.005
lr_tau = 0.01
wd_latent = 1e-4
wd_proj = 1e-2
reg_sparse_gates = 0.1
reg_beta_gates = 0.1
reg_theta = 0.5 

model = LinearRobustARD(N, J, M, K_MODEL, d_features, x_j_input).to(device)

# [FAITHFUL FIX 1] Split Optimizers for Alternating Minimization
# Group A: Local Parameters (Theta) - Step 1 in Alg 1
opt_local = optim.Adam([
    {'params': [model.theta], 'lr': lr_latent, 'weight_decay': wd_latent}
], lr=lr_latent)

# Group B: Global Parameters (W, Tau, Gates) - Step 2 in Alg 1
opt_global = optim.Adam([
    # Tau: L1 applied manually (cite: 141)
    {'params': model.tau_raw, 'lr': lr_tau, 'weight_decay': 0.0},
    
    # W: L2 Weight Decay (cite: 146 uses ||W||^2_F)
    {'params': [model.W], 'lr': lr_proj, 'weight_decay': wd_proj}, 

    # Difficulty & Gates: L2 Decay
    {'params': list(model.difficulty_proj.parameters()) + [model.u_logits, model.delta_m], 
     'lr': lr_proj, 'weight_decay': wd_proj}
], lr=lr_proj)

hyperparams = {
    'lambda_tau': lambda_tau, 
}

print("\nStarting Faithful Alternating Optimization...")
n_epochs = 2001
early_stop_patience = 20  # Number of epochs to wait for improvement
best_loss = float('inf')
epochs_no_improve = 0

for e in range(n_epochs):
    model.train()
    current_temp = max(0.1, 1.0 - (e / 1000.0))

    # --- STEP 1: Local Inference ---
    opt_local.zero_grad()
    logits_y, logits_z = model(temp=current_temp)
    lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    lik_z = (F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none') * (z_mask & train_mask.unsqueeze(2))).sum()
    loss_theta = reg_theta * torch.sum(model.theta**2)
    loss_local = lik_y + lik_z + loss_theta
    loss_local.backward()
    opt_local.step()

    # --- STEP 2: Global Update ---
    opt_global.zero_grad()
    logits_y, logits_z = model(temp=current_temp)
    lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    lik_z = (F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none') * (z_mask & train_mask.unsqueeze(2))).sum()
    gates = torch.sigmoid(model.u_logits / current_temp)
    loss_sparse_gates = reg_sparse_gates * torch.sum(gates)
    loss_beta_gates = reg_beta_gates * torch.sum(gates * (1.0 - gates))
    reg_tau_loss = hyperparams['lambda_tau'] * torch.norm(model.tau, 1)
    loss_global = lik_y + lik_z + reg_tau_loss + loss_sparse_gates + loss_beta_gates
    loss_global.backward()
    opt_global.step()

    # --- STEP 3: Zero-Snapping ---
    with torch.no_grad():
        model.tau_raw[model.tau < 0.01] = -0.1

    # --- EARLY STOPPING ---
    if loss_global.item() < best_loss:
        best_loss = loss_global.item()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping at epoch {e} | Best Loss: {best_loss:.2e}")
        break

    if e % 100 == 0:
        active_dims = (model.tau > 0.01).sum().item()
        print(f"Ep {e} | T: {current_temp:.2f} | Loss: {loss_global.item():.2e} | Active Dims: {active_dims}")

# %%
# ==========================================
# 6. EVALUATION
# ==========================================
print("\n=== EVALUATING (LINEAR MODEL) ===")
model.eval()
with torch.no_grad():
    logits_y, _ = model()
    probs = torch.sigmoid(logits_y)
    
    y_test = torch.masked_select(y_data, test_mask).cpu().numpy()
    p_test = torch.masked_select(probs, test_mask).cpu().numpy()
    
    y_train = torch.masked_select(y_data, train_mask).cpu().numpy()
    p_train = torch.masked_select(probs, train_mask).cpu().numpy()
    
    test_auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.0
    test_acc = np.mean((p_test > 0.5) == y_test)
    train_auc = roc_auc_score(y_train, p_train) if len(np.unique(y_train)) > 1 else 0.0
    train_acc = np.mean((p_train > 0.5) == y_train)
    
    print(f"\n[Test Set] N={len(y_test)}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")

    print(f"\n[Train Set] N={len(y_train)}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    
    # Print sparsity only for active (non-zero tau) dims in W after training
    W_all = model.W.detach().cpu().numpy()
    tau_active = model.tau.detach().cpu().numpy() > 0.01
    zero_counts = np.sum(np.abs(W_all) < 1e-3, axis=1)
    print(f"\nSparsity of ACTIVE W rows ({active_dims}):")
    for i, count in enumerate(zero_counts):
        if tau_active[i]:
            print(f"W[{i}] zeros: {count}/{W_all.shape[1]}")

# %%
# ==========================================
# 7. INTERPRET DISCOVERED LATENT FACTORS
# ==========================================
print("\n=== INTERPRETING LATENT FACTORS ===")

# 1. Identify Active Dimensions (where tau > 0.01)
tau_values = model.tau.detach().cpu().numpy()
active_indices = np.where(tau_values > 0.01)[0]
print(f"Analyzing {len(active_indices)} active dimensions: {active_indices}")

# 2. Get the W matrix (The mapping from Skills -> SAE Features)
W_matrix = model.W.detach().cpu().numpy()

# 3. For each active dimension, find the SAE features with the highest weights
for k in active_indices:
    print(f"\n--- Latent Factor (Skill) #{k} ---")
    
    # Get weights for this dimension across all 64 SAE features
    weights = W_matrix[k]
    
    # Get indices of the top 5 positive weights (Positive contributors to difficulty/skill requirement)
    # Note: Depending on sign convention in theta*W, positive might mean "Requires this skill"
    top_feature_indices = np.argsort(weights)[-5:][::-1]
    
    print("  Driven by SAE Features:")
    for f_idx in top_feature_indices:
        weight_val = weights[f_idx]
        
        # Look up the description from your interpret_sae dataframe
        # Assuming feature_descriptions_df is available from your earlier step
        try:
            desc = feature_descriptions_df.loc[feature_descriptions_df['neuron_idx'] == f_idx, 'interpretation'].values[0]
            # Truncate for display
            desc = (desc[:75] + '..') if len(desc) > 75 else desc
        except:
            desc = "No description available"
            
        print(f"    Neuron {f_idx} (w={weight_val:.3f}): {desc}")
