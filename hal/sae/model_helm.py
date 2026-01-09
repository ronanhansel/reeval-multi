import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import warnings
import argparse
import ast
from sklearn.metrics import roc_auc_score

# [ADDED] Imports for HypotheSAEs
from hypothesaes.quickstart import train_sae, interpret_sae

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.manual_seed(42)
np.random.seed(42)

# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 0. SETUP & ARGS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, nargs='+', default=None)
# [NOTE] Lowered lambda_tau. 500.0 is very aggressive and might force all features to zero.
parser.add_argument('--lambda_tau', type=float, default=1500.0) 
parser.add_argument('--K_MODEL', type=int, default=100)
parser.add_argument('--reg_sparse_gates', type=float, default=0.1)
parser.add_argument('--reg_beta_gates', type=float, default=0.1)
parser.add_argument('--reg_theta', type=float, default=0.5)
parser.add_argument('--lr_tau', type=float, default=0.01)
parser.add_argument('--lr_proj', type=float, default=0.005)
parser.add_argument('--lr_latent', type=float, default=0.01)
parser.add_argument('--wd_proj', type=float, default=1e-2)
parser.add_argument('--wd_latent', type=float, default=1e-4)
args = parser.parse_args()

# ==========================================
# 1. LOAD DATA & ALIGNMENT (CRITICAL FIX)
# ==========================================
print("Loading data...")
y_df = pd.read_pickle('../../data-reeval-multi/resmat.pkl')
emb_df = pd.read_pickle('../../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl')

z_names = ['environmentalbarrier', 'instructionfollowing', 'selfcorrection', 'tooluse', 'verification']
# Rubrics data not available for HELM - will create NaN placeholders
z_df_list = None

# 1. Filter Benchmarks in y_df first
if args.benchmark:
    print(f"Filtering by: {args.benchmark}")
    benchmark_mask = np.array([any(b in str(c) for b in args.benchmark) for c in y_df.columns])
    y_df = y_df.iloc[:, benchmark_mask]

# 2. Filter Rows/Cols in y_df
y_df = y_df[y_df.notna().any(axis=1)]
valid_cols_list = []
for c in y_df.columns:
    valid_cols_list.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
y_df = y_df.iloc[:, valid_cols_list]
print(f"Target Matrix Shape: {y_df.shape}")

# 3. ALIGNMENT LOGIC (The Fix)
print("\nAligning Embeddings to Question Text...")

# Create a lookup dictionary from emb_df
# Assuming emb_df has 'question' and 'embedding' columns
if 'question' not in emb_df.columns:
    # Fallback if 'question' column name differs, e.g., 'input.text'
    print("Warning: 'question' column not found in emb_df. printing cols:", emb_df.columns)
    # Try to find the text column
    text_col = [c for c in emb_df.columns if 'text' in str(c) or 'question' in str(c)][0]
    emb_df = emb_df.rename(columns={text_col: 'question'})

# Normalize embeddings in dictionary for fast lookup
emb_map = {}
text_map = {}
for _, row in emb_df.iterrows():
    q_text = row['question']
    emb = row['embedding']
    if isinstance(emb, str): emb = ast.literal_eval(emb)
    emb_map[q_text] = emb
    text_map[q_text] = q_text

# Get questions from y_df columns
# y_df columns are often MultiIndex. We try to grab the 'input.text' level if it exists.
try:
    if isinstance(y_df.columns, pd.MultiIndex):
        # Check if 'input.text' is a level name
        if 'input.text' in y_df.columns.names:
            questions = y_df.columns.get_level_values('input.text').tolist()
        else:
            # Fallback: assume the last level is unique ID or text
            questions = y_df.columns.get_level_values(-1).tolist()
    else:
        questions = y_df.columns.tolist()
except Exception as e:
    print(f"Error extracting column names: {e}")
    questions = [str(c) for c in y_df.columns]

# Build the aligned lists
aligned_raw_embs = []
aligned_texts = []
valid_indices_mask = []

for i, q in enumerate(questions):
    if q in emb_map:
        aligned_raw_embs.append(emb_map[q])
        aligned_texts.append(text_map[q])
        valid_indices_mask.append(True)
    else:
        # If mapping fails, we cannot use this item.
        # Mark it False to filter y_df later
        valid_indices_mask.append(False)

# 4. Final Filtering based on Alignment
valid_indices_mask = np.array(valid_indices_mask)
print(f"Found embeddings for {valid_indices_mask.sum()} / {len(questions)} items")

y_df = y_df.iloc[:, valid_indices_mask]
x_j_dense = torch.tensor(np.array(aligned_raw_embs), dtype=torch.float32)
x_j_dense = F.normalize(x_j_dense, p=2, dim=1).to(device)

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

# Z Data - Create placeholders
print("\nCreating zero-filled placeholders for rubrics data (avoiding NaNs)...")
M = len(z_names)
z_data = torch.zeros((N, J, M), dtype=torch.float32).to(device)
z_mask = torch.zeros((N, J, M), dtype=torch.bool).to(device)  # All masked (no data)

# ==========================================
# 3. TRAIN SAE (Sparse Autoencoder)
# ==========================================
print("\nTraining/Loading SAE...")
embeddings_np = x_j_dense.cpu().numpy()

sae = train_sae(
    embeddings=embeddings_np,
    M=1024,
    K=32, # Increased K slightly to capture more nuance
    batch_size=512,
    n_epochs=50, # Reduced epochs for speed
    learning_rate=5e-4,
    checkpoint_dir='checkpoints/my_sae'
)

# Transform dense embeddings to sparse activations
print("Transforming embeddings to SAE activations...")
sae_activations_np = sae.get_activations(embeddings_np)
x_j_input = torch.tensor(sae_activations_np, dtype=torch.float32).to(device)

d_features = sae.m_total_neurons 
print(f"New Feature Dimension (SAE): {d_features}")

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
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.01) # Init smaller
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)    # Sparsity Scale
        
        # [FEATURE 3] Subskills
        self.u_logits = nn.Parameter(torch.ones(M, K_model) * 2.0)
        self.delta_m = nn.Parameter(torch.zeros(1, M)) 

        # [FEATURE 4] Linear Amortized Difficulty 
        self.difficulty_proj = nn.Linear(d_features, 1)

    @property
    def tau(self):
        return F.relu(self.tau_raw)

    def get_gates(self, t):
        return torch.sigmoid(self.u_logits / t)

    def forward(self, temp=1.0):
        # 1. Input
        x_j = self.x_j
        
        # 2. Linear Difficulty Projection
        # (J, d) @ (d, 1) -> (1, J)
        pred_delta = self.difficulty_proj(x_j).squeeze().unsqueeze(0)
        
        # 3. Linear Loading Projection
        W_norm = F.normalize(self.W, dim=1)
        a_j = (x_j @ W_norm.T) * self.tau.unsqueeze(0)

        # 4. Overall Prediction
        logits_y = self.theta @ a_j.T + pred_delta
        
        # 5. Subskill Prediction (Gated)
        g_m = self.get_gates(temp)
        logits_z = []
        for m in range(self.M):
            lz = self.theta @ (a_j * g_m[m].unsqueeze(0)).T + self.delta_m[:, m].unsqueeze(1)
            logits_z.append(lz.unsqueeze(2))
            
        return logits_y, torch.cat(logits_z, dim=2)

# ==========================================
# 5. OPTIMIZATION
# ==========================================
K_MODEL = args.K_MODEL
model = LinearRobustARD(N, J, M, K_MODEL, d_features, x_j_input).to(device)

opt_local = optim.Adam([
    {'params': [model.theta], 'lr': args.lr_latent, 'weight_decay': args.wd_latent}
], lr=args.lr_latent)

opt_global = optim.Adam([
    {'params': model.tau_raw, 'lr': args.lr_tau, 'weight_decay': 0.0},
    {'params': [model.W], 'lr': args.lr_proj, 'weight_decay': args.wd_proj}, 
    {'params': list(model.difficulty_proj.parameters()) + [model.u_logits, model.delta_m], 
     'lr': args.lr_proj, 'weight_decay': args.wd_proj}
], lr=args.lr_proj)

hyperparams = {'lambda_tau': args.lambda_tau}

print("\nStarting Faithful Alternating Optimization...")
n_epochs = 2001

for e in range(n_epochs):
    model.train()
    current_temp = max(0.1, 1.0 - (e / 1000.0))
    
    # --- STEP 1: Local Inference ---
    opt_local.zero_grad()
    logits_y, logits_z = model(temp=current_temp)
    
    lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    lik_z = (F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none') * (z_mask & train_mask.unsqueeze(2))).sum()
    reg_theta = args.reg_theta * torch.sum(model.theta**2)
    
    loss_local = lik_y + lik_z + reg_theta
    loss_local.backward()
    opt_local.step()
    
    # --- STEP 2: Global Update ---
    opt_global.zero_grad()
    logits_y, logits_z = model(temp=current_temp) 
    
    lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    lik_z = (F.binary_cross_entropy_with_logits(logits_z, z_data, reduction='none') * (z_mask & train_mask.unsqueeze(2))).sum()

    gates = torch.sigmoid(model.u_logits / current_temp)
    reg_sparse_gates = args.reg_sparse_gates * torch.sum(gates)
    reg_beta_gates = args.reg_beta_gates * torch.sum(gates * (1.0 - gates))
    reg_tau = hyperparams['lambda_tau'] * torch.norm(model.tau, 1)
    
    loss_global = lik_y + lik_z + reg_tau + reg_sparse_gates + reg_beta_gates
    
    if torch.isnan(loss_global):
        print("WARNING: Loss is NaN! Stopping.")
        break
        
    loss_global.backward()
    opt_global.step()
    
    with torch.no_grad():
        model.tau_raw[model.tau < 0.01] = -0.1

    if e % 100 == 0:
        active_dims = (model.tau > 0.01).sum().item()
        print(f"Ep {e} | Loss: {loss_global.item():.2e} | Active Dims: {active_dims}")

# ==========================================
# 6. EVALUATION
# ==========================================
print("\n=== EVALUATING ===")
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
    
    print(f"\n[Test Set] N={len(y_test)}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"\n[Train Set] N={len(y_train)}")
    print(f"Train AUC: {train_auc:.4f}")