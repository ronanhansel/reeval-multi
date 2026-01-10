import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import ast
import warnings
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 0. SETUP & DATA LOADING
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, nargs='+', default=None)
args = parser.parse_args()

print("Loading data...")
y_df = pd.read_pickle('../../data-reeval-multi/resmat.pkl')
emb_df = pd.read_pickle('../../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl')

# 1. Filter Benchmarks
if args.benchmark:
    print(f"Filtering by: {args.benchmark}")
    benchmark_mask = np.array([any(b in str(c) for b in args.benchmark) for c in y_df.columns])
    y_df = y_df.iloc[:, benchmark_mask]

# 2. Filter Valid Rows/Cols
y_df = y_df[y_df.notna().any(axis=1)]
valid_cols_list = []
for c in y_df.columns:
    valid_cols_list.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
y_df = y_df.iloc[:, valid_cols_list]

# 3. Align Embeddings
print("Aligning Embeddings...")
if 'question' not in emb_df.columns:
    text_col = [c for c in emb_df.columns if 'text' in str(c) or 'question' in str(c)][0]
    emb_df = emb_df.rename(columns={text_col: 'question'})

emb_map = {}
for _, row in emb_df.iterrows():
    q_text = row['question']
    emb = row['embedding']
    if isinstance(emb, str): emb = ast.literal_eval(emb)
    emb_map[q_text] = emb

# Get Questions
try:
    if isinstance(y_df.columns, pd.MultiIndex) and 'input.text' in y_df.columns.names:
        questions = y_df.columns.get_level_values('input.text').tolist()
    else:
        questions = [str(c) for c in y_df.columns]
except:
    questions = [str(c) for c in y_df.columns]

# Build Aligned Tensors
aligned_embs = []
valid_indices = []
for i, q in enumerate(questions):
    if q in emb_map:
        aligned_embs.append(emb_map[q])
        valid_indices.append(True)
    else:
        valid_indices.append(False)

valid_indices = np.array(valid_indices)
y_df = y_df.iloc[:, valid_indices]
x_j_dense = torch.tensor(np.array(aligned_embs), dtype=torch.float32).to(device)
x_j_dense = F.normalize(x_j_dense, p=2, dim=1)

print(f"Final Data Shape: {y_df.shape}")

# 4. Train/Test Split (Cold Start Item Split)
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
train_mask_t = torch.from_numpy(train_mask).to(device)
test_mask_t = torch.from_numpy(test_mask).to(device)

# ==========================================
# MODEL 0: NAIVE USER MEAN (Baseline)
# ==========================================
print("\n=== Training Model 0: Naive User Mean ===")
# Calculate mean accuracy for each user on TRAINING items only
user_sums = (y_data * train_mask_t).sum(dim=1)
user_counts = train_mask_t.sum(dim=1)
# Avoid division by zero
user_means = user_sums / (user_counts + 1e-6)

# Prediction is simply the user's mean repeated for all items
# (N,) -> (N, 1) -> (N, J)
naive_preds = user_means.unsqueeze(1).expand(N, J)

# Evaluate
y_tr = torch.masked_select(y_data, train_mask_t).cpu().numpy()
p_tr = torch.masked_select(naive_preds, train_mask_t).cpu().numpy()
train_auc = roc_auc_score(y_tr, p_tr)

y_te = torch.masked_select(y_data, test_mask_t).cpu().numpy()
p_te = torch.masked_select(naive_preds, test_mask_t).cpu().numpy()
test_auc = roc_auc_score(y_te, p_te)

print(f"Naive Train AUC: {train_auc:.4f}")
print(f"Naive Test AUC:  {test_auc:.4f} (Captures Theta, Ignores Difficulty)")


# ==========================================
# MODEL 1: SIMPLE RASCH (Item-Specific Parameters)
# ==========================================
class SimpleRasch(nn.Module):
    def __init__(self, N, J):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(N))
        self.beta = nn.Parameter(torch.zeros(J))
        
    def forward(self):
        return self.theta.unsqueeze(1) - self.beta.unsqueeze(0)

print("\n=== Training Model 1: Simple Rasch (Item-Specific) ===")
rasch = SimpleRasch(N, J).to(device)
opt_rasch = optim.Adam(rasch.parameters(), lr=0.1)

for e in range(1001):
    rasch.train()
    opt_rasch.zero_grad()
    logits = rasch()
    loss = (F.binary_cross_entropy_with_logits(logits, y_data, reduction='none') * train_mask_t).sum()
    loss.backward()
    opt_rasch.step()
    
    if e % 500 == 0:
        print(f"Ep {e} | Loss: {loss.item():.2e}")

rasch.eval()
with torch.no_grad():
    logits = rasch()
    probs = torch.sigmoid(logits)
    
    y_tr = torch.masked_select(y_data, train_mask_t).cpu().numpy()
    p_tr = torch.masked_select(probs, train_mask_t).cpu().numpy()
    train_auc = roc_auc_score(y_tr, p_tr)
    
    y_te = torch.masked_select(y_data, test_mask_t).cpu().numpy()
    p_te = torch.masked_select(probs, test_mask_t).cpu().numpy()
    test_auc = roc_auc_score(y_te, p_te)
    
    print(f"Rasch Train AUC: {train_auc:.4f}")
    print(f"Rasch Test AUC:  {test_auc:.4f} (Warning: Uses learned betas for test items)")


# ==========================================
# MODEL 2: AMORTIZED DIFFICULTY (Text-Based)
# ==========================================
class AmortizedRasch(nn.Module):
    def __init__(self, N, d_model, x_input):
        super().__init__()
        self.x_j = x_input
        self.theta = nn.Parameter(torch.zeros(N))
        self.diff_proj = nn.Linear(d_model, 1)
        
    def forward(self):
        pred_beta = self.diff_proj(self.x_j).squeeze().unsqueeze(0)
        return self.theta.unsqueeze(1) + pred_beta

print("\n=== Training Model 2: Amortized Difficulty (Text-Based) ===")
amort = AmortizedRasch(N, x_j_dense.shape[1], x_j_dense).to(device)
opt_amort = optim.Adam(amort.parameters(), lr=0.01)

for e in range(1001):
    amort.train()
    opt_amort.zero_grad()
    logits = amort()
    loss = (F.binary_cross_entropy_with_logits(logits, y_data, reduction='none') * train_mask_t).sum()
    loss.backward()
    opt_amort.step()
    
    if e % 500 == 0:
        print(f"Ep {e} | Loss: {loss.item():.2e}")

amort.eval()
with torch.no_grad():
    logits = amort()
    probs = torch.sigmoid(logits)
    
    y_tr = torch.masked_select(y_data, train_mask_t).cpu().numpy()
    p_tr = torch.masked_select(probs, train_mask_t).cpu().numpy()
    train_auc = roc_auc_score(y_tr, p_tr)
    
    y_te = torch.masked_select(y_data, test_mask_t).cpu().numpy()
    p_te = torch.masked_select(probs, test_mask_t).cpu().numpy()
    test_auc = roc_auc_score(y_te, p_te)
    
    print(f"Amortized Train AUC: {train_auc:.4f}")
    print(f"Amortized Test AUC:  {test_auc:.4f} (True Cold Start capability)")