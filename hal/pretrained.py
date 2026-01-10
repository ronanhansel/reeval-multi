import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import argparse
import ast
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoConfig
import os

base_dir = '/home/azureuser/cloudfiles/code/reeval-multi/hal/sae'
cache_dir = os.path.join(base_dir, '.cache/huggingface')

# Model settings
batch_size = 8
max_chars = 20000  # Truncate extremely long inputs to prevent OOM
# truncate_dim = 512 # Matryoshka embedding dimension

# Setup environment
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. SETUP & ARGS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, nargs='+', default=None)
parser.add_argument('--K_MODEL', type=int, default=10, help="Context length")
# [UPDATED] Switch to Qwen 3 8B
parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B", help="HuggingFace model ID")
parser.add_argument('--lr_context', type=float, default=0.01)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
y_df = pd.read_csv('../data/result_matrix_merged.csv', index_col=0)
emb_df = pd.read_pickle('../result/all_benchmarks_embeddings_4096_8B.pkl')

if args.benchmark:
    cols = [c for c in y_df.columns if any(b in str(c) for b in args.benchmark)]
    y_df = y_df[cols]

y_df = y_df[y_df.notna().any(axis=1)]
valid_cols = []
for c in y_df.columns:
    valid_cols.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
y_df = y_df.loc[:, valid_cols]

# ==========================================
# 2. ITEM-WISE SPLIT
# ==========================================
y_vals = y_df.values.astype(np.float32)
N, J = y_vals.shape
J_indices = np.arange(J)
np.random.shuffle(J_indices)
n_test = int(0.1 * J)
test_idx = J_indices[:n_test]
train_idx = J_indices[n_test:]

train_mask = torch.from_numpy((~np.isnan(y_vals))[:, train_idx]).to(device)
test_mask = torch.from_numpy((~np.isnan(y_vals))[:, test_idx]).to(device)
y_data = torch.from_numpy(np.nan_to_num(y_vals, nan=0.0)).to(device)

# ==========================================
# 3. PREPARE INPUTS (4096 Dims)
# ==========================================
print("\nPreparing Embeddings...")
emb_map = {str(r['benchmark.task_id']): r['embedding'] for _, r in emb_df.iterrows()}
raw_embs = []

INPUT_DIM = 4096 # You mentioned your embeddings are now 4096

for c in y_df.columns:
    e = emb_map.get(str(c), np.zeros(INPUT_DIM))
    if isinstance(e, str): e = ast.literal_eval(e)
    # Pad/Crop safety check
    if len(e) != INPUT_DIM:
        e_fixed = np.zeros(INPUT_DIM)
        min_len = min(len(e), INPUT_DIM)
        e_fixed[:min_len] = e[:min_len]
        e = e_fixed
    raw_embs.append(e)

x_j_input = torch.tensor(np.array(raw_embs), dtype=torch.float32).to(device)

# ==========================================
# 4. QWEN 3 (8B) EVALUATOR (FIXED)
# ==========================================
class QwenEvaluator(nn.Module):
    def __init__(self, N, input_dim, context_len, model_name):
        super().__init__()
        self.N = N
        self.context_len = context_len
        
        # [FIX] Define J (number of items) from the global input tensor
        self.J = x_j_input.shape[0]
        
        print(f"Loading {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Load in bfloat16/float16
        self.backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto" 
        )
        
        # Freeze Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hidden_dim = self.config.hidden_size 
        print(f"Model Hidden Dimension: {self.hidden_dim}")
        print(f"Input Embedding Dimension: {input_dim}")
        print(f"Number of Items (J): {self.J}")

        # Adapter: 4096 -> 4096 (or 2560 if using smaller models)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim).to(device)

        # Context Theta 
        self.theta_context = nn.Parameter(torch.randn(N, context_len, self.hidden_dim) * 0.02)
        
        # Head
        self.head = nn.Linear(self.hidden_dim, 1).to(device)
        
        self.register_buffer('item_embeddings', x_j_input)

    def forward(self):
        # 1. Project Items
        items_proj = self.input_proj(self.item_embeddings) # [J, Hidden]
        items_expanded = items_proj.unsqueeze(1) # [J, 1, Hidden]

        logits_list = []
        
        # Cast parameters to match backbone dtype while keeping device
        dtype = self.backbone.dtype
        theta = self.theta_context.to(dtype=dtype, device=items_expanded.device)
        items = items_expanded.to(dtype)
        
        for i in range(self.N):
            ctx = theta[i].unsqueeze(0) # [1, context_len, Hidden]
            # [FIX] Now self.J is defined
            ctx_expanded = ctx.expand(self.J, -1, -1)
            
            # [Context, Item]
            input_embeds = torch.cat([ctx_expanded, items], dim=1)
            
            # Forward pass
            outputs = self.backbone(inputs_embeds=input_embeds)
            last_hidden = outputs.last_hidden_state[:, -1, :] 
            
            # Head (cast back to float32 for stable BCE loss)
            logits = self.head(last_hidden.to(torch.float32).to(device)) 
            logits_list.append(logits.T)
            
        return torch.cat(logits_list, dim=0)

# ==========================================
# 5. OPTIMIZATION
# ==========================================
model = QwenEvaluator(
    N, 
    input_dim=INPUT_DIM, 
    context_len=args.K_MODEL, 
    model_name=args.model_name
) 
# Note: Backbone is already on device via device_map="auto"

optimizer = optim.Adam([
    {'params': model.theta_context, 'lr': args.lr_context},
    {'params': model.input_proj.parameters(), 'lr': 1e-3},
    {'params': model.head.parameters(), 'lr': 1e-3}
])

print("\nStarting Qwen3-Based Training...")
for e in range(201):
    model.train()
    optimizer.zero_grad()
    
    logits_y = model()
    loss = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
    
    loss.backward()
    optimizer.step()

    if e % 20 == 0:
        print(f"Ep {e} | Loss {loss.item():.2e}")

# ==========================================
# 6. EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    logits_y = model()
    probs = torch.sigmoid(logits_y)
    
    y_test = torch.masked_select(y_data, test_mask).cpu().numpy()
    p_test = torch.masked_select(probs, test_mask).cpu().numpy()
    
    print(f"\n[Test Set] AUC: {roc_auc_score(y_test, p_test):.4f}")