import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import ast
from sklearn.model_selection import train_test_split
from hypothesaes.quickstart import train_sae

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading and preparing data...")
# Load your data frames as before
y_df = pd.read_pickle('../../../data-reeval-multi/resmat.pkl')
emb_df = pd.read_pickle('../../../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl')

raw_embs = emb_df['embedding'].tolist()
raw_embs = [ast.literal_eval(e) if isinstance(e, str) else e for e in raw_embs]

# Normalize embeddings
x_j_input = torch.tensor(np.array(raw_embs), dtype=torch.float32)
x_j_input = F.normalize(x_j_input, p=2, dim=1).to(device)
embeddings_np = x_j_input.cpu().numpy()

# Split Data (Validation set is crucial for fair comparison)
train_np, val_np = train_test_split(embeddings_np, test_size=0.1, random_state=42)

# ---------------------------------------------------------
# K-Sweeping Loop
# ---------------------------------------------------------
# Heuristics: 
# Small texts (sentences) -> K=4 to 8
# Long texts (paragraphs/docs) -> K=16 to 32 or higher
candidate_ks = [4, 8, 16, 32, 64] 
fixed_m = 1024  # Use the best M you found in the previous step

results = []

print(f"\nSweeping K values (fixed M={fixed_m})...")
print(f"{'K':<5} | {'Val Loss':<10} | {'Change':<10}")
print("-" * 35)

prev_loss = None

for k in candidate_ks:
    # Train SAE with current K
    sae = train_sae(
        embeddings=train_np,
        M=fixed_m,
        K=k,
        val_embeddings=val_np,
        batch_size=512,
        n_epochs=50, 
        learning_rate=5e-4,
        # Save to distinct folders so they don't overwrite
        checkpoint_dir=f'checkpoints/sae_search_k_{k}',
        show_progress=False
    )
    
    # Calculate Validation Loss
    sae.eval()
    val_tensor = torch.tensor(val_np, dtype=torch.float).to(sae.device)
    with torch.no_grad():
        recon, info = sae(val_tensor)
        # Compute pure reconstruction loss (MSE)
        loss_val = sae.compute_loss(val_tensor, recon, info, aux_coef=0.0, multi_coef=0.0).item()
    
    # Calculate improvement from previous K
    change_str = "N/A"
    if prev_loss is not None:
        pct_change = ((loss_val - prev_loss) / prev_loss) * 100
        change_str = f"{pct_change:.1f}%"
    
    print(f"{k:<5} | {loss_val:.4f}     | {change_str}")
    
    results.append({'K': k, 'Loss': loss_val})
    prev_loss = loss_val

print("-" * 35)
print("Recommendation: Pick the K where the '% Change' starts to flatten out (diminishing returns).")