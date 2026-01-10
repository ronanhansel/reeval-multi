import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import ast
from sklearn.model_selection import train_test_split
from hypothesaes.quickstart import train_sae, interpret_sae

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
# Ensure these paths are correct for your environment
y_df = pd.read_pickle('../../../data-reeval-multi/resmat.pkl')
emb_df = pd.read_pickle('../../../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl')
texts_list = emb_df['question'].tolist()

print("\nPreparing Embeddings...")
# Directly align embeddings with y_df columns (already in same order)
raw_embs = emb_df['embedding'].tolist()
raw_embs = [ast.literal_eval(e) if isinstance(e, str) else e for e in raw_embs]

x_j_input = torch.tensor(np.array(raw_embs), dtype=torch.float32)
# Normalize embeddings (Crucial for SAEs)
x_j_input = F.normalize(x_j_input, p=2, dim=1).to(device)

# Convert to numpy for the library
embeddings_np = x_j_input.cpu().numpy()

# ---------------------------------------------------------
# MODIFICATION: Split data to validate reconstruction loss
# ---------------------------------------------------------
print("Splitting data into Train (90%) and Validation (10%)...")
train_np, val_np = train_test_split(embeddings_np, test_size=0.1, random_state=42)

# Define the candidates for M you want to test
candidate_ms = [512, 1024, 2048] 
best_m = None
best_loss = float('inf')
best_sae = None

print(f"\nStarting Search for Best M...")
print(f"{'M':<10} | {'Val Loss (MSE)':<15}")
print("-" * 30)

for m in candidate_ms:
    # Train SAE with current M
    # We use a unique checkpoint dir for each M so they don't overwrite each other
    sae = train_sae(
        embeddings=train_np,
        M=m,
        K=32,
        val_embeddings=val_np,  # Pass validation set for early stopping
        batch_size=512,
        n_epochs=50,
        learning_rate=5e-4,
        checkpoint_dir=f'checkpoints/sae_search_m_{m}', 
        show_progress=False     # Disable progress bar to keep output clean
    )
    
    # Manually compute final validation loss to compare
    sae.eval()
    val_tensor = torch.tensor(val_np, dtype=torch.float).to(sae.device)
    
    with torch.no_grad():
        recon, info = sae(val_tensor)
        # Compute normalized MSE loss
        # aux_coef=0, multi_coef=0 ensures we measure pure reconstruction fidelity
        loss = sae.compute_loss(val_tensor, recon, info, aux_coef=0.0, multi_coef=0.0)
        loss_val = loss.item()
    
    print(f"{m:<10} | {loss_val:.4f}")
    
    # Track the best model
    if loss_val < best_loss:
        best_loss = loss_val
        best_m = m
        best_sae = sae

print("-" * 30)
print(f"Selected Best M: {best_m} (Loss: {best_loss:.4f})")

# ---------------------------------------------------------
# Interpret the Best Model found
# ---------------------------------------------------------
print(f"\nInterpreting the best model (M={best_m})...")

feature_descriptions_df = interpret_sae(
    texts=texts_list,           # Use full list of texts
    embeddings=embeddings_np,   # Use full embeddings to find top examples
    sae=best_sae,               # Use the winning model
    n_top_neurons=50,
    interpreter_model="gpt-4"   
)

print(feature_descriptions_df[['neuron_idx', 'interpretation']])