import torch
import numpy as np
from hypothesaes.quickstart import train_sae, interpret_sae
import pandas as pd
import torch.nn.functional as F
import ast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
y_df = pd.read_pickle('../../data-reeval-multi/resmat.pkl')
emb_df = pd.read_pickle('../../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl')
texts_list = emb_df['question'].tolist()

print("\nPreparing Embeddings...")
# Directly align embeddings with y_df columns (already in same order)
raw_embs = emb_df['embedding'].tolist()
raw_embs = [ast.literal_eval(e) if isinstance(e, str) else e for e in raw_embs]

x_j_input = torch.tensor(np.array(raw_embs), dtype=torch.float32)
x_j_input = F.normalize(x_j_input, p=2, dim=1).to(device)

d_features = x_j_input.shape[1]

# 1. Prepare your embeddings from your existing code
# x_j_input is currently a Tensor on device; convert to numpy for SAE training
embeddings_np = x_j_input.cpu().numpy()

# 2. Train the SAE
# M: Total number of features to learn (e.g., 512, 1024, or larger for more granularity)
# K: Number of active features per item (sparsity constraint, e.g., 32)
sae = train_sae(
    embeddings=embeddings_np,
    M=1024,
    K=32, # Increased K slightly to capture more nuance
    batch_size=512,
    n_epochs=50, # Reduced epochs for speed
    learning_rate=5e-4,
    checkpoint_dir='checkpoints/my_sae'
)


# This generates a dataframe with descriptions for the neurons
# Requires an OpenAI API key or a local LLM setup
feature_descriptions_df = interpret_sae(
    texts=texts_list,           # The raw text for each item
    embeddings=embeddings_np,   # The dense embeddings
    sae=sae,
    n_top_neurons=50,           # Interpret the 50 most active features
    interpreter_model="gpt-4"   # Or a local model if configured
)

print(feature_descriptions_df[['neuron_idx', 'interpretation']])