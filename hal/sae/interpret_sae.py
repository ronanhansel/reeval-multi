from hypothesaes.quickstart import interpret_sae
import torch
import numpy as np
from hypothesaes.quickstart import train_sae
import pandas as pd
import torch.nn.functional as F
import ast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
y_df = pd.read_csv('../data/result_matrix_merged.csv', index_col=0)
emb_df = pd.read_pickle('../data/all_benchmarks_embeddings.pkl')

print("\nPreparing Embeddings...")
emb_map = {str(r['benchmark.task_id']): r['embedding'] for _, r in emb_df.iterrows()}
raw_embs = []
for c in y_df.columns:
    e = emb_map.get(str(c), np.zeros(512))
    if isinstance(e, str): e = ast.literal_eval(e)
    raw_embs.append(e)

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
    M=1024,             # Try 2x or 4x your embedding dimension
    K=32,               # Number of active concepts allowed per item
    batch_size=512,
    n_epochs=100,
    learning_rate=5e-4,
    checkpoint_dir='checkpoints/my_sae'
)

# You need a list of strings corresponding to your embeddings (emb_df)
# texts_list = emb_df['text_content'].tolist() 

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
# Output Example:
# Neuron 0: "involves geometric shape calculation"
# Neuron 1: "requires debugging python code"