import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle

# ===================================================================
# A) Configuration & Device Setup
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
RESULT_DIR = "./output"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================================================================
# B) Neural Network for Predicting MIRT Parameters from Embeddings
# ===================================================================
class MIRTParamPredictor(nn.Module):
    """Neural network to predict MIRT parameters (a and b) from question embeddings."""
    
    def __init__(self, embedding_dim, k_factors):
        super(MIRTParamPredictor, self).__init__()
        self.k_factors = k_factors
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Separate heads for a and b parameters
        self.a_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, k_factors),
            nn.Softplus()  # Ensure positive discriminations
        )
        
        self.b_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # No activation for difficulty (can be negative)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        a_params = self.a_head(shared_features)  # Shape: (batch_size, k_factors)
        b_params = self.b_head(shared_features).squeeze(-1)  # Shape: (batch_size,)
        return a_params, b_params

# ===================================================================
# C) Data Loading and Preparation
# ===================================================================
def load_embeddings_and_questions():
    """Load question embeddings and match them with response matrix questions."""
    print("Loading embeddings...")
    
    # Load embeddings
    with open("../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl", "rb") as f:
        df_embed = pickle.load(f)
    
    # Load response matrix
    resmat = pd.read_pickle("../data/resmat.pkl")
    
    # Create question to embedding mapping
    question_to_emb = dict(zip(df_embed["question"], df_embed["embedding"]))
    questions = resmat.columns.get_level_values("input.text").tolist()
    embeds = [question_to_emb.get(q, None) for q in questions]
    
    # Count available embeddings
    n_with_embeddings = sum(1 for e in embeds if e is not None)
    print(f"Questions with embeddings: {n_with_embeddings}/{len(embeds)} ({n_with_embeddings/len(embeds):.2%})")
    
    return resmat, embeds, question_to_emb

def prepare_embedding_data(embeds, a_target, b_target):
    """Prepare embedding data for training, filtering out items without embeddings."""
    # Filter items that have embeddings
    valid_indices = [i for i, emb in enumerate(embeds) if emb is not None]
    
    if len(valid_indices) == 0:
        raise ValueError("No items with embeddings found!")
    
    # Extract valid embeddings and targets
    X_embeddings = np.array([embeds[i] for i in valid_indices])
    a_targets = a_target[valid_indices]  # Shape: (n_valid_items, k)
    b_targets = b_target[valid_indices]  # Shape: (n_valid_items,)
    
    return X_embeddings, a_targets, b_targets, valid_indices

# ===================================================================
# D) Training Functions
# ===================================================================
def train_traditional_mirt(resmat, k, n_epochs=20, patience=5, reg_strength=0.01):
    """Train traditional MIRT model to get target parameters for NN training."""
    print(f"Training traditional MIRT with k={k} to get target parameters...")
    
    # Prepare data (same as k-trials.py)
    observed_pairs = np.argwhere(~resmat.isna().values)
    np.random.seed(42)
    np.random.shuffle(observed_pairs)
    
    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    train_pairs = observed_pairs[n_test:]
    
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    n_persons, n_items = resmat.shape
    
    train_rows_t = torch.from_numpy(train_rows).to(device)
    train_cols_t = torch.from_numpy(train_cols).to(device)
    train_ys_t = torch.from_numpy(train_ys).float().to(device)
    
    # Item weights for imbalance
    item_counts = pd.Series(train_cols).value_counts().reindex(range(n_items), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights = inv_freq_weights.iloc[train_cols].values
    train_weights_t = torch.from_numpy(train_weights).float().to(device)
    
    # Initialize model parameters
    theta = torch.randn(n_persons, k, device=device, requires_grad=True)
    a = torch.randn(n_items, k, device=device, requires_grad=True)
    b = torch.randn(n_items, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([theta, a, b], lr=0.01)
    BATCH_SIZE = 65536
    
    train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t, train_weights_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop with early stopping
    best_params = None
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for batch_rows, batch_cols, batch_ys, batch_wts in train_loader:
            optimizer.zero_grad()
            
            theta_r = theta[batch_rows]
            a_c = a[batch_cols]
            b_c = b[batch_cols]
            
            dot = torch.sum(theta_r * a_c, axis=1)
            logits = dot - b_c
            
            loss_per_obs = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            weighted_loss = (batch_wts * loss_per_obs).sum()
            l2_reg = reg_strength * (torch.sum(theta**2) + torch.sum(a**2) + torch.sum(b**2))
            loss = (weighted_loss + l2_reg) / len(batch_ys)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_rows)
        
        avg_epoch_loss = epoch_loss / len(train_dataset)
        print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_params = {
                'theta': theta.detach().clone(),
                'a': a.detach().clone(),
                'b': b.detach().clone()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_params['a'], best_params['b'], best_params['theta']

def train_embedding_predictor(X_embeddings, a_targets, b_targets, k, n_epochs=100, batch_size=64):
    """Train neural network to predict MIRT parameters from embeddings."""
    print(f"Training embedding-based parameter predictor...")
    
    # Create datasets
    X_train_t = torch.tensor(X_embeddings, dtype=torch.float32).to(device)
    a_train_t = torch.tensor(a_targets, dtype=torch.float32).to(device)
    b_train_t = torch.tensor(b_targets, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_t, a_train_t, b_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    embedding_dim = X_embeddings.shape[1]
    model = MIRTParamPredictor(embedding_dim, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        
        for X_batch, a_batch, b_batch in train_loader:
            optimizer.zero_grad()
            
            a_pred, b_pred = model(X_batch)
            
            # Separate losses for a and b parameters
            loss_a = F.mse_loss(a_pred, a_batch)
            loss_b = F.mse_loss(b_pred, b_batch)
            
            # Combined loss
            loss = loss_a + loss_b
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")
    
    return model

# ===================================================================
# E) Evaluation Functions
# ===================================================================
def evaluate_mirt_model(theta, a_pred, b_pred, resmat, train_pairs, test_pairs):
    """Evaluate MIRT model performance."""
    
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat.values[test_rows, test_cols]
    
    # Convert to tensors
    theta = torch.tensor(theta, dtype=torch.float32, device=device)
    a_pred = torch.tensor(a_pred, dtype=torch.float32, device=device)
    b_pred = torch.tensor(b_pred, dtype=torch.float32, device=device)
    
    # Compute predictions
    with torch.no_grad():
        # Training predictions
        theta_train = theta[train_rows]
        a_train = a_pred[train_cols]
        b_train = b_pred[train_cols]
        logits_train = torch.sum(theta_train * a_train, dim=1) - b_train
        probs_train = torch.sigmoid(logits_train).cpu().numpy()
        
        # Test predictions
        theta_test = theta[test_rows]
        a_test = a_pred[test_cols]
        b_test = b_pred[test_cols]
        logits_test = torch.sum(theta_test * a_test, dim=1) - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
    
    # Compute AUC
    train_auc = roc_auc_score(train_ys, probs_train)
    test_auc = roc_auc_score(test_ys, probs_test)
    
    return train_auc, test_auc

# ===================================================================
# F) Main Execution
# ===================================================================
def main():
    print("Starting Embedding-based MIRT Analysis...")
    
    # Load data
    resmat, embeds, question_to_emb = load_embeddings_and_questions()
    
    # Test different k values
    k_values = [2, 3, 4, 5, 6]
    results = []
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Processing k = {k}")
        print(f"{'='*50}")
        
        try:
            # Step 1: Train traditional MIRT to get target parameters
            a_target, b_target, theta_target = train_traditional_mirt(resmat, k)
            
            # Step 2: Prepare embedding data
            X_embeddings, a_targets, b_targets, valid_indices = prepare_embedding_data(
                embeds, a_target.cpu().numpy(), b_target.cpu().numpy()
            )
            
            # Step 3: Train embedding predictor
            embedding_model = train_embedding_predictor(X_embeddings, a_targets, b_targets, k)
            
            # Step 4: Predict parameters for all items
            embedding_model.eval()
            n_items = len(embeds)
            embedding_dim = len(next(e for e in embeds if e is not None))
            
            # Create full embedding matrix (zero for items without embeddings)
            full_embeddings = np.zeros((n_items, embedding_dim))
            for i, emb in enumerate(embeds):
                if emb is not None:
                    full_embeddings[i] = emb
            
            with torch.no_grad():
                input_tensor = torch.tensor(full_embeddings, dtype=torch.float32).to(device)
                a_pred_all, b_pred_all = embedding_model(input_tensor)
                
                # For items without embeddings, use traditional parameters
                a_final = a_target.clone()
                b_final = b_target.clone()
                
                # Replace with NN predictions for items with embeddings
                for i, idx in enumerate(valid_indices):
                    a_final[idx] = a_pred_all[idx]
                    b_final[idx] = b_pred_all[idx]
            
            # Step 5: Evaluate
            observed_pairs = np.argwhere(~resmat.isna().values)
            np.random.seed(42)
            np.random.shuffle(observed_pairs)
            
            test_frac = 0.20
            n_test = int(len(observed_pairs) * test_frac)
            test_pairs = observed_pairs[:n_test]
            train_pairs = observed_pairs[n_test:]
            
            train_auc, test_auc = evaluate_mirt_model(
                theta_target.cpu().numpy(), 
                a_final.cpu().numpy(), 
                b_final.cpu().numpy(), 
                resmat, train_pairs, test_pairs
            )
            
            print(f"Results for k={k}:")
            print(f"  Train AUC: {train_auc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")
            
            results.append({
                'k': k,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'n_with_embeddings': len(valid_indices),
                'n_total_items': n_items
            })
            
            # Save model
            model_path = os.path.join(RESULT_DIR, f"embedding_mirt_k{k}.pt")
            torch.save({
                'embedding_model': embedding_model.state_dict(),
                'theta': theta_target,
                'a': a_final,
                'b': b_final,
                'k': k,
                'embedding_dim': embedding_dim
            }, model_path)
            
        except Exception as e:
            print(f"Error processing k={k}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULT_DIR, "embedding_mirt_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(results_df)
    
    # Find best k
    best_result = results_df.loc[results_df['test_auc'].idxmax()]
    print(f"\nBest configuration: k={best_result['k']} with Test AUC={best_result['test_auc']:.4f}")

if __name__ == "__main__":
    main()
