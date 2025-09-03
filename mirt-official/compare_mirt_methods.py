import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from embedding_mirt import MIRTParamPredictor, load_embeddings_and_questions, prepare_embedding_data, train_embedding_predictor

# ===================================================================
# Configuration
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "./output"
os.makedirs(RESULT_DIR, exist_ok=True)

def train_traditional_mirt_full(resmat, k, n_epochs=20, patience=5):
    """
    Complete traditional MIRT training with full evaluation.
    Based on k-trials.py but with cleaner interface.
    """
    print(f"Training traditional MIRT (k={k})...")
    
    # Data preparation
    observed_pairs = np.argwhere(~resmat.isna().values)
    np.random.seed(42)
    np.random.shuffle(observed_pairs)
    
    test_frac = 0.20
    n_test = int(len(observed_pairs) * test_frac)
    test_pairs = observed_pairs[:n_test]
    train_pairs = observed_pairs[n_test:]
    
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat.values[test_rows, test_cols]
    n_persons, n_items = resmat.shape
    
    # Convert to tensors
    train_rows_t = torch.from_numpy(train_rows).to(device)
    train_cols_t = torch.from_numpy(train_cols).to(device)
    train_ys_t = torch.from_numpy(train_ys).float().to(device)
    
    # Item weights for imbalance
    item_counts = pd.Series(train_cols).value_counts().reindex(range(n_items), fill_value=0)
    inv_freq_weights = 1.0 / (item_counts + 1e-6)
    inv_freq_weights /= inv_freq_weights.mean()
    train_weights = inv_freq_weights.iloc[train_cols].values
    train_weights_t = torch.from_numpy(train_weights).float().to(device)
    
    # Initialize parameters
    theta = torch.randn(n_persons, k, device=device, requires_grad=True)
    a = torch.randn(n_items, k, device=device, requires_grad=True)
    b = torch.randn(n_items, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([theta, a, b], lr=0.01)
    train_dataset = TensorDataset(train_rows_t, train_cols_t, train_ys_t, train_weights_t)
    train_loader = DataLoader(train_dataset, batch_size=65536, shuffle=True)
    
    # Training with early stopping
    best_auc = -np.inf
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for batch_rows, batch_cols, batch_ys, batch_wts in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            theta_r = theta[batch_rows]
            a_c = a[batch_cols]
            b_c = b[batch_cols]
            
            dot = torch.sum(theta_r * a_c, axis=1)
            logits = dot - b_c
            
            loss_per_obs = F.binary_cross_entropy_with_logits(logits, batch_ys, reduction='none')
            weighted_loss = (batch_wts * loss_per_obs).sum()
            l2_reg = 0.01 * (torch.sum(theta**2) + torch.sum(a**2) + torch.sum(b**2))
            loss = (weighted_loss + l2_reg) / len(batch_ys)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_rows)
        
        avg_epoch_loss = epoch_loss / len(train_dataset)
        
        # Validation AUC
        with torch.no_grad():
            theta_test = theta[test_rows]
            a_test = a[test_cols]
            b_test = b[test_cols]
            logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
            probs_test = torch.sigmoid(logits_test).cpu().numpy()
            val_auc = roc_auc_score(test_ys, probs_test)
        
        print(f"  Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Val AUC={val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {
                'theta': theta.detach().clone(),
                'a': a.detach().clone(),
                'b': b.detach().clone(),
                'auc': val_auc
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation with best model
    theta, a, b = best_state['theta'], best_state['a'], best_state['b']
    
    with torch.no_grad():
        # Train AUC
        theta_train = theta[train_rows]
        a_train = a[train_cols]
        b_train = b[train_cols]
        logits_train = torch.sum(theta_train * a_train, axis=1) - b_train
        probs_train = torch.sigmoid(logits_train).cpu().numpy()
        train_auc = roc_auc_score(train_ys, probs_train)
        
        # Test AUC
        theta_test = theta[test_rows]
        a_test = a[test_cols]
        b_test = b[test_cols]
        logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        test_auc = roc_auc_score(test_ys, probs_test)
    
    print(f"Traditional MIRT k={k}: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}")
    
    return {
        'theta': theta,
        'a': a,
        'b': b,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs
    }

def train_embedding_mirt_full(resmat, embeds, k):
    """
    Complete embedding-based MIRT training.
    """
    print(f"Training embedding MIRT (k={k})...")
    
    # Step 1: Train traditional MIRT for targets
    traditional_results = train_traditional_mirt_full(resmat, k, n_epochs=10)  # Fewer epochs for target generation
    a_target = traditional_results['a']
    b_target = traditional_results['b']
    theta_target = traditional_results['theta']
    train_pairs = traditional_results['train_pairs']
    test_pairs = traditional_results['test_pairs']
    
    # Step 2: Prepare embedding data
    X_embeddings, a_targets, b_targets, valid_indices = prepare_embedding_data(
        embeds, a_target.cpu().numpy(), b_target.cpu().numpy()
    )
    
    print(f"Using {len(valid_indices)} items with embeddings out of {len(embeds)} total items")
    
    # Step 3: Train embedding predictor
    embedding_model = train_embedding_predictor(X_embeddings, a_targets, b_targets, k, n_epochs=100)
    
    # Step 4: Generate final parameters
    embedding_model.eval()
    n_items = len(embeds)
    embedding_dim = len(next(e for e in embeds if e is not None))
    
    # Create full embedding matrix
    full_embeddings = np.zeros((n_items, embedding_dim))
    for i, emb in enumerate(embeds):
        if emb is not None:
            full_embeddings[i] = emb
    
    with torch.no_grad():
        input_tensor = torch.tensor(full_embeddings, dtype=torch.float32).to(device)
        a_pred_all, b_pred_all = embedding_model(input_tensor)
        
        # Hybrid approach: use NN for items with embeddings, traditional for others
        a_final = a_target.clone()
        b_final = b_target.clone()
        
        # Replace with NN predictions where embeddings are available
        for idx in valid_indices:
            a_final[idx] = a_pred_all[idx]
            b_final[idx] = b_pred_all[idx]
    
    # Step 5: Retrain person parameters with new item parameters
    print("Fine-tuning person parameters with predicted item parameters...")
    theta_new = theta_target.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta_new], lr=0.01)
    
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    
    train_rows_t = torch.from_numpy(train_rows).to(device)
    train_cols_t = torch.from_numpy(train_cols).to(device)
    train_ys_t = torch.from_numpy(train_ys).float().to(device)
    
    for epoch in range(10):  # Quick fine-tuning
        optimizer.zero_grad()
        
        theta_r = theta_new[train_rows_t]
        a_c = a_final[train_cols_t]
        b_c = b_final[train_cols_t]
        
        dot = torch.sum(theta_r * a_c, axis=1)
        logits = dot - b_c
        
        loss = F.binary_cross_entropy_with_logits(logits, train_ys_t)
        loss.backward()
        optimizer.step()
    
    # Final evaluation
    train_rows, train_cols = train_pairs[:, 0], train_pairs[:, 1]
    train_ys = resmat.values[train_rows, train_cols]
    test_rows, test_cols = test_pairs[:, 0], test_pairs[:, 1]
    test_ys = resmat.values[test_rows, test_cols]
    
    with torch.no_grad():
        # Train AUC
        theta_train = theta_new[train_rows]
        a_train = a_final[train_cols]
        b_train = b_final[train_cols]
        logits_train = torch.sum(theta_train * a_train, axis=1) - b_train
        probs_train = torch.sigmoid(logits_train).cpu().numpy()
        train_auc = roc_auc_score(train_ys, probs_train)
        
        # Test AUC
        theta_test = theta_new[test_rows]
        a_test = a_final[test_cols]
        b_test = b_final[test_cols]
        logits_test = torch.sum(theta_test * a_test, axis=1) - b_test
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        test_auc = roc_auc_score(test_ys, probs_test)
    
    print(f"Embedding MIRT k={k}: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}")
    
    return {
        'theta': theta_new,
        'a': a_final,
        'b': b_final,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'embedding_model': embedding_model,
        'n_with_embeddings': len(valid_indices),
        'n_total_items': n_items
    }

def compare_methods():
    """Compare traditional MIRT vs embedding-based MIRT."""
    print("Loading data...")
    resmat, embeds, _ = load_embeddings_and_questions()
    
    k_values = [2, 3, 4, 5, 6]
    results = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"COMPARING METHODS FOR k = {k}")
        print(f"{'='*60}")
        
        try:
            # Traditional MIRT
            traditional_results = train_traditional_mirt_full(resmat, k)
            
            # Embedding MIRT  
            embedding_results = train_embedding_mirt_full(resmat, embeds, k)
            
            # Store comparison results
            results.append({
                'k': k,
                'traditional_train_auc': traditional_results['train_auc'],
                'traditional_test_auc': traditional_results['test_auc'],
                'embedding_train_auc': embedding_results['train_auc'],
                'embedding_test_auc': embedding_results['test_auc'],
                'n_with_embeddings': embedding_results['n_with_embeddings'],
                'n_total_items': embedding_results['n_total_items'],
                'embedding_coverage': embedding_results['n_with_embeddings'] / embedding_results['n_total_items']
            })
            
            # Save individual models
            torch.save({
                'traditional': traditional_results,
                'embedding': embedding_results,
                'k': k
            }, os.path.join(RESULT_DIR, f"comparison_k{k}.pt"))
            
        except Exception as e:
            print(f"Error with k={k}: {e}")
            continue
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULT_DIR, "method_comparison.csv")
    results_df.to_csv(results_path, index=False)
    
    # Display results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(results_df.round(4))
    
    # Create visualization
    create_comparison_plots(results_df)
    
    return results_df

def create_comparison_plots(results_df):
    """Create comparison plots."""
    plt.style.use('default')
    
    # Plot 1: AUC comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training AUC
    ax1.plot(results_df['k'], results_df['traditional_train_auc'], 'b-o', label='Traditional MIRT', linewidth=2, markersize=8)
    ax1.plot(results_df['k'], results_df['embedding_train_auc'], 'r-s', label='Embedding MIRT', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Factors (k)', fontsize=12)
    ax1.set_ylabel('Training AUC', fontsize=12)
    ax1.set_title('Training Performance Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test AUC
    ax2.plot(results_df['k'], results_df['traditional_test_auc'], 'b-o', label='Traditional MIRT', linewidth=2, markersize=8)
    ax2.plot(results_df['k'], results_df['embedding_test_auc'], 'r-s', label='Embedding MIRT', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Factors (k)', fontsize=12)
    ax2.set_ylabel('Test AUC', fontsize=12)
    ax2.set_title('Test Performance Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'mirt_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Performance improvement
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvement = results_df['embedding_test_auc'] - results_df['traditional_test_auc']
    colors = ['green' if x > 0 else 'red' for x in improvement]
    
    bars = ax.bar(results_df['k'], improvement, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Number of Factors (k)', fontsize=12)
    ax.set_ylabel('AUC Improvement\n(Embedding - Traditional)', fontsize=12)
    ax.set_title('Performance Improvement of Embedding MIRT over Traditional MIRT', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, improvement):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001 if height >= 0 else height - 0.002,
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'mirt_improvement.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    print(f"Average Test AUC - Traditional: {results_df['traditional_test_auc'].mean():.4f}")
    print(f"Average Test AUC - Embedding: {results_df['embedding_test_auc'].mean():.4f}")
    print(f"Average Improvement: {improvement.mean():.4f}")
    print(f"Best Traditional (k={results_df.loc[results_df['traditional_test_auc'].idxmax(), 'k']}): {results_df['traditional_test_auc'].max():.4f}")
    print(f"Best Embedding (k={results_df.loc[results_df['embedding_test_auc'].idxmax(), 'k']}): {results_df['embedding_test_auc'].max():.4f}")
    print(f"Embedding Coverage: {results_df['embedding_coverage'].iloc[0]:.1%} of items have embeddings")

if __name__ == "__main__":
    results = compare_methods()
