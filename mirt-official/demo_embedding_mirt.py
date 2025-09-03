#!/usr/bin/env python3
"""
Demo script showing how to use embedding-based MIRT.

This script demonstrates:
1. Loading embeddings and response data
2. Training an embedding-based MIRT model
3. Making predictions for new items with embeddings
4. Comparing with traditional MIRT

Usage:
    python demo_embedding_mirt.py
"""

import torch
import numpy as np
import pandas as pd
from embedding_mirt import (
    MIRTParamPredictor, 
    load_embeddings_and_questions,
    prepare_embedding_data,
    train_embedding_predictor,
    train_traditional_mirt
)

def demo_basic_usage():
    """Demonstrate basic usage of embedding-based MIRT."""
    print("="*60)
    print("EMBEDDING-BASED MIRT DEMO")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    resmat, embeds, question_to_emb = load_embeddings_and_questions()
    
    n_persons, n_items = resmat.shape
    n_with_embeddings = sum(1 for e in embeds if e is not None)
    
    print(f"   Response matrix: {n_persons} persons × {n_items} items")
    print(f"   Items with embeddings: {n_with_embeddings} ({n_with_embeddings/n_items:.1%})")
    
    # Choose number of factors
    k = 4  # You can experiment with different values
    print(f"\n2. Training models with k={k} factors...")
    
    # Train traditional MIRT for comparison
    print("\n   a) Traditional MIRT (for target generation)...")
    a_target, b_target, theta_target = train_traditional_mirt(resmat, k, n_epochs=15)
    
    # Prepare embedding data
    print("\n   b) Preparing embedding data...")
    X_embeddings, a_targets, b_targets, valid_indices = prepare_embedding_data(
        embeds, a_target.cpu().numpy(), b_target.cpu().numpy()
    )
    
    print(f"      Training NN on {len(valid_indices)} items with embeddings")
    
    # Train embedding predictor
    print("\n   c) Training embedding-based parameter predictor...")
    embedding_model = train_embedding_predictor(
        X_embeddings, a_targets, b_targets, k, n_epochs=50
    )
    
    print("\n3. Making predictions...")
    
    # Create predictions for all items
    embedding_dim = X_embeddings.shape[1]
    full_embeddings = np.zeros((n_items, embedding_dim))
    
    for i, emb in enumerate(embeds):
        if emb is not None:
            full_embeddings[i] = emb
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(full_embeddings, dtype=torch.float32).to(device)
        a_pred, b_pred = embedding_model(input_tensor)
    
    print(f"   Predicted parameters for {n_items} items")
    print(f"   Discrimination parameters (a): shape {a_pred.shape}")
    print(f"   Difficulty parameters (b): shape {b_pred.shape}")
    
    # Show some example predictions vs targets
    print("\n4. Example predictions (first 5 items with embeddings):")
    print("   Item | True a[0] | Pred a[0] | True b    | Pred b")
    print("   " + "-"*50)
    
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        true_a0 = a_target[idx, 0].item()
        pred_a0 = a_pred[idx, 0].item()
        true_b = b_target[idx].item()
        pred_b = b_pred[idx].item()
        
        print(f"   {idx:4d} | {true_a0:8.3f} | {pred_a0:8.3f} | {true_b:8.3f} | {pred_b:8.3f}")
    
    # Show parameter statistics
    print("\n5. Parameter statistics:")
    print(f"   Traditional a parameters - Mean: {a_target.mean():.3f}, Std: {a_target.std():.3f}")
    print(f"   Predicted a parameters   - Mean: {a_pred.mean():.3f}, Std: {a_pred.std():.3f}")
    print(f"   Traditional b parameters - Mean: {b_target.mean():.3f}, Std: {b_target.std():.3f}")
    print(f"   Predicted b parameters   - Mean: {b_pred.mean():.3f}, Std: {b_pred.std():.3f}")
    
    return {
        'embedding_model': embedding_model,
        'traditional_params': {'a': a_target, 'b': b_target, 'theta': theta_target},
        'predicted_params': {'a': a_pred, 'b': b_pred},
        'valid_indices': valid_indices,
        'embeddings': full_embeddings
    }

def demo_new_item_prediction():
    """Demonstrate how to predict parameters for new items with embeddings."""
    print("\n" + "="*60)
    print("NEW ITEM PREDICTION DEMO")
    print("="*60)
    
    # This would normally be done with a trained model
    # For demo purposes, we'll simulate having a trained model
    
    print("\nScenario: You have a trained embedding-based MIRT model and want to")
    print("predict parameters for new items that you have embeddings for.")
    
    # Load a sample question embedding (simulate new item)
    resmat, embeds, question_to_emb = load_embeddings_and_questions()
    
    # Take a few sample embeddings as "new items"
    sample_embeddings = []
    sample_questions = []
    
    questions = resmat.columns.get_level_values("input.text").tolist()
    for i, (question, embed) in enumerate(zip(questions[:3], embeds[:3])):
        if embed is not None:
            sample_embeddings.append(embed)
            sample_questions.append(question[:100] + "..." if len(question) > 100 else question)
    
    if len(sample_embeddings) == 0:
        print("No embeddings available for demo")
        return
    
    print(f"\nPredicting parameters for {len(sample_embeddings)} sample 'new' items:")
    
    # Simulate trained model (in practice, you'd load this)
    k = 4
    embedding_dim = len(sample_embeddings[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy trained model (in practice, you'd load your saved model)
    model = MIRTParamPredictor(embedding_dim, k).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(sample_embeddings, dtype=torch.float32).to(device)
        a_pred, b_pred = model(input_tensor)
    
    # Display predictions
    for i in range(len(sample_embeddings)):
        print(f"\nItem {i+1}: {sample_questions[i]}")
        print(f"  Discrimination (a): {a_pred[i].cpu().numpy()}")
        print(f"  Difficulty (b): {b_pred[i].cpu().item():.3f}")
        
        # Interpretation
        if b_pred[i].cpu().item() > 0:
            difficulty = "hard"
        elif b_pred[i].cpu().item() < -0.5:
            difficulty = "easy"
        else:
            difficulty = "medium"
        
        print(f"  → This item appears to be {difficulty}")

def demo_model_inspection():
    """Demonstrate how to inspect and understand the trained model."""
    print("\n" + "="*60)
    print("MODEL INSPECTION DEMO")  
    print("="*60)
    
    # Train a small model for inspection
    resmat, embeds, _ = load_embeddings_and_questions()
    a_target, b_target, theta_target = train_traditional_mirt(resmat, k=3, n_epochs=5)
    
    X_embeddings, a_targets, b_targets, valid_indices = prepare_embedding_data(
        embeds, a_target.cpu().numpy(), b_target.cpu().numpy()
    )
    
    model = train_embedding_predictor(X_embeddings, a_targets, b_targets, k=3, n_epochs=20)
    
    print("\n1. Model Architecture:")
    print(f"   Input dimension: {X_embeddings.shape[1]} (embedding size)")
    print(f"   Output dimensions: {3} factors for 'a' + 1 for 'b'")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())} total")
    
    print("\n2. Model Layers:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            if hasattr(module, 'weight'):
                print(f"   {name}: {module.weight.shape}")
    
    print("\n3. Prediction Quality:")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(X_embeddings, dtype=torch.float32).to(device)
        a_pred, b_pred = model(input_tensor)
        
        # Calculate correlation between predictions and targets
        a_corr = np.corrcoef(a_pred.cpu().numpy().flatten(), a_targets.flatten())[0, 1]
        b_corr = np.corrcoef(b_pred.cpu().numpy(), b_targets)[0, 1]
        
        print(f"   Correlation for 'a' parameters: {a_corr:.3f}")
        print(f"   Correlation for 'b' parameters: {b_corr:.3f}")
        
        # Mean squared error
        mse_a = np.mean((a_pred.cpu().numpy() - a_targets) ** 2)
        mse_b = np.mean((b_pred.cpu().numpy() - b_targets) ** 2)
        
        print(f"   MSE for 'a' parameters: {mse_a:.4f}")
        print(f"   MSE for 'b' parameters: {mse_b:.4f}")

if __name__ == "__main__":
    print("Running Embedding-based MIRT demonstrations...\n")
    
    try:
        # Basic usage demo
        results = demo_basic_usage()
        
        # New item prediction demo  
        demo_new_item_prediction()
        
        # Model inspection demo
        demo_model_inspection()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("- Try different values of k (number of factors)")
        print("- Experiment with the neural network architecture")
        print("- Use the trained model for new item calibration")
        print("- Compare performance with traditional MIRT using compare_mirt_methods.py")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have the required data files and dependencies.")
