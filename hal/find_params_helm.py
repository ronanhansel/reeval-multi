"""
Grid Search for Optimal HELM Benchmark Parameters
Searches over K_MODEL and lambda_tau to find best parameters per benchmark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import pandas as pd
import warnings
import json
import os
import itertools
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
from hypothesaes.quickstart import train_sae
from tqdm import tqdm
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fixed seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create survey directory
os.makedirs('../survey', exist_ok=True)

# ==========================================
# Model Architecture
# ==========================================
class LinearRobustARD(nn.Module):
    def __init__(self, N, J, K_model, d_features, x_j_input):
        super().__init__()
        self.N, self.J, self.K = N, J, K_model
        self.register_buffer('x_j', x_j_input)

        # Latent Factors
        self.theta = nn.Parameter(torch.randn(N, K_model) * 0.1)
        self.W = nn.Parameter(torch.randn(K_model, d_features) * 0.01)
        self.tau_raw = nn.Parameter(torch.ones(K_model) * 0.5)
        
        # Linear Amortized Difficulty
        self.difficulty_proj = nn.Linear(d_features, 1)

    @property
    def tau(self):
        return F.relu(self.tau_raw)

    def forward(self):
        x_j = self.x_j
        
        # Linear Difficulty Projection
        pred_delta = self.difficulty_proj(x_j).squeeze().unsqueeze(0)
        
        # Linear Loading Projection
        W_norm = F.normalize(self.W, dim=1)
        a_j = (x_j @ W_norm.T) * self.tau.unsqueeze(0)

        # Overall Prediction
        logits_y = self.theta @ a_j.T + pred_delta
        return logits_y


# ==========================================
# Training Function
# ==========================================
def train_model(y_data, train_mask, test_mask, x_j_scenario, d_features, K_MODEL, lambda_tau, 
                reg_theta=0.5, lr_tau=0.01, lr_proj=0.005, lr_latent=0.01,
                wd_proj=1e-2, wd_latent=1e-4, max_epochs=2000, patience=50, min_delta=1e-5, verbose=False):
    """Train a single model with given parameters"""
    
    N, J = y_data.shape
    
    # Initialize model with fixed seed
    torch.manual_seed(SEED)
    model = LinearRobustARD(N, J, K_MODEL, d_features, x_j_scenario).to(device)
    
    # Optimizers
    opt_local = optim.Adam(
        [{'params': [model.theta], 'lr': lr_latent, 'weight_decay': wd_latent}],
        lr=lr_latent
    )
    
    opt_global = optim.Adam([
        {'params': model.tau_raw, 'lr': lr_tau, 'weight_decay': 0.0},
        {'params': [model.W], 'lr': lr_proj, 'weight_decay': wd_proj},
        {'params': list(model.difficulty_proj.parameters()), 'lr': lr_proj, 'weight_decay': wd_proj}
    ], lr=lr_proj)
    
    # Training with early stopping
    best_test_loss = float('inf')
    patience_counter = 0
    final_epoch = 0
    
    for epoch in range(max_epochs):
        model.train()
        
        # Local update (theta)
        opt_local.zero_grad()
        logits_y = model()
        lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
        reg_theta_term = reg_theta * torch.sum(model.theta**2)
        loss_local = lik_y + reg_theta_term
        loss_local.backward()
        opt_local.step()
        
        # Global update (W, difficulty_proj, tau)
        opt_global.zero_grad()
        logits_y = model()
        lik_y = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * train_mask).sum()
        reg_tau = lambda_tau * torch.norm(model.tau, 1)
        loss_global = lik_y + reg_tau
        
        if torch.isnan(loss_global):
            if verbose:
                print("WARNING: Loss is NaN! Stopping.")
            break
        
        loss_global.backward()
        opt_global.step()
        
        with torch.no_grad():
            model.tau_raw[model.tau < 0.01] = -0.1
        
        # Early stopping check
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_y = model()
                test_loss = (F.binary_cross_entropy_with_logits(logits_y, y_data, reduction='none') * test_mask).sum()
                
                if test_loss < best_test_loss - min_delta:
                    best_test_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and epoch % 100 == 0:
                    active_dims = (model.tau > 0.01).sum().item()
                    print(f"Epoch {epoch} | Train Loss: {loss_global.item():.2e} | Test Loss: {test_loss.item():.2e} | Active Dims: {active_dims}")
                
                if patience_counter >= patience:
                    final_epoch = epoch
                    break
        
        final_epoch = epoch
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits_y = model()
        probs = torch.sigmoid(logits_y)
        
        # Test metrics
        y_test = torch.masked_select(y_data, test_mask).cpu().numpy()
        p_test = torch.masked_select(probs, test_mask).cpu().numpy()
        
        # Train metrics
        y_train = torch.masked_select(y_data, train_mask).cpu().numpy()
        p_train = torch.masked_select(probs, train_mask).cpu().numpy()
        
        # Calculate metrics
        test_auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.0
        test_acc = accuracy_score(y_test, (p_test > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, p_train) if len(np.unique(y_train)) > 1 else 0.0
        train_acc = accuracy_score(y_train, (p_train > 0.5).astype(int))
        active_dims = (model.tau > 0.01).sum().item()
    
    # Cleanup
    del model, opt_local, opt_global
    torch.cuda.empty_cache()
    
    return {
        'train_auc': float(train_auc),
        'train_acc': float(train_acc),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'final_epoch': int(final_epoch),
        'active_dims': int(active_dims),
        'n_test_samples': int(len(y_test)),
        'n_train_samples': int(len(y_train))
    }


# ==========================================
# Main Grid Search
# ==========================================
def main():
    # Grid search parameters
    lambda_tau_values = [5, 15, 20, 25, 30, 40, 50, 75, 80, 100]
    K_MODEL_values = [50, 100]
    
    print(f"Grid Search Configuration:")
    print(f"  lambda_tau values: {lambda_tau_values}")
    print(f"  K_MODEL values: {K_MODEL_values}")
    print(f"  Total combinations: {len(lambda_tau_values) * len(K_MODEL_values)}")
    print()
    
    # ==========================================
    # Load Data
    # ==========================================
    print("Loading data...")
    y_df = pd.read_pickle('../data-reeval-multi/resmat.pkl')
    emb_df = pd.read_pickle('../data/embed_meta-llama_Llama-3.1-8B-Instruct.pkl')

    # Filter Rows/Cols in y_df
    y_df = y_df[y_df.notna().any(axis=1)]
    valid_cols_list = []
    for c in y_df.columns:
        valid_cols_list.append(y_df[c].notna().any() and (y_df[c].dropna() != 0).any())
    y_df = y_df.iloc[:, valid_cols_list]
    print(f"Target Matrix Shape: {y_df.shape}")

    # Alignment Logic
    print("\nAligning Embeddings to Question Text...")
    if 'question' not in emb_df.columns:
        text_col = [c for c in emb_df.columns if 'text' in str(c) or 'question' in str(c)][0]
        emb_df = emb_df.rename(columns={text_col: 'question'})

    # Create embedding lookup
    emb_map = {}
    for _, row in emb_df.iterrows():
        q_text = row['question']
        emb = row['embedding']
        if isinstance(emb, str):
            import ast
            emb = ast.literal_eval(emb)
        emb_map[q_text] = emb

    # Extract questions from y_df columns
    if isinstance(y_df.columns, pd.MultiIndex):
        if 'input.text' in y_df.columns.names:
            questions = y_df.columns.get_level_values('input.text').tolist()
        else:
            questions = y_df.columns.get_level_values(-1).tolist()
    else:
        questions = y_df.columns.tolist()

    # Build aligned lists
    aligned_raw_embs = []
    valid_indices_mask = []
    for q in questions:
        if q in emb_map:
            aligned_raw_embs.append(emb_map[q])
            valid_indices_mask.append(True)
        else:
            valid_indices_mask.append(False)

    valid_indices_mask = np.array(valid_indices_mask)
    print(f"Found embeddings for {valid_indices_mask.sum()} / {len(questions)} items")

    # Filter y_df based on alignment
    y_df = y_df.iloc[:, valid_indices_mask]
    x_j_dense = torch.tensor(np.array(aligned_raw_embs), dtype=torch.float32)
    x_j_dense = F.normalize(x_j_dense, p=2, dim=1).to(device)

    print(f"\nFinal aligned data shape: {y_df.shape}")
    print(f"Embeddings shape: {x_j_dense.shape}")

    # ==========================================
    # Extract Scenarios
    # ==========================================
    if isinstance(y_df.columns, pd.MultiIndex) and 'scenario' in y_df.columns.names:
        scenarios = y_df.columns.get_level_values('scenario').unique().tolist()
    else:
        scenarios = ['all_data']

    print(f"\nFound {len(scenarios)} scenarios/benchmarks:")
    for scenario in scenarios:
        if isinstance(y_df.columns, pd.MultiIndex) and 'scenario' in y_df.columns.names:
            mask = y_df.columns.get_level_values('scenario') == scenario
            n_items = mask.sum()
            n_models = y_df.loc[:, mask].notna().any(axis=1).sum()
        else:
            n_items = y_df.shape[1]
            n_models = y_df.notna().any(axis=1).sum()
        print(f"  {scenario}: {n_models} models, {n_items} items")

    # ==========================================
    # Train SAE
    # ==========================================
    print("\nTraining SAE on all embeddings...")
    embeddings_np = x_j_dense.cpu().numpy()

    sae = train_sae(
        embeddings=embeddings_np,
        M=1024,
        K=32,
        batch_size=512,
        n_epochs=50,
        learning_rate=5e-4,
        checkpoint_dir='checkpoints/helm_sae'
    )

    print("Transforming embeddings to SAE activations...")
    sae_activations_np = sae.get_activations(embeddings_np)
    x_j_global = torch.tensor(sae_activations_np, dtype=torch.float32).to(device)
    d_features = sae.m_total_neurons
    print(f"SAE Feature Dimension: {d_features}")

    # ==========================================
    # Grid Search for Each Benchmark
    # ==========================================
    all_results = {}
    best_params = {}
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Grid Search for Benchmark: {scenario}")
        print(f"{'='*80}")
        
        # Extract scenario-specific data
        if isinstance(y_df.columns, pd.MultiIndex) and 'scenario' in y_df.columns.names:
            mask = y_df.columns.get_level_values('scenario') == scenario
            y_scenario = y_df.loc[:, mask]
            x_j_scenario = x_j_global[mask]
        else:
            y_scenario = y_df
            x_j_scenario = x_j_global
        
        # Convert to arrays
        y_vals = y_scenario.values.astype(np.float32)
        N, J = y_vals.shape
        
        print(f"Scenario shape: {N} models x {J} items")
        
        # Create item-wise train/test split (cold start) with fixed seed
        np.random.seed(SEED)
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
        
        # Grid search
        scenario_results = []
        best_test_auc = -1
        best_param_combo = None
        
        param_combinations = list(itertools.product(K_MODEL_values, lambda_tau_values))
        
        for K_MODEL, lambda_tau in tqdm(param_combinations, desc=f"Grid search {scenario}"):
            print(f"\n  Testing K_MODEL={K_MODEL}, lambda_tau={lambda_tau}")
            
            # Train model
            results = train_model(
                y_data=y_data,
                train_mask=train_mask,
                test_mask=test_mask,
                x_j_scenario=x_j_scenario,
                d_features=d_features,
                K_MODEL=K_MODEL,
                lambda_tau=lambda_tau,
                verbose=False
            )
            
            # Add parameters to results
            results['K_MODEL'] = K_MODEL
            results['lambda_tau'] = lambda_tau
            results['scenario'] = scenario
            results['n_models'] = N
            results['n_items'] = J
            
            scenario_results.append(results)
            
            print(f"    Train AUC: {results['train_auc']:.4f} | Test AUC: {results['test_auc']:.4f} | Test Acc: {results['test_acc']:.4f}")
            
            # Track best parameters
            if results['test_auc'] > best_test_auc:
                best_test_auc = results['test_auc']
                best_param_combo = {'K_MODEL': K_MODEL, 'lambda_tau': lambda_tau}
        
        # Store results
        all_results[scenario] = scenario_results
        best_params[scenario] = {
            'best_K_MODEL': best_param_combo['K_MODEL'],
            'best_lambda_tau': best_param_combo['lambda_tau'],
            'best_test_auc': best_test_auc
        }
        
        # Save scenario results
        scenario_df = pd.DataFrame(scenario_results)
        scenario_df = scenario_df.sort_values('test_auc', ascending=False)
        scenario_df.to_csv(f'survey/grid_search_{scenario}.csv', index=False)
        
        print(f"\n  Best parameters for {scenario}:")
        print(f"    K_MODEL: {best_param_combo['K_MODEL']}")
        print(f"    lambda_tau: {best_param_combo['lambda_tau']}")
        print(f"    Test AUC: {best_test_auc:.4f}")
        print(f"  Results saved to survey/grid_search_{scenario}.csv")
    
    # ==========================================
    # Save Summary Results
    # ==========================================
    # Save best parameters
    with open('survey/best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Create summary DataFrame
    best_params_df = pd.DataFrame(best_params).T
    best_params_df = best_params_df.sort_values('best_test_auc', ascending=False)
    best_params_df.to_csv('survey/best_parameters.csv')
    
    # Save all results
    with open('survey/all_grid_search_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print("\nBest Parameters Summary:")
    print(best_params_df.to_string())
    print(f"\nAll results saved to survey/")
    print(f"  - best_parameters.json")
    print(f"  - best_parameters.csv")
    print(f"  - all_grid_search_results.json")
    print(f"  - grid_search_<scenario>.csv (one per benchmark)")
    
    # Generate benchmark_params dictionary for easy copy-paste
    print("\n" + "="*80)
    print("OPTIMIZED PARAMETERS FOR calibration_helm.ipynb:")
    print("="*80)
    print("\nbenchmark_params = {")
    for scenario in scenarios:
        params = best_params[scenario]
        print(f"    '{scenario}': {{'K_MODEL': {params['best_K_MODEL']}, 'lambda_tau': {params['best_lambda_tau']}}},")
    print("}")


if __name__ == "__main__":
    main()
