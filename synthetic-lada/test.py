import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from torch.distributions import Bernoulli
from torch.optim import LBFGS, Adam

# ==========================================
# 1. SETUP & GENERATIVE PROCESS (PURE LADA)
# ==========================================
SEED = 42
print(f"Setting random seed to {SEED}")
np.random.seed(SEED)
torch.manual_seed(SEED)

n_models = 50       # Total Agents
n_train_items = 2000 # Anchor Task Size
n_test_items = 100   # Downstream Task Size
calib_budgets = [5, 10, 15] # Number of Agents used for calibration
true_k = 3

print(">>> STEP 1: Generating Data from Pure LADA Priors...")

# True Hidden Profiles
theta_true = torch.randn(n_models, true_k)

def generate_lada_data(n_items, phi_mean_shift=None):
    d = torch.randn(n_items)
    phi = torch.randn(n_items, true_k)
    
    if phi_mean_shift is not None:
        phi = phi + torch.tensor(phi_mean_shift)
        
    w = F.softmax(phi, dim=1)
    logits = d[None, :] + (theta_true[:, None, :] * w[None, :, :]).sum(dim=2)
    probs = torch.sigmoid(logits)
    resmat = torch.bernoulli(probs).numpy()
    return resmat, probs.mean(dim=1).numpy()

# 1. Anchor Data (Unstructured/General)
train_data, _ = generate_lada_data(n_train_items, phi_mean_shift=None)

# 2. Downstream Tasks
# Skewed: Requires specific skill mix (e.g. Coding)
skew_data, skew_gt = generate_lada_data(n_test_items, phi_mean_shift=[2.0, -1.0, -1.0])
# Even: General intelligence task
even_data, even_gt = generate_lada_data(n_test_items, phi_mean_shift=[0, 0, 0])

# ==========================================
# 2. ANCHOR TRAINING (Get Theta)
# ==========================================
print(">>> STEP 2: Training Anchor Models (Recovering Theta)...")

def fit_irt_anchor(data):
    X = torch.tensor(data, dtype=torch.float32)
    N, I = X.shape
    theta = torch.randn(N, requires_grad=True)
    z = torch.randn(I, requires_grad=True)
    optim = LBFGS([theta, z], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")
    
    def closure():
        optim.zero_grad()
        probs = torch.sigmoid(theta[:, None] + z[None, :])
        nll = -Bernoulli(probs=probs).log_prob(X).sum()
        loss = nll / X.numel() + 0.001*(theta**2).sum()
        loss.backward()
        return loss
    
    for _ in range(50): optim.step(closure)
    return theta.detach()[:, None] # (N, 1)

class LADA_Anchor(nn.Module):
    def __init__(self, n_u, n_i, k):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_u, k))
        self.d = nn.Parameter(torch.randn(n_i))
        self.phi = nn.Parameter(torch.randn(n_i, k))

    def forward(self, u_idx, i_idx):
        w = F.softmax(self.phi[i_idx], dim=1)
        return (self.theta[u_idx] * w).sum(1) + self.d[i_idx]

def fit_lada_anchor(data, k):
    model = LADA_Anchor(n_models, data.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    r, c = np.where(data > -1)
    r_t, c_t = torch.tensor(r), torch.tensor(c)
    y = torch.tensor(data[r, c], dtype=torch.float32)
    
    for _ in range(600):
        opt.zero_grad()
        logits = model(r_t, c_t)
        nll = F.binary_cross_entropy_with_logits(logits, y, reduction='mean')
        reg = 0.01 * (model.theta**2).mean() # Regularize Theta to N(0,1)
        loss = nll + reg
        loss.backward()
        opt.step()
    return model.theta.detach() # (N, K)

# Train all variants once
print("   Fitting IRT...")
theta_irt = fit_irt_anchor(train_data)
thetas_lada = {}
for k in [2, 3, 10, 20]:
    print(f"   Fitting LADA k={k}...")
    thetas_lada[k] = fit_lada_anchor(train_data, k)

# ==========================================
# 3. CALIBRATION & PREDICTION (The New Logic)
# ==========================================
print("\n>>> STEP 3: Evaluation via Calibration (Budget = Agents)...")

def calibrate_and_predict(mode, theta_known, task_data, gt_scores, n_agents_budget):
    """
    1. Split agents into Calib Set (Budget) and Test Set.
    2. Use Calib Set to learn d_new and w_new (or phi_new).
    3. Predict scores for Test Set.
    """
    N, I = task_data.shape
    
    # Shuffle indices to ensure random selection
    indices = np.random.permutation(N)
    calib_idx = indices[:n_agents_budget]
    test_idx = indices[n_agents_budget:]
    
    # Data for Calibration
    # We fix Theta (from anchor) and use observed Responses
    theta_calib = theta_known[calib_idx] 
    resp_calib = torch.tensor(task_data[calib_idx], dtype=torch.float32)
    
    # Init New Task Parameters (to be learned)
    d_new = torch.zeros(I, requires_grad=True)
    
    if mode == "IRT":
        # For IRT, w is fixed to 1.0. We only learn d.
        opt = Adam([d_new], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            # Logits = Theta + d (broadcasting)
            # theta: (N_cal, 1), d: (I) -> (N_cal, I)
            logits = theta_calib + d_new[None, :] 
            loss = F.binary_cross_entropy_with_logits(logits, resp_calib)
            loss.backward()
            opt.step()
        
        # Predict for Test Set
        theta_test = theta_known[test_idx]
        with torch.no_grad():
            pred_logits = theta_test + d_new[None, :]
            pred_probs = torch.sigmoid(pred_logits)
            
    else: # LADA
        K = theta_known.shape[1]
        phi_new = torch.zeros((I, K), requires_grad=True) # Init uniform weights
        
        opt_phi = Adam([phi_new], lr=0.1)
        
        # Block Coordinate Descent
        for _ in range(200):
            # 1. Update d
            d_new.requires_grad_(True)
            w_fixed = F.softmax(phi_new.detach(), dim=1)
            # Logits = d + Theta @ W.T
            logits = d_new[None, :] + (theta_calib @ w_fixed.T)
            loss_d = F.binary_cross_entropy_with_logits(logits, resp_calib)
            grad_d = torch.autograd.grad(loss_d, d_new)[0]
            with torch.no_grad():
                d_new -= 0.1 * grad_d # Gradient Step
            
            # 2. Update Phi
            opt_phi.zero_grad()
            w_dynamic = F.softmax(phi_new, dim=1)
            logits = d_new.detach()[None, :] + (theta_calib @ w_dynamic.T)
            loss_phi = F.binary_cross_entropy_with_logits(logits, resp_calib)
            loss_phi.backward()
            opt_phi.step()
            
        # Predict for Test Set
        theta_test = theta_known[test_idx]
        with torch.no_grad():
            w_final = F.softmax(phi_new, dim=1)
            pred_logits = d_new[None, :] + (theta_test @ w_final.T)
            pred_probs = torch.sigmoid(pred_logits)

    # Evaluate Correlation on Test Set
    preds_avg = pred_probs.mean(dim=1).numpy()
    actuals = gt_scores[test_idx]
    rho, _ = spearmanr(actuals, preds_avg)
    return rho

# ==========================================
# 4. RUN EXPERIMENTS
# ==========================================
results = []

for budget in calib_budgets:
    print(f"\n[Budget: {budget} Agents]")
    
    for task_name, task_data, task_gt in [("Skewed", skew_data, skew_gt), ("Even", even_data, even_gt)]:
        # 1. IRT
        rhos_irt = [calibrate_and_predict("IRT", theta_irt, task_data, task_gt, budget) for _ in range(10)]
        avg_irt = np.mean(rhos_irt)
        
        # 2. LADA Variants
        lada_scores = {}
        for k, th in thetas_lada.items():
            rhos = [calibrate_and_predict("LADA", th, task_data, task_gt, budget) for _ in range(10)]
            lada_scores[k] = np.mean(rhos)
            
        print(f"Task: {task_name:<8} | IRT: {avg_irt:.3f} | LADA k=2: {lada_scores[2]:.3f} | k=3: {lada_scores[3]:.3f} | k=10: {lada_scores[10]:.3f} | k=20: {lada_scores[20]:.3f}")
        
        results.append({
            "Budget": budget, "Task": task_name, "IRT": avg_irt,
            "LADA_k2": lada_scores[2], "LADA_k3": lada_scores[3],
            "LADA_k10": lada_scores[10], "LADA_k20": lada_scores[20]
        })

# ==========================================
# 5. SUMMARY
# ==========================================
print("\n" + "="*80)
print("FINAL SUMMARY: Predictive Power (Spearman Rho)")
print("Budget = Number of Agents used to 'learn' the test structure")
print("="*80)
df = pd.DataFrame(results)
print(df.round(4).to_string(index=False))