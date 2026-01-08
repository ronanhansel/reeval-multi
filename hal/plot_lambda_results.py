#!/usr/bin/env python3
"""
Plot lambda_tau experiment results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Get benchmark string from command line or use default
if len(sys.argv) > 1:
    benchmark_str = sys.argv[1]
else:
    benchmark_str = 'assistantbench_taubench_airline_corebench_scienceagentbench'

summary_file = f'lambda_tau_summary_{benchmark_str}.json'

try:
    with open(summary_file, 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"Error: {summary_file} not found")
    print("Please run run_lambda_experiments.py first")
    sys.exit(1)

# Extract data
lambda_values = [r['lambda_tau'] for r in results]
test_aucs = [r['test_auc'] for r in results]
train_aucs = [r['train_auc'] for r in results]
test_accs = [r['test_acc'] for r in results]
active_dims = [r['active_dims'] for r in results]

# Sort by lambda_tau
sorted_data = sorted(zip(lambda_values, test_aucs, train_aucs, test_accs, active_dims))
lambda_values, test_aucs, train_aucs, test_accs, active_dims = zip(*sorted_data)

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Plot 1: Test AUC vs lambda_tau
ax1 = axes[0]
ax1.plot(lambda_values, test_aucs, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Test AUC')
ax1.plot(lambda_values, train_aucs, 's--', linewidth=2, markersize=6, color='#A23B72', alpha=0.7, label='Train AUC')
ax1.set_xlabel('Lambda_tau (L1 Regularization Strength)', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance vs Lambda_tau', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='best')
# ax1.set_xscale('log')

# Highlight best test AUC
best_idx = test_aucs.index(max(test_aucs))
ax1.scatter([lambda_values[best_idx]], [test_aucs[best_idx]], 
           color='red', s=200, zorder=5, marker='*', 
           label=f'Best: λ={lambda_values[best_idx]}, AUC={test_aucs[best_idx]:.4f}')
ax1.legend(fontsize=11, loc='best')

# Plot 2: Test Accuracy vs lambda_tau
ax2 = axes[1]
ax2.plot(lambda_values, test_accs, 'o-', linewidth=2, markersize=8, color='#F18F01')
ax2.set_xlabel('Lambda_tau (L1 Regularization Strength)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Test Accuracy vs Lambda_tau', fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xscale('log')

# Plot 3: Active Dimensions vs lambda_tau
ax3 = axes[2]
ax3.plot(lambda_values, active_dims, 'o-', linewidth=2, markersize=8, color='#6A994E')
ax3.set_xlabel('Lambda_tau (L1 Regularization Strength)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Active Dimensions', fontsize=12, fontweight='bold')
ax3.set_title('Model Sparsity vs Lambda_tau', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xscale('log')

plt.tight_layout()

# Save figure
output_file = f'lambda_tau_results_{benchmark_str}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Print summary statistics
print("\n" + "=" * 70)
print("Lambda_tau Experiment Summary")
print("=" * 70)
print(f"{'Lambda_tau':<12} {'Test AUC':<12} {'Train AUC':<12} {'Test Acc':<12} {'Active Dims':<12}")
print("-" * 70)
for lam, test_auc, train_auc, test_acc, active in zip(lambda_values, test_aucs, train_aucs, test_accs, active_dims):
    marker = " ★" if test_auc == max(test_aucs) else ""
    print(f"{lam:<12.1f} {test_auc:<12.4f} {train_auc:<12.4f} {test_acc:<12.4f} {active:<12}{marker}")
print("=" * 70)
print(f"\nBest lambda_tau: {lambda_values[best_idx]} (Test AUC: {test_aucs[best_idx]:.4f})")
print(f"Active dimensions at best: {active_dims[best_idx]}")

# Show plot
plt.show()
