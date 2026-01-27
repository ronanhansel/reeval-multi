import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# ==============================================================================
# 1. DATA ENTRY
# ==============================================================================
# Using the latest values from your summary run
data = {
    'model': ['Baseline Mean', 'Rasch IRT', 'SAE Beta-IRT', 'PCA Beta-IRT'],
    'train_rmse': [0.180234, 0.128248, 0.160648, 0.160115],
    'test_rmse':  [0.192840, 0.189031, 0.174559, 0.172277],
    'train_corr': [0.0,      0.703902, 0.453243, 0.456685], # 0.0 for Baseline Mean
    'test_corr':  [0.0,      0.282237, 0.419988, 0.447056]  # 0.0 for Baseline Mean
}
df = pd.DataFrame(data)

# The theoretical noise limit calculated earlier
NOISE_FLOOR = 0.1247

# ==============================================================================
# 2. PLOTTING CONFIGURATION
# ==============================================================================
# Use default Seaborn style which defaults to the standard 'tab10' palette
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
colors = sns.color_palette() 

# Assign default tab10 colors to models for consistency
model_color_map = {
    'Baseline Mean': colors[7], # Tab:Gray
    'Rasch IRT':     colors[3], # Tab:Red
    'SAE Beta-IRT':  colors[2],  # Tab:Green
    'PCA Beta-IRT':  colors[0], # Tab:Blue
}

# ==============================================================================
# PLOT 1: Zoomed RMSE Bar Chart (The "Performance Gap" Chart)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df))
width = 0.35

# Use neutral colors for Train/Test bars to avoid clashing
rects1 = ax.bar(x - width/2, df['train_rmse'], width, label='Train RMSE', color=colors[2]) # Light Blue/Cyan
rects2 = ax.bar(x + width/2, df['test_rmse'], width, label='Test RMSE', color=colors[0]) # Standard Blue

# Theoretical Limit Line
ax.axhline(NOISE_FLOOR, color="firebrick", linestyle='--', linewidth=2) # Red dashed line
ax.text(3.4, NOISE_FLOOR + 0.002, f'Noise Floor ({NOISE_FLOOR})', color="firebrick", fontsize=11, ha='right', fontweight='bold')

# Formatting
ax.set_ylabel('RMSE (Lower is Better)')
ax.set_title('RMSE Comparison: Closing the Gap to the Noise Floor')
ax.set_xticks(x)
ax.set_xticklabels(df['model'])
ax.set_ylim(0.12, 0.20)  # ZOOMED
ax.legend(loc='upper right', frameon=True)

# Add value labels
def autolabel(rects, is_test=False):
    for rect in rects:
        height = rect.get_height()
        font_weight = 'bold' if is_test else 'normal'
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight=font_weight)

autolabel(rects1)
autolabel(rects2, is_test=True)

plt.tight_layout()
plt.savefig('paper_rmse_comparison.png', dpi=300)
print("Saved paper_rmse_comparison.png")

# ==============================================================================
# PLOT 3: Pearson Correlation (The "Ranking" Chart)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars using the consistent model colors
bars = ax.bar(df['model'], df['test_corr'], color=colors[0])

ax.set_ylabel('Pearson Correlation (Higher is Better)')
ax.set_title('Test Set Ranking Ability')
ax.set_ylim(0, 0.5)

# Add values
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 0.01,
                'N/A', ha='center', va='bottom', color='gray')

plt.tight_layout()
plt.savefig('paper_correlation.png', dpi=300)
print("Saved paper_correlation.png")