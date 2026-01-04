import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util.rename_helper import rename_labels

# load all benchmarks
file = "hal-paper-analysis/qualitative/results/rubrics/rubrics_merged/all_benchmarks_merged.csv"
df = pd.read_csv(file)
df = rename_labels(df)

# Convert label columns to binary indicators
label_columns = ['Self-Correction', 'Tool Use', 'Environmental Barrier', 'Verification', 'Instruction Following']
for col in label_columns:
    df[col] = (df[col] == 'match').astype(int)

df["Binary Success Rate"] = df["Binary Success Rate"].astype(float)

# Compute pairwise correlations
correlation_matrix = df[label_columns + ["Binary Success Rate"]].corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            vmin=-1, vmax=1, square=True)
plt.yticks(rotation=0)  # Make y-axis labels horizontal
plt.title('Rubric Criteria Correlation')
plt.savefig("plots/correlation_matrix.png")
plt.tight_layout()
plt.show()