import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    result_path = 'model/result/remediation_impact.csv'
    if not os.path.exists(result_path):
        print(f"File not found: {result_path}")
        return

    df = pd.read_csv(result_path)
    
    sns.set_theme(style="whitegrid")
    
    # Plot AUC
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='scenario', y='test_auc', hue='scenario', palette='viridis', legend=False)
    plt.title('Impact of Item Revision on Test AUC')
    plt.ylabel('Test AUC')
    plt.xlabel('Scenario')
    plt.ylim(0.5, max(df['test_auc']) + 0.1)
    # add value labels
    for i, v in enumerate(df['test_auc']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('model/result/remediation_auc.pdf')
    plt.savefig('model/result/remediation_auc.png', dpi=300)
    print("Saved AUC plot to model/result/remediation_auc.png")

    # Plot RMSE
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='scenario', y='test_rmse', hue='scenario', palette='magma', legend=False)
    plt.title('Impact of Item Revision on Test RMSE')
    plt.ylabel('Test RMSE')
    plt.xlabel('Scenario')
    plt.ylim(0.0, max(df['test_rmse']) + 0.1)
    # add value labels
    for i, v in enumerate(df['test_rmse']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('model/result/remediation_rmse.pdf')
    plt.savefig('model/result/remediation_rmse.png', dpi=300)
    print("Saved RMSE plot to model/result/remediation_rmse.png")
    
    # Print active SAE features for Post-8
    post_8_row = df[df['scenario'] == 'Post-8']
    if not post_8_row.empty:
        idx = post_8_row.iloc[0]['active_indices']
        dims = post_8_row.iloc[0]['active_dims']
        print(f"\nPost-revision (N=8) active dimensions: {dims}")
        print(f"Active SAE feature indices: {idx}")

if __name__ == '__main__':
    main()
