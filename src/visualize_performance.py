import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def analyze_performance():
    # load data
    if not os.path.exists('data/model_predictions.csv'):
        print("Error: Predictions not found.")
        return

    df = pd.read_csv('data/model_predictions.csv')
    
    # since we optimized the model to only signal '1' for the Top 20% we just need to measure accuracy on those specific rows.
    
    buys = df[df['Predicted_Signal'] == 1]
    total_trades = len(buys)
    wins = len(buys[buys['Target'] == 1])
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    print("\n" + "="*40)
    print("PERFORMANCE AUDIT (Top 20% Strategy)")
    print("="*40)
    print(f"Total Trades Executed: {total_trades}")
    print(f"Win Rate (Precision): {win_rate:.2%}")
    
    # outcomes for plotting
    conditions = [
        (df['Predicted_Signal'] == 1) & (df['Target'] == 1), # Win
        (df['Predicted_Signal'] == 1) & (df['Target'] == 0), # Loss
        (df['Predicted_Signal'] == 0) & (df['Target'] == 1), # Missed
        (df['Predicted_Signal'] == 0) & (df['Target'] == 0)  # Correct Avoid
    ]
    df['Outcome'] = np.select(conditions, ['Win', 'Loss', 'Missed', 'Correct Avoid'], default='Error')

    # generate plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # confusion matrix
    cm = confusion_matrix(df['Target'], df['Predicted_Signal'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                xticklabels=['Ignore', 'Buy'], 
                yticklabels=['Underperform', 'Outperform'], 
                ax=ax[0])
    ax[0].set_title("Confusion Matrix (Top Tier Strategy)")
    
    # outcome distribution
    palette = {'Win': '#2ecc71', 'Loss': '#e74c3c', 'Missed': '#95a5a6', 'Correct Avoid': '#3498db'}
    sns.countplot(data=df, x='Outcome', hue='Outcome', ax=ax[1], palette=palette, legend=False)
    ax[1].set_title(f"Trade Outcomes (Win Rate: {win_rate:.1%})")
    
    plt.tight_layout()
    
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig('results/model_audit.png', dpi=300)
    print(f"✅ Chart saved to: results/model_audit.png")

if __name__ == "__main__":
    analyze_performance()