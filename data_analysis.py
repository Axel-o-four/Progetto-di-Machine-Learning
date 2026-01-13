import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
import numpy as np
from scipy import stats

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 0. SETUP ---
output_folder = 'analysis_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the dataset
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("ERROR: 'creditcard.csv' file not found. Please, enter the file in the root directory to proceed.")
    exit()

# --- 1. DATASET OVERVIEW ---
print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# --- 2. DATASET ANALYSIS ---
results = []
features_to_analyze = df.columns.tolist()

for col in features_to_analyze:
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    mean_val = df[col].mean()
    min_val = df[col].min()
    max_val = df[col].max()
    std_val = df[col].std()
    
    summary = f"Mean: {mean_val:,.2f}\nStd: {std_val:,.2f}\nMin: {min_val:,.2f} | Max: {max_val:,.2f}"
    results.append({
        'Feature': col,
        'Type': 'Numeric',
        'Missing': f"{missing_count} ({missing_pct:.1f}%)",
        'Statistics': summary
    })
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    plot_path = os.path.join(output_folder, f'distribution_{col}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution graph for '{col}' saved.")

# Combined histogram for V1 to V28 features
v_features = [f'V{j}' for j in range(1, 29)]
fig, axes = plt.subplots(7, 4, figsize=(16, 20))
axes = axes.flatten()

for i, col in enumerate(v_features):
    axes[i].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    kde = stats.gaussian_kde(df[col].dropna())
    x_range = np.linspace(df[col].min(), df[col].max(), 100)
    axes[i].plot(x_range, kde(x_range) * len(df[col]) * (df[col].max() - df[col].min()) / 30, 
                 color='skyblue', linewidth=2)
    axes[i].set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=8)
    axes[i].set_ylabel('Frequency', fontsize=8)
    axes[i].grid(alpha=0.3, axis='y')

plt.tight_layout()
v_combined_path = os.path.join(output_folder, 'distribution_V1-28_combined.png')
plt.savefig(v_combined_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Combined histogram for V1-V28 features saved.")

# Create summary DataFrame
summary_df = pd.DataFrame(results)

# --- 3. SAVE SUMMARY TABLE AS IMAGE ---
fig, ax = plt.subplots(figsize=(16, max(12, len(summary_df) * 0.5)))
ax.axis('tight')
ax.axis('off')

the_table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    loc='center',
    cellLoc='left',
    colWidths=[0.15, 0.1, 0.15, 0.6]
)

the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
the_table.scale(1, 2.5)

for (row, col), cell in the_table.get_celld().items():
    if row > 0:
        cell.get_text().set_wrap(True)

fig.text(0.5, 0.02, f'Total Samples: {len(df):,}', ha='center', fontsize=12, fontweight='bold')

output_path = os.path.join(output_folder, 'statistical_summary.png')
try:
    plt.savefig(output_path, bbox_inches='tight', dpi=200, pad_inches=0.4)
    print(f"\nSummary table saved as image: {output_path}")
except Exception as e:
    print(f"Error saving summary table: {e}")

plt.close()

# --- 4. CLASS DISTRIBUTION TABLE ---
if 'Class' in df.columns:
    class_counts = df['Class'].value_counts().sort_index()
    class_labels = {0: 'Legitimate', 1: 'Fraudulent'}
    
    class_data = []
    for class_val in sorted(class_counts.index):
        count = class_counts[class_val]
        pct = (count / len(df)) * 100
        label = class_labels.get(class_val, f'Class {class_val}')
        class_data.append({
            'Class': label,
            'Samples': f"{count:,}",
            'Percentage': f"{pct:.2f}%"
        })
    
    class_dist_df = pd.DataFrame(class_data)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    class_table = ax.table(
        cellText=class_dist_df.values,
        colLabels=class_dist_df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.3, 0.3]
    )
    
    class_table.auto_set_font_size(False)
    class_table.set_fontsize(10)
    class_table.scale(1, 2.5)
    
    fig.text(0.5, 0.02, f'Total Samples: {len(df):,}', ha='center', fontsize=10, fontweight='bold')
    
    class_output_path = os.path.join(output_folder, 'class_balance.png')
    try:
        plt.savefig(class_output_path, bbox_inches='tight', dpi=200, pad_inches=0.4)
        print(f"Class distribution table saved as image: {class_output_path}")
    except Exception as e:
        print(f"Error saving class distribution table: {e}")
    
    plt.close()

# --- 5. BOXPLOT ANALYSIS ---
print("\nGenerating boxplot visualizations...")

# 5.1 Boxplot for Amount
plt.figure(figsize=(10, 6))
plt.boxplot(df['Amount'], vert=True, patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='black'),
            medianprops=dict(color='skyblue', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
plt.title('Boxplot of Amount', fontsize=14, fontweight='bold')
plt.ylabel('Amount', fontsize=12)
plt.grid(alpha=0.3, axis='y')
boxplot_amount_path = os.path.join(output_folder, 'boxplot_amount.png')
plt.savefig(boxplot_amount_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Boxplot for Amount saved.")

# 5.2 Boxplot for Amount by Class
if 'Class' in df.columns:
    plt.figure(figsize=(10, 6))
    class_labels = {0: 'Legitimate', 1: 'Fraudulent'}
    
    data_to_plot = [df[df['Class'] == 0]['Amount'], df[df['Class'] == 1]['Amount']]
    bp = plt.boxplot(data_to_plot, labels=['Legitimate', 'Fraudulent'], patch_artist=True,
                     boxprops=dict(facecolor='skyblue', color='black'),
                     medianprops=dict(color='skyblue', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))
    
    colors = ['skyblue', 'skyblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Amount Distribution by Class', fontsize=14, fontweight='bold')
    plt.ylabel('Amount', fontsize=12)
    plt.grid(alpha=0.3, axis='y')
    boxplot_class_path = os.path.join(output_folder, 'boxplot_amount_by_class.png')
    plt.savefig(boxplot_class_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot for Amount by Class saved.")

print("\n" + "="*70)
print("Analysis complete! All results saved to the 'analysis_output' folder.")
print("="*70)