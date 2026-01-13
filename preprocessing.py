import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 0. SETUP ---
output_folder = 'preprocessing_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the dataset
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("ERROR: 'creditcard.csv' file not found. Please, enter the file in the root directory to proceed.")
    exit()

# --- 1. DUPLICATE CHECK ---
print("\n" + "="*70)
print("1. DUPLICATE CHECK")
print("="*70)

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. Dataset shape: {df.shape}")

# --- 2. OUTLIER DETECTION AND HANDLING (Amount) ---
print("\n" + "="*70)
print("2. OUTLIER ANALYSIS (Amount feature)")
print("="*70)

# Calculate IQR for Amount
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]
outlier_pct = (len(outliers) / len(df)) * 100

print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)} ({outlier_pct:.2f}%)")
print(f"Outliers range: {outliers['Amount'].min():.2f} - {outliers['Amount'].max():.2f}")

# Decision: Cap outliers instead of removing (preserve data and fraud patterns)
df['Amount'] = df['Amount'].clip(lower=lower_bound, upper=upper_bound)
print(f"\nOutliers capped to bounds [{lower_bound:.2f}, {upper_bound:.2f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before (from original loaded data)
axes[0].hist(pd.read_csv('creditcard.csv')['Amount'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0].set_title('Amount Distribution - Original', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

# After
axes[1].hist(df['Amount'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Amount Distribution - After Outlier Capping', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Frequency')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'amount_before_after_outliers.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Visualization saved: amount_before_after_outliers.png")

# --- 3. FEATURE ENGINEERING: LOG TRANSFORMATION ---
print("\n" + "="*70)
print("3. FEATURE ENGINEERING (Log Transformation)")
print("="*70)

# Apply log transformation to Amount (add 1 to avoid log(0))
df['Amount_log'] = np.log1p(df['Amount'])

print(f"\nOriginal Amount - Skewness: {df['Amount'].skew():.4f}")
print(f"Log-transformed Amount - Skewness: {df['Amount_log'].skew():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Amount'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title(f'Amount - Original (Skewness: {df["Amount"].skew():.4f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

axes[1].hist(df['Amount_log'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title(f'Amount - Log Transformed (Skewness: {df["Amount_log"].skew():.4f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log(Amount)')
axes[1].set_ylabel('Frequency')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'amount_log_transformation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Visualization saved: amount_log_transformation.png")

# Replace original Amount with log-transformed version
df = df.drop('Amount', axis=1)
df = df.rename(columns={'Amount_log': 'Amount'})

# --- 4. TRAIN/TEST SPLIT (STRATIFIED) ---
print("\n" + "="*70)
print("4. TRAIN/TEST SPLIT (Stratified)")
print("="*70)

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

print(f"\nTrain set - Legitimate: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"Train set - Fraudulent: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
print(f"Test set - Legitimate: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
print(f"Test set - Fraudulent: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")

# --- 5. CLASS IMBALANCE HANDLING (SMOTE) ---
print("\n" + "="*70)
print("5. CLASS IMBALANCE HANDLING (SMOTE)")
print("="*70)

# Store original class distribution before SMOTE
y_train_before = y_train.copy()
X_train_before = X_train.copy()

# Apply SMOTE to training set only (test set remains unchanged)
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nBefore SMOTE:")
print(f"  - Legitimate: {(y_train_before == 0).sum()} ({(y_train_before == 0).sum()/len(y_train_before)*100:.2f}%)")
print(f"  - Fraudulent: {(y_train_before == 1).sum()} ({(y_train_before == 1).sum()/len(y_train_before)*100:.2f}%)")
print(f"  - Total samples: {len(y_train_before)}")

print(f"\nAfter SMOTE:")
print(f"  - Legitimate: {(y_train_smote == 0).sum()} ({(y_train_smote == 0).sum()/len(y_train_smote)*100:.2f}%)")
print(f"  - Fraudulent: {(y_train_smote == 1).sum()} ({(y_train_smote == 1).sum()/len(y_train_smote)*100:.2f}%)")
print(f"  - Total samples: {len(y_train_smote)}")
print(f"  - Synthetic samples added: {len(y_train_smote) - len(y_train_before)}")

# Visualize class distribution before and after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before SMOTE
classes_before = y_train_before.value_counts().sort_index()
axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], [classes_before[0], classes_before[1]], color=['skyblue', 'lightcoral'], edgecolor='black', linewidth=2)
axes[0].set_title(f'Class Distribution - Before SMOTE\n(Ratio: {classes_before[0]/classes_before[1]:.1f}:1)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Samples')
axes[0].grid(alpha=0.3, axis='y')

# After SMOTE
classes_after = y_train_smote.value_counts().sort_index()
axes[1].bar(['Legitimate (0)', 'Fraudulent (1)'], [classes_after[0], classes_after[1]], color=['skyblue', 'lightcoral'], edgecolor='black', linewidth=2)
axes[1].set_title(f'Class Distribution - After SMOTE\n(Ratio: {classes_after[0]/classes_after[1]:.1f}:1)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Samples')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'class_distribution_before_after_smote.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\nVisualization saved: class_distribution_before_after_smote.png")

# Use SMOTE-balanced training data for model training
X_train = X_train_smote
y_train = y_train_smote

# --- 6. SAVE PREPROCESSED DATA ---
print("\n" + "="*70)
print("6. SAVING PREPROCESSED DATA")
print("="*70)

X_train.to_csv(os.path.join(output_folder, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(output_folder, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)

print(f"\nX_train.csv - {X_train.shape} (after SMOTE)")
print(f"X_test.csv - {X_test.shape}")
print(f"y_train.csv - {y_train.shape} (after SMOTE)")
print(f"y_test.csv - {y_test.shape}")

print(f"\nTraining Set:")
print(f"  Class 0 (Legitimate): {(y_train == 0).sum()}")
print(f"  Class 1 (Fraudulent): {(y_train == 1).sum()}")

print(f"\nTest Set:")
print(f"  Class 0 (Legitimate): {(y_test == 0).sum()}")
print(f"  Class 1 (Fraudulent): {(y_test == 1).sum()}")

print("\n" + "="*70)
print("Preprocessing complete! Files saved to 'preprocessing_output' folder.")
print("="*70)