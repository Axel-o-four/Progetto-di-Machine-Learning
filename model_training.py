import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score)
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 0. SETUP ---
output_folder = 'models_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

preprocessing_folder = 'preprocessing_output'

# Load the dataset

try:
    X_train = pd.read_csv(os.path.join(preprocessing_folder, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(preprocessing_folder, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(preprocessing_folder, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(preprocessing_folder, 'y_test.csv')).values.ravel()
except FileNotFoundError as e:
    print(f"ERROR: preprocessed dataset file(s) not found. Please, run the preprocessing script again to proceed.")
    exit()

# --- 1. TRAIN GAUSSIAN NAIVE BAYES ---
print("\n" + "="*70)
print("TRAINING GAUSSIAN NAIVE BAYES")
print("="*70)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
y_pred_proba_gnb = gnb.predict_proba(X_test)[:, 1]

print("Training complete.")

# --- 2. TRAIN RANDOM FOREST ---
print("\n" + "="*70)
print("TRAINING RANDOM FOREST")
print("="*70)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 5, 10],
    'class_weight': ['balanced']
}

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
cv_folds = 5
total_fits = total_combinations * cv_folds

print(f"\nHyperparameter Tuning Configuration:")
print(f"  - Parameter combinations: {total_combinations}")
print(f"  - Cross-validation folds: {cv_folds}")
print(f"  - Total model fits: {total_fits}")
print(f"\nStarting GridSearchCV...\n")

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_base, 
    param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=2
)

grid_search.fit(X_train, y_train)
print("\n")

rf = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\nBest parameters found:")
for param, value in best_params.items():
    print(f"  - {param}: {value}")
print(f"\nBest CV F1-Score: {grid_search.best_score_:.4f}")

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

print("Training complete.")

# --- 3. EVALUATE MODELS ---
def print_metrics(model_name, y_true, y_pred, y_pred_proba):
    print(f"\n{model_name}:")
    print("-" * 50)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraudulent'], zero_division=0))
    
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
            'F1-Score': f1, 'ROC-AUC': roc_auc}

metrics_gnb = print_metrics('GAUSSIAN NAIVE BAYES', y_test, y_pred_gnb, y_pred_proba_gnb)
metrics_rf = print_metrics('RANDOM FOREST', y_test, y_pred_rf, y_pred_proba_rf)

# --- 4. MODEL COMPARISON ---
comparison_df = pd.DataFrame({
    'Gaussian NB': metrics_gnb,
    'Random Forest': metrics_rf
}).T

print("\n" + comparison_df.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
comparison_df.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.xticks(rotation=45)
plt.ylim([0, 1.1])
plt.grid(alpha=0.3, axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved: model_comparison.png")

# --- 5. CONFUSION MATRICES ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_gnb = confusion_matrix(y_test, y_pred_gnb)
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Gaussian NB - Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'],
            cbar_kws={'label': 'Count'})
axes[1].set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved: confusion_matrices.png")

# --- 6. ROC CURVES ---
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_proba_gnb)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(10, 8))
plt.plot(fpr_gnb, tpr_gnb, label=f'Gaussian NB (AUC = {roc_auc_gnb:.4f})', linewidth=2, color='skyblue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})', linewidth=2, color='lightcoral')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved: roc_curves.png")

# --- 7. SAVE MODELS AND EVALUATION REPORT ---
joblib.dump(gnb, os.path.join(output_folder, 'gaussian_nb_model.pkl'))
joblib.dump(rf, os.path.join(output_folder, 'random_forest_model.pkl'))

print("\nModel saved: gaussian_nb_model.pkl")
print("\nModel saved: random_forest_model.pkl")

report_text = f"""
{'='*70}
FRAUD DETECTION CLASSIFIER - MODEL TRAINING REPORT
{'='*70}

DATASET INFORMATION:
- Training samples: {len(X_train)} (Fraud: {(y_train == 1).sum()}, Legitimate: {(y_train == 0).sum()})
- Test samples: {len(X_test)} (Fraud: {(y_test == 1).sum()}, Legitimate: {(y_test == 0).sum()})
- Features: {X_train.shape[1]}

{'='*70}
GAUSSIAN NAIVE BAYES
{'='*70}
Accuracy:  {metrics_gnb['Accuracy']:.4f}
Precision: {metrics_gnb['Precision']:.4f}
Recall:    {metrics_gnb['Recall']:.4f}
F1-Score:  {metrics_gnb['F1-Score']:.4f}
ROC-AUC:   {metrics_gnb['ROC-AUC']:.4f}

{'='*70}
RANDOM FOREST
{'='*70}
Accuracy:  {metrics_rf['Accuracy']:.4f}
Precision: {metrics_rf['Precision']:.4f}
Recall:    {metrics_rf['Recall']:.4f}
F1-Score:  {metrics_rf['F1-Score']:.4f}
ROC-AUC:   {metrics_rf['ROC-AUC']:.4f}

Hyperparameter Tuning (GridSearchCV with 5-fold CV):
Best CV F1-Score: {grid_search.best_score_:.4f}
Best Parameters:
- n_estimators: {best_params['n_estimators']}
- max_depth: {best_params['max_depth']}
- min_samples_split: {best_params['min_samples_split']}
- min_samples_leaf: {best_params['min_samples_leaf']}
- class_weight: {best_params['class_weight']}
"""

with open(os.path.join(output_folder, 'training_report.txt'), 'w') as f:
    f.write(report_text)

print(f"Training complete! Results saved to 'models_output' folder.")