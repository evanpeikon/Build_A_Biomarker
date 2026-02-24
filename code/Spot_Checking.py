from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cv_folds = 5

# Define scoring metrics
scoring = {'roc_auc': 'roc_auc','precision': make_scorer(precision_score, zero_division=0),'recall': make_scorer(recall_score, zero_division=0),'f1': make_scorer(f1_score, zero_division=0)}

# Define models to spot-check
models = {
   'Logistic Regression (L2)': LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
   'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
   'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
   'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
   'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1),
   'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)}

print(f"\nEvaluation setup:")
print(f"  Method: {cv_folds}-fold cross-validation")
print(f"  Primary metric: ROC-AUC")
print(f"  Secondary metrics: Precision, Recall, F1-score")
print(f"  Training samples: {X_train_selected.shape[0]}")
print(f"  Features: {X_train_selected.shape[1]}")
print(f"\nSpot-checking {len(models)} algorithms...")
print("-"*60)

# Store results
results = []

# Evaluate each model
for name, model in models.items():
   print(f"\nTraining {name}...")
   # Cross-validation with multiple metrics
   cv_results = cross_validate( model, X_train_selected, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1, return_train_score=False)

   # Store mean scores
   results.append({ 'Model': name, 'ROC-AUC': cv_results['test_roc_auc'].mean(),'ROC-AUC Std': cv_results['test_roc_auc'].std(), 'Precision': cv_results['test_precision'].mean(), 'Recall': cv_results['test_recall'].mean(), 'F1-Score': cv_results['test_f1'].mean()})

   print(f"  ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} (+/ {cv_results['test_roc_auc'].std():.4f})")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC-AUC', ascending=False)

# Display results
print("\n" + "="*60)
print("SPOT-CHECKING RESULTS")
print("="*60)
print("\nRanked by ROC-AUC:")
print(results_df.to_string(index=False))

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: ROC-AUC comparison with error bars
ax1 = axes[0]
models_sorted = results_df['Model'].tolist()
aucs = results_df['ROC-AUC'].tolist()
stds = results_df['ROC-AUC Std'].tolist()

y_pos = np.arange(len(models_sorted))
bars = ax1.barh(y_pos, aucs, xerr=stds, alpha=0.7, capsize=5)

# Color bars by performance
colors = ['green' if auc > 0.75 else 'orange' if auc > 0.65 else 'red' for auc in aucs]
for bar, color in zip(bars, colors):
   bar.set_color(color)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(models_sorted)
ax1.invert_yaxis()
ax1.set_xlabel('ROC-AUC Score', fontsize=12)
ax1.set_title('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Random Chance')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: All metrics comparison for top 3 models
ax2 = axes[1]
top_3 = results_df.head(3)
metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

for i, (idx, row) in enumerate(top_3.iterrows()):
   values = [row['ROC-AUC'], row['Precision'], row['Recall'], row['F1-Score']]
   ax2.bar(x + i*width, values, width, label=row['Model'], alpha=0.8)

ax2.set_xlabel('Metrics', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Top 3 Models: All Metrics', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.show()

# Identify top performers
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
top_3_models = results_df.head(3)
print(f"\nTop 3 models for hyperparameter tuning:")
for i, (idx, row) in enumerate(top_3_models.iterrows(), 1):
   print(f"  {i}. {row['Model']} (ROC-AUC: {row['ROC-AUC']:.4f})")
