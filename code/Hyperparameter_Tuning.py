from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Use your actual training data (replace with balanced if you used SMOTE)
X_train_tune = X_train_selected
y_train_tune = y_train

# Define parameter grids for top 3 models
param_grids = {
   'Logistic Regression (L2)': {
       'model': LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
       'params': { 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}},
   'Logistic Regression (L1)': {
       'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
       'params': { 'C': [0.001, 0.01, 0.1, 1, 10, 100]}}

print(f"\nTuning {len(param_grids)} models using GridSearchCV...")
print(f"  Training samples: {X_train_tune.shape[0]}")
print(f"  Features: {X_train_tune.shape[1]}")
print(f"  Evaluation: 5-fold cross-validation")
print(f"  Metric: ROC-AUC")
print("-"*60)

# Store tuned models
tuned_models = {}
tuning_results = []

# Tune each model
for name, config in param_grids.items():
   print(f"\nTuning {name}...")
   # Grid search
   grid_search = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

   # Fit
   grid_search.fit(X_train_tune, y_train_tune)
   # Store results
   tuned_models[name] = grid_search.best_estimator_

   tuning_results.append({ 'Model': name,  'Best CV ROC-AUC': grid_search.best_score_, 'Best Params': str(grid_search.best_params_)})
   print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
   print(f"  Best parameters: {grid_search.best_params_}")

# Create results DataFrame
tuning_df = pd.DataFrame(tuning_results)
tuning_df = tuning_df.sort_values('Best CV ROC-AUC', ascending=False)

print("\n" + "="*60)
print("TUNING RESULTS")
print("="*60)
print(tuning_df.to_string(index=False))

# Evaluate tuned models on validation set
print("\n" + "="*60)
print("VALIDATION SET EVALUATION")
print("="*60)

val_results = []
for name, model in tuned_models.items():
   # Predict on validation set
   y_val_pred = model.predict(X_val_selected)
   y_val_proba = model.predict_proba(X_val_selected)[:, 1]
   # Calculate metrics
   val_auc = roc_auc_score(y_val, y_val_proba)
   val_results.append({'Model': name,'Validation ROC-AUC': val_auc})
   print(f"\n{name}:")
   print(f"  Validation ROC-AUC: {val_auc:.4f}")

# Final comparison
val_df = pd.DataFrame(val_results)
val_df = val_df.sort_values('Validation ROC-AUC', ascending=False)

# Merge with tuning results
final_comparison = tuning_df.merge(val_df, on='Model')
final_comparison['Performance Drop'] = final_comparison['Best CV ROC-AUC'] - final_comparison['Validation ROC-AUC']

print("\n" + "="*60)
print("FINAL COMPARISON: CV vs Validation")
print("="*60)
print(final_comparison[['Model', 'Best CV ROC-AUC', 'Validation ROC-AUC', 'Performance Drop']].to_string(index=False))

# Select best model
best_model_name = final_comparison.iloc[0]['Model']
best_model = tuned_models[best_model_name]
best_val_auc = final_comparison.iloc[0]['Validation ROC-AUC']

print("\n" + "="*60)
print("FINAL MODEL SELECTION")
print("="*60)
print(f"\nSelected model: {best_model_name}")
print(f"Validation ROC-AUC: {best_val_auc:.4f}")

# Detailed evaluation of best model
print("\n" + "-"*60)
print(f"Detailed Validation Results for {best_model_name}")
print("-"*60)

y_val_pred_best = best_model.predict(X_val_selected)
y_val_proba_best = best_model.predict_proba(X_val_selected)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_best, target_names=['RCB 2/3', 'RCB 0/1']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred_best)
print(cm)
print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
