import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc)

use_retrained = True  # Set to False to use original best_model
if use_retrained:
   print("Using retrained model (train + val data)")
   test_model = final_model
else:
   print("Using original best model (train data only)")
   test_model = best_model

print(f"Model: {best_model_name}")
print(f"Test samples: {X_test_selected.shape[0]}")

# Make predictions on test set
print("\nMaking predictions on test set...")
y_test_pred = test_model.predict(X_test_selected)
y_test_proba = test_model.predict_proba(X_test_selected)[:, 1]

# Calculate metrics
test_auc = roc_auc_score(y_test, y_test_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print("\n" + "="*60)
print("TEST SET PERFORMANCE")
print("="*60)
print(f"\nROC-AUC:    {test_auc:.4f}")
print(f"Accuracy:   {test_accuracy:.4f}")
print(f"Precision:  {test_precision:.4f}")
print(f"Recall:     {test_recall:.4f}")
print(f"F1-Score:   {test_f1:.4f}")

# Compare to validation
print("\n" + "-"*60)
print("PERFORMANCE COMPARISON")
print("-"*60)
print(f"\nValidation ROC-AUC: {best_val_auc:.4f}")
print(f"Test ROC-AUC:       {test_auc:.4f}")
print(f"Difference:         {test_auc - best_val_auc:+.4f}")

if abs(test_auc - best_val_auc) < 0.05:
   print("✓ Performance is consistent (< 5% difference)")
elif abs(test_auc - best_val_auc) < 0.10:
   print("  Moderate difference (5-10%)")
else:
   print(" Large difference (> 10%) - possible overfitting or dataset shift")

# Detailed classification report
print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
print(classification_report(y_test, y_test_pred, target_names=['Non-pCR', 'pCR']))

# Confusion matrix
print("\n" + "-"*60)
print("CONFUSION MATRIX")
print("-"*60)

cm = confusion_matrix(y_test, y_test_pred)
print(f"\n{cm}")
print(f"\n  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print("\n" + "-"*60)
print("CLINICAL METRICS")
print("-"*60)
print(f"\nSensitivity (Recall):  {sensitivity:.2%}  ({tp}/{tp+fn} pCR patients correctly identified)")
print(f"Specificity:           {specificity:.2%}  ({tn}/{tn+fp} non-pCR patients correctly identified)")
print(f"PPV (Precision):       {ppv:.2%}  ({tp}/{tp+fp} predicted pCR are correct)")
print(f"NPV:                   {npv:.2%}  ({tn}/{tn+fn} predicted non-pCR are correct)")

# Plot ROC curve (optional)
print("\n" + "-"*60)
print("ROC CURVE")
print("-"*60)

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - Test Set\n{best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('test_roc_curve.png', dpi=300, bbox_inches='tight')
print("\n→ ROC curve saved as 'test_roc_curve.png'")
plt.show()

print("\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)
print(f"\nModel: {best_model_name}")
print(f"Features: {X_test_selected.shape[1]} proteins")
print(f"Training approach: {'Combined train+val' if use_retrained else 'Train only'}")
print(f"\nPerformance journey:")
print(f"  Cross-validation (training): {final_comparison.iloc[0]['Best CV ROC-AUC']:.4f}")
print(f"  Validation set:              {best_val_auc:.4f}")
print(f"  Test set (FINAL):            {test_auc:.4f}")
