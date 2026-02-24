# Combine train and validation
X_train_val = pd.concat([X_train_selected, X_val_selected], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

print(f"  Training samples: {X_train_selected.shape[0]}")
print(f"  Validation samples: {X_val_selected.shape[0]}")
print(f"  Combined samples: {X_train_val.shape[0]}")

# Get best hyperparameters from Step 9
print(f"\nRetraining {best_model_name} with optimal hyperparameters...")
print(f"  Best parameters: {final_comparison.iloc[0]['Best Params']}")

# Recreate model with best parameters
if best_model_name == 'SVM (RBF)':
   # Extract best parameters (assuming SVM won)
   final_model = SVC(C=100, gamma=0.001, kernel='rbf', probability=True, random_state=42)
elif best_model_name == 'Logistic Regression (L2)':
   final_model = LogisticRegression(C=0.1, solver='lbfgs', penalty='l2', max_iter=1000, random_state=42)
elif best_model_name == 'Logistic Regression (L1)':
   final_model = LogisticRegression(C=1, solver='liblinear', penalty='l1', max_iter=1000, random_state=42)

# Fit on combined data
final_model.fit(X_train_val, y_train_val)

print(f"\n✓ Model retrained on {X_train_val.shape[0]} samples")
