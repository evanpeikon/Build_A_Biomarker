from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectFromModel

def select_features_lasso(X_train, X_val, X_test, y_train, C=1.0, task='classification', n_display=20):
   print(f"\nTask: {task}")
   print(f"Starting features: {X_train.shape[1]}")
  
   # Select appropriate LASSO model
   if task == 'classification':
       print(f"Regularization parameter C: {C}")
       lasso_model = LogisticRegression( penalty='l1', solver='liblinear', C=C, random_state=13, max_iter=1000)
       lasso_model.fit(X_train, y_train)
       coefficients = lasso_model.coef_[0]
      
   elif task == 'regression':
       print("Using LassoCV with cross-validation for alpha selection")
       lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
       lasso_model.fit(X_train, y_train)
       print(f"Optimal alpha: {lasso_model.alpha_:.6f}")
       coefficients = lasso_model.coef_
   else:
       raise ValueError("task must be 'classification' or 'regression'")
  
   # Get selected features (non-zero coefficients)
   selected_mask = coefficients != 0
   selected_features = X_train.columns[selected_mask].tolist()
   print(f"\nFeatures selected: {len(selected_features)}")
  
   # Fallback if no features selected
   if len(selected_features) == 0:
       print("\n No features selected with strict threshold")
       print("→ Using SelectFromModel with 'median' threshold")
       selector = SelectFromModel(lasso_model, threshold='median', prefit=True)
       selected_mask = selector.get_support()
       selected_features = X_train.columns[selected_mask].tolist()
       print(f"Features selected: {len(selected_features)}")
  
   # Apply selection to all sets
   X_train_selected = X_train[selected_features]
   X_val_selected = X_val[selected_features]
   X_test_selected = X_test[selected_features]
  
   # Display top features
   print(f"\n{'='*60}")
   print(f"Top {min(n_display, len(selected_features))} Selected Features")
   print(f"{'='*60}")
  
   # Get feature importances (absolute coefficients)
   feature_coefs = [(feat, coefficients[X_train.columns.get_loc(feat)])
                    for feat in selected_features]
   feature_coefs_sorted = sorted(feature_coefs, key=lambda x: abs(x[1]), reverse=True)
   print(f"\n{'Feature':<40} {'Coefficient':>12} {'Direction':>10}")
   print("-"*60)
   for i, (feature, coef) in enumerate(feature_coefs_sorted[:n_display], 1):
       direction = "↑ (pos)" if coef > 0 else "↓ (neg)"
       print(f"{i:2d}. {feature:<35} {coef:>12.4f} {direction:>10}")
   if len(selected_features) > n_display:
       print(f"\n... and {len(selected_features) - n_display} more features")
  
   # Summary
   reduction_pct = (1 - len(selected_features)/X_train.shape[1]) * 100
   print(f"\nReduction: {reduction_pct:.1f}% ({X_train.shape[1]} → {len(selected_features)} features)")
   return X_train_selected, X_val_selected, X_test_selected, selected_features, lasso_model

# Usage example:
X_train_selected, X_val_selected, X_test_selected, selected_features, lasso_model = select_features_lasso(X_train_scaled, X_val_scaled,  X_test_scaled, y_train, C=0.2, task='classification', n_display=10
