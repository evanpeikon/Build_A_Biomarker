def scale_proteomics_data(X_train, X_val, X_test, log_transformed=True):
   from sklearn.preprocessing import StandardScaler
   import numpy as np
  
   # Step 1: log transformation if needed
   if not log_transformed:
       X_train_log = np.log2(X_train + 1)
       X_val_log = np.log2(X_val + 1)
       X_test_log = np.log2(X_test + 1)
      
       # Preserve DataFrame structure
       X_train_log = pd.DataFrame(X_train_log, index=X_train.index, columns=X_train.columns)
       X_val_log = pd.DataFrame(X_val_log, index=X_val.index, columns=X_val.columns)
       X_test_log = pd.DataFrame(X_test_log, index=X_test.index, columns=X_test.columns) 
   else:
       X_train_log = X_train.copy()
       X_val_log = X_val.copy()
       X_test_log = X_test.copy()
  
   # Step 2: standardization (fit on training data only)
   print("\nApplying standardization (z-score)...")
   print("  Fitting scaler on training data only...")
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train_log)
   print("  Applying scaler to validation data...")
   X_val_scaled = scaler.transform(X_val_log)
   print("  Applying scaler to test data...")
   X_test_scaled = scaler.transform(X_test_log)
  
   # Convert back to DataFrames
   X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_log.index, columns=X_train_log.columns)
   X_val_scaled = pd.DataFrame(X_val_scaled,index=X_val_log.index,columns=X_val_log.columns)
   X_test_scaled = pd.DataFrame(X_test_scaled,index=X_test_log.index, columns=X_test_log.columns)
  
   # Verification
   print(f"\nTraining set (should have mean≈0, std≈1):")
   print(f"  Mean: {X_train_scaled.mean().mean():.6f}")
   print(f"  Std:  {X_train_scaled.std().mean():.6f}")
   print(f"\nValidation set (will differ - uses train parameters):")
   print(f"  Mean: {X_val_scaled.mean().mean():.6f}")
   print(f"  Std:  {X_val_scaled.std().mean():.6f}")
   print(f"\nTest set (will differ - uses train parameters):")
   print(f"  Mean: {X_test_scaled.mean().mean():.6f}")
   print(f"  Std:  {X_test_scaled.std().mean():.6f}")
   return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# Usage example:
X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_proteomics_data(x_train, x_val, x_test, log_transformed=True)
