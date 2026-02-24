def clean_proteomics_data(data, output_name, sample_threshold=0.40, protein_threshold=0.20, log_transformed=True):
   # Remove samples with missing outcome
   n_start = len(data)
   data_clean = data.dropna(subset=[output_name])
   n_removed_outcome = n_start - len(data_clean)
   if n_removed_outcome > 0:
       print(f"\nRemoved {n_removed_outcome} samples with missing outcome")
  
   # Separate x and y for processing
   X = data_clean.drop(output_name, axis=1)
   y = data_clean[output_name]
   print(f"\nStarting: {X.shape[0]} samples × {X.shape[1]} proteins")
   print(f"Total missing: {X.isna().sum().sum()} values")
  
   # Stage 1: remove samples with high missingness
   sample_missing_pct = X.isna().mean(axis=1)
   sufficient_samples = sample_missing_pct <= sample_threshold
   n_removed_samples = (~sufficient_samples).sum()
   X_stage1 = X[sufficient_samples]
   y_stage1 = y[sufficient_samples]
   print(f"\nStage 1: Removed {n_removed_samples} samples (>{sample_threshold*100:.0f}% missing)")
  
   # Stage 2: remove proteins with high missingness
   protein_missing_pct = X_stage1.isna().mean(axis=0)
   reliable_proteins = protein_missing_pct <= protein_threshold
   n_removed_proteins = (~reliable_proteins).sum()
   X_stage2 = X_stage1.loc[:, reliable_proteins]
   print(f"Stage 2: Removed {n_removed_proteins} proteins (>{protein_threshold*100:.0f}% missing)")

   # Stage 3: impute remaining missing values
   n_to_impute = X_stage2.isna().sum().sum()
   if n_to_impute > 0:
       X_imputed = X_stage2.copy()
       for protein in X_imputed.columns:
           if X_imputed[protein].isna().any():
               min_val = X_imputed[protein].min()
               if log_transformed:
                   impute_val = min_val - 1.0
               else:
                   impute_val = 0.1 * min_val
               X_imputed[protein].fillna(impute_val, inplace=True)
       impute_method = "min - 1.0 (log2)" if log_transformed else "0.1 × min"
       print(f"Stage 3: Imputed {n_to_impute} values ({impute_method})")
   else:
       X_imputed = X_stage2.copy()
       print(f"Stage 3: No imputation needed")
  
   # Reconstruct full dataframe
   data_final = X_imputed.copy()
   data_final[output_name] = y_stage1
  
   # Summary
   print(f"\n{'='*60}")
   print(f"Final: {data_final.shape[0]} samples × {X_imputed.shape[1]} proteins")
   print(f"Missing values: {X_imputed.isna().sum().sum()}")
   return data_final

# Usage example
proteomic_data_clean = clean_proteomics_data(proteomic_data, output_name='pCR', sample_threshold=0.40, protein_threshold=0.20, log_transformed=True)
