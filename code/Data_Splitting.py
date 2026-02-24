import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_test_split(data, output_name='', test_size_1=0.20, test_size_2=0.20):
 # Separate x (protein columns) and y (outcome column)
 x = data.drop(output_name, axis=1)
 y = data[output_name]

 # Shuffle rows (patients) in case data is ordered by subtype
 np.random.seed(13)
 shuffled_indices = np.random.permutation(x.index)
 x_shuffled = x.loc[shuffled_indices]
 y_shuffled = y.loc[shuffled_indices]

 # Split 1: Separate test set (20%)
 x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(
   x_shuffled, y_shuffled, test_size=test_size_1, stratify=y_shuffled, random_state=13)

 # Split 2: Split x_train_and_val into training (64%) and validation (16%)
 x_train, x_val, y_train, y_val = train_test_split(
   x_train_and_val, y_train_and_val, test_size=test_size_2, stratify=y_train_and_val, random_state=13)
 # Summary
 print(f"\nTraining set:")
 print(f"  X_train shape: {x_train.shape}")
 print(f"  y_train shape: {y_train.shape}")
 print(f"  Proportion of total: {len(x_train)/len(x)*100:.1f}%")
 print(f"  Class distribution: {y_train.value_counts().to_dict()}")
 print(f"\nValidation set:")
 print(f"  X_val shape: {x_val.shape}")
 print(f"  y_val shape: {y_val.shape}")
 print(f"  Proportion of total: {len(x_val)/len(x)*100:.1f}%")
 print(f"  Class distribution: {y_val.value_counts().to_dict()}")
 print(f"\nTest set:")
 print(f"  X_test shape: {x_test.shape}")
 print(f"  y_test shape: {y_test.shape}")
 print(f"  Proportion of total: {len(x_test)/len(x)*100:.1f}%")
 print(f"  Class distribution: {y_test.value_counts().to_dict()}")
 return x_train, x_val, y_train, y_val, x_test, y_test

# Usage example 
x_train, x_val, y_train, y_val, x_test, y_test= train_val_test_split(proteomic_data_clean, output_name='pCR', test_size_1=0.20, test_size_2=0.20)
