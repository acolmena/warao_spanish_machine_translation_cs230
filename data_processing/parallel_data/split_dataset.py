import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """ 
    Split dataset into train/validation/test sets.
    
    Args:
        X: Warao sentences 
        y: Spanish translations 
        test_size: percentage of data used for test set, 15% 
        val_size: percentage of data used for validation set, 15% 
        random_state: Random seed for reproducibility, so we always get the same split
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Source set: {len(X)} samples")
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == "__main__":   
    input_filename = 'parallel_data_all.csv'
    input_path = os.path.join("output", input_filename)
    df = pd.read_csv(input_path) 
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X=df['warao_sentences'], 
        y=df['spanish_sentences'], 
        test_size=0.15, 
        val_size=0.15, 
        random_state=42
    )
    

    
    # Example 4: Save splits to disk
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Saving Splits")
    print("=" * 50)
    
    
    # Save data as CSVs
    if isinstance(X_train, pd.DataFrame):
        pd.concat([X_train, y_train], axis=1).to_csv('parallel_train.csv', index=False)
        pd.concat([X_val, y_val], axis=1).to_csv('parallel_val.csv', index=False)
        pd.concat([X_test, y_test], axis=1).to_csv('parallel_test.csv', index=False)
    
    print("Splits saved successfully!")