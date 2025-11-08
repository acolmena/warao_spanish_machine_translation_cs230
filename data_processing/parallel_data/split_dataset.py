import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(data, test_size=0.30, random_state=42):
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    training_data, temp_data = train_test_split(
        data,
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    # val_size_adjusted = val_size / (1 - test_size)
    
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5, 
        random_state=random_state, 
        shuffle=True
    )

    print(f"Train set: {len(training_data)} samples ({len(training_data)/len(data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    return training_data, val_data, test_data



if __name__ == "__main__":   
    input_filename = 'parallel_data_all.csv'
    input_path = os.path.join("output", input_filename)
    df = pd.read_csv(input_path) 
    print(f"Source set: {len(df)} samples")
    training_data, val_data, test_data = split_dataset(
        # X=df['warao_sentence'], 
        # y=df['spanish_sentence'],
        data=df,
        test_size=0.30,  
        random_state=42
    )
    

    
    # Example 4: Save splits to disk
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Saving Splits")
    print("=" * 50)
    
    
    # Save data as CSVs
    if isinstance(training_data, pd.DataFrame):
        training_data.to_csv(os.path.join("output", 'parallel_train.csv'), index=False)
        val_data.to_csv(os.path.join("output", 'parallel_val.csv'), index=False)
        test_data.to_csv(os.path.join("output", 'parallel_test.csv'), index=False)
    
    print("Splits saved successfully!")