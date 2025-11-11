import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.30
VAL_RATIO = 0.5

def split_dataset(data, holdout_size=0.30, val_ratio=0.5, random_state=42):
    """ 
    Split dataset into train/validation/test sets.
    
    Args:
        data: Warao-Spanish parallel data 
        holdout_size: percentage of data used for holdout (includes test and validation), 15% 
        val_ratio: percentage of holdout data used for validation, 50% 
        random_state: random seed for reproducibility, so we always get the same split
    
    Returns:
        training_data, val_data, test_data
    """    
    training_data, temp_data = train_test_split(
        data,
        test_size=holdout_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    
    val_data, test_data = train_test_split(
        temp_data,
        test_size=val_ratio, 
        random_state=random_state, 
        shuffle=True
    )

    print("\n" + "=" * 50)
    print(f"Train set: {len(training_data)} samples ({len(training_data)/len(data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    print("=" * 50)

    return training_data, val_data, test_data



if __name__ == "__main__":   
    input_filename = 'parallel_data_all.csv'
    input_path = os.path.join("input", input_filename)
    df = pd.read_csv(input_path) 

    print("\n" + "=" * 50)
    print(f"Source set: {len(df)} samples")
    print(f"Preview: {df.head(7)}")
    print("=" * 50)
    training_data, val_data, test_data = split_dataset(
        data=df,
        holdout_size=HOLDOUT_SIZE, 
        val_ratio=VAL_RATIO, 
        random_state=RANDOM_STATE
    )
    

    print("\n" + "=" * 50)
    print("Saving splits . . .")
    print("=" * 50)

    os.makedirs("output", exist_ok=True)
    
    # save split data to CSVs
    if isinstance(training_data, pd.DataFrame):
        training_data.to_csv(os.path.join("output", 'parallel_train.csv'), index=False)
        val_data.to_csv(os.path.join("output", 'parallel_val.csv'), index=False)
        test_data.to_csv(os.path.join("output", 'parallel_test.csv'), index=False)
    
    print("\nSplits saved successfully!")