import pandas as pd
import os

def load_data(filepath):
    """
    Load a CSV file given its full path.
    
    Parameters:
        filepath (str): Full path to the CSV file (train/val/test, raw or processed)
    
    Returns:
        pd.DataFrame: loaded dataframe
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df



def validate_dataset(df):
    """
    Run basic checks on the dataset:
    - missing values
    - coordinate ranges
    - column types
    """
    # Columns we expect
    expected_cols = ['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime',
                     'passenger_count', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration']
    
    # Check columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Check coordinate ranges (example for NYC)
    print("Pickup Longitude:", df['pickup_longitude'].min(), df['pickup_longitude'].max())
    print("Pickup Latitude:", df['pickup_latitude'].min(), df['pickup_latitude'].max())
    print("Dropoff Longitude:", df['dropoff_longitude'].min(), df['dropoff_longitude'].max())
    print("Dropoff Latitude:", df['dropoff_latitude'].min(), df['dropoff_latitude'].max())
    
    # Column types
    print("Column types:")
    print(df.dtypes)
    
    return True



def save_processed_data(df, path):
    """
    Save the processed DataFrame to a given path.
    
    Args:
        df (pd.DataFrame): Data to save
        path (str): Full path including filename (e.g. 'data/processed/train_processed.csv')
    """
    import os
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=False)
    print(f" Processed data saved to {path}")

def inspect_data(train_df, val_df, target_col):
    print("=== TRAIN DATA ===")
    print(train_df.head())
    print(train_df.info())
    print(train_df.describe(include='all'))
    print(f"{target_col} range: min={train_df[target_col].min()}, max={train_df[target_col].max()}\n")
    
    print("=== VAL DATA ===")
    print(val_df.head())
    print(val_df.info())
    print(val_df.describe(include='all'))
    if target_col in val_df.columns:
        print(f"{target_col} range: min={val_df[target_col].min()}, max={val_df[target_col].max()}\n")
    else:
        print(f"{target_col} not in validation data.\n")

