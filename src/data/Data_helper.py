import pandas as pd
import os

def load_raw_data(filename="train.csv"):
    """
    Load raw CSV from data/raw/
    """
    filepath = os.path.join("data", "raw", filename)
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



def save_processed_data(df, filename="train_processed.csv"):
    """
    Save processed dataframe to data/processed/ without touching raw data
    """
    os.makedirs("data/processed", exist_ok=True)
    filepath = os.path.join("data", "processed", filename)
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")