from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import pandas as pd

def scale_data(X: pd.DataFrame, scaler_type: str = "standard"):
    """
    Apply scaling to a DataFrame based on scaler_type.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features (numeric).
    scaler_type : str
        Type of scaler: ["standard", "minmax", "robust", "maxabs"]
    
    Returns
    -------
    X_scaled : pd.DataFrame
        Scaled dataframe with same column names.
    scaler : fitted scaler object
        Scaler used (can be saved and reused for test data).
    """
    
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler()
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Invalid scaler_type '{scaler_type}'. Choose from {list(scalers.keys())}")
    
    scaler = scalers[scaler_type]
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled, scaler