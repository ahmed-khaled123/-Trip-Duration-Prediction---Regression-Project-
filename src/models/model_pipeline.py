import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# Allow Python to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling_Utils.data_split import split_features_target
from modeling_Utils.scaling import scale_data
from modeling_Utils.train_eval_test import train_model, evaluate_model


# =================== MODEL & SCALER ===================
DEFAULT_MODEL = LinearRegression(fit_intercept=True)
DEFAULT_SCALER_TYPE = "minmax"
DEFAULT_TARGET_COL = "log_trip_duration"


def run_model_pipeline(train_df, val_df, target_col=DEFAULT_TARGET_COL, 
                       model=DEFAULT_MODEL, scaler_type=DEFAULT_SCALER_TYPE, 
                       selected_metrics=None):
    """
    Full pipeline: split, scale, train, evaluate.
    
    Parameters:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        target_col : str
        model : sklearn estimator (default: LinearRegression)
        scaler_type : str (default: "minmax")
        selected_metrics : list of str (optional)

    Returns:
        dict containing:
            - model : trained model
            - scaler : fitted scaler
            - metrics : dict of train/val metrics
            - y_train_pred : np.ndarray
            - y_val_pred : np.ndarray
    """
    # 1. Split features/target
    X_train, y_train, X_val, y_val = split_features_target(train_df, val_df, target_col)
    
    # 2. Scale features
    X_train_scaled, scaler = scale_data(X_train, scaler_type=scaler_type)
    X_val = X_val[X_train.columns]  # ensure same columns order
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    
    # 3. Train model
    trained_model = train_model(model, X_train_scaled, y_train)
    
    # 4. Evaluate
    metrics, y_train_pred, y_val_pred = evaluate_model(
        trained_model, X_train_scaled, y_train, X_val_scaled, y_val, selected_metrics
    )
    
    return  trained_model, scaler, metrics, y_train_pred, y_val_pred



