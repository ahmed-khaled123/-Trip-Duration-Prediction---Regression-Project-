import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Allow Python to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling_Utils.data_split import split_features_target
from modeling_Utils.scaling import scale_data
from modeling_Utils.train_eval import train_model, evaluate_model, print_metrics
from modeling_Utils.train_eval import test_model

from data.Data_helper import load_data

# =================== LOAD DATA ===================
train_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\train_processed_filtered.csv")
val_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\val_processed.csv")
test_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\test_processed.csv")

# =================== MODEL & SCALER ===================
model = LinearRegression(fit_intercept=True)
scaler_type = "minmax"
target_col = "log_trip_duration_min"

# =================== TRAIN + VAL PIPELINE ===================
def run_model_pipeline(train_df, val_df, target_col, model, scaler_type="minmax", selected_metrics=None):
    """
    Full pipeline: split, scale, train, evaluate.
    Returns trained model, metrics dict, y_train_pred, y_val_pred
    """
    # 1. Split features/target
    X_train, y_train, X_val, y_val = split_features_target(train_df, val_df, target_col)
    
    # 2. Scale features
    X_train_scaled, scaler = scale_data(X_train, scaler_type=scaler_type)
    X_val = X_val[X_train.columns]
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    
    # 3. Train model
    trained_model = train_model(model, X_train_scaled, y_train)
    
    # 4. Evaluate
    metrics, y_train_pred, y_val_pred = evaluate_model(trained_model, X_train_scaled, y_train, X_val_scaled, y_val, selected_metrics)
    
    return trained_model, scaler, metrics, y_train_pred, y_val_pred

trained_model, scaler, metrics, y_train_pred, y_val_pred = run_model_pipeline(
    train_df=train_df,
    val_df=val_df,
    target_col=target_col,
    model=model,
    scaler_type=scaler_type
)
print_metrics(metrics)

# # Example usage:

test_df, test_metrics = test_model(trained_model, scaler, test_df, target_col="trip_duration_min")

print("=== TEST METRICS ===")
if test_metrics:
    for metric, value in test_metrics['test'].items():
        print(f"  {metric}: {value:.4f}")

print("\n=== Test Predictions vs True Values (first 10 rows) ===")
print(test_df[['trip_duration_min', 'pred_trip_duration_min']].head(20).max ())






