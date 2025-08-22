from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
import os
import pandas as pd
# Allow Python to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling_Utils.data_split import split_features_target
from modeling_Utils.scaling import scale_data
from modeling_Utils.train_eval import train_model, evaluate_model, print_metrics
from modeling_Utils.train_eval import test_model
from models.model import run_model_pipeline

from data.Data_helper import load_data





# =================== LOAD DATA ===================
train_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\train_processed_filtered.csv")
val_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\val_processed.csv")
test_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\test_processed.csv")

# =================== MODEL & SCALER ===================
model = LinearRegression(fit_intercept=True)
scaler_type = "minmax"
target_col = "log_trip_duration"


trained_model, scaler, metrics, y_train_pred, y_val_pred = run_model_pipeline(
    train_df=train_df,
    val_df=val_df,
    target_col=target_col,
    model=model,
    scaler_type=scaler_type
)
print_metrics(metrics)

# # Example usage:
if "log_trip_duration" not in test_df.columns:
    test_df["log_trip_duration"] = np.log1p(test_df["trip_duration"])
      # Ensure target column exists
test_df, test_metrics = test_model(trained_model, scaler, test_df, target_col="log_trip_duration")


print("=== TEST METRICS ===")
print(test_metrics)

print("\n=== Test Predictions vs True Values (first 10 rows) ===")
print(test_df[['log_trip_duration', 'pred_log_trip_duration']].head(10))





print(train_df["log_trip_duration"].describe())
print(test_df["log_trip_duration"].describe())
