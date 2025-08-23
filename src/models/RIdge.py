import sys
import os
import pandas as pd
from sklearn.linear_model import Ridge

# Allow Python to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling_Utils.data_split import split_features_target
from modeling_Utils.scaling import scale_data
from modeling_Utils.train_eval_test import train_model, evaluate_model, print_metrics, test_model
from models.model_pipeline import run_model_pipeline
from data_Utils.Data_helper import load_data

# =================== LOAD DATA ===================
train_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\train_processed_filtered.csv")
val_df   = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\val_processed.csv")
test_df  = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\test_processed.csv")

# =================== MODEL PIPELINE ===================
ridge_model = Ridge(alpha=1, fit_intercept=True)
scaler_type = "minmax"
target_col = "log_trip_duration"

trained_model, scaler, metrics, y_train_pred, y_val_pred = run_model_pipeline(
    train_df=train_df,
    val_df=val_df,
    target_col=target_col,
    model=ridge_model,
    scaler_type=scaler_type
)

# =================== PRINT METRICS ===================
print("=== TRAIN & VAL METRICS FOR RIDGE MODEL ===")
print_metrics(metrics)

# =================== TEST ===================
test_df, test_metrics = test_model(trained_model, scaler, test_df, target_col=target_col)
print("=== TEST METRICS FOR RIDGE MODEL ===")
print_metrics(test_metrics)
