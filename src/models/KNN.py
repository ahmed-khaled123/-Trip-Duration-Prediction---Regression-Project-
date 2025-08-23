import sys
import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Allow Python to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling_Utils.train_eval_test import  print_metrics, test_model
from models.model_pipeline import run_model_pipeline
from data.Data_helper import load_data

# =================== LOAD DATA ===================
train_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\train_processed_filtered.csv")
val_df   = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\val_processed.csv")
test_df  = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\test_processed.csv")

# =================== MODEL PIPELINE ===================
knn_model = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    p=2  # Euclidean
)
scaler_type = "minmax"  # very important for KNN
target_col = "log_trip_duration"

trained_model, scaler, metrics, y_train_pred, y_val_pred = run_model_pipeline(
    train_df=train_df,
    val_df=val_df,
    target_col=target_col,
    model=knn_model,
    scaler_type=scaler_type
)

# =================== PRINT METRICS ===================
print("=== TRAIN & VAL METRICS FOR KNN MODEL ===")
print_metrics(metrics)

# =================== TEST ===================
test_df, test_metrics = test_model(trained_model, scaler, test_df, target_col=target_col)
print("=== TEST METRICS FOR KNN MODEL ===")
print_metrics(test_metrics)
