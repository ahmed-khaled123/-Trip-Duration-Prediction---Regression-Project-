import sys
import os
import pandas as pd
from xgboost import XGBRegressor

# السماح للبايثون يوصل للمجلدات utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling_Utils.data_split import split_features_target
from modeling_Utils.scaling import scale_data
from modeling_Utils.train_eval_test import train_model, evaluate_model, print_metrics, test_model
from models.model_pipeline import run_model_pipeline
from data.Data_helper import load_data

# =================== LOAD DATA ===================
train_df = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\train_processed_filtered.csv")
val_df   = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\val_processed.csv")
test_df  = load_data(r"G:\ML mostafa saad\slides\my work\13 Project 1 - Regression - Trip Duration Prediction\data\processed\test_processed.csv")

# =================== MODEL PIPELINE ===================
xgb_model = XGBRegressor(
    n_estimators=500,      # عدد الأشجار
    learning_rate=0.05,    # معدل التعلم
    max_depth=6,            # أقصى عمق للشجرة
    subsample=0.8,          # أخذ عينات من البيانات لكل شجرة
    colsample_bytree=0.8,   # أخذ عينات من الأعمدة لكل شجرة
    random_state=42,
    n_jobs=-1
)

scaler_type = "minmax"  # مش ضروري مع XGBoost، بس لو البايبلاين محتاجه نخليه زي البقية
target_col = "log_trip_duration"

trained_model, scaler, metrics, y_train_pred, y_val_pred = run_model_pipeline(
    train_df=train_df,
    val_df=val_df,
    target_col=target_col,
    model=xgb_model,
    scaler_type=scaler_type
)

# =================== PRINT METRICS ===================
print("=== TRAIN & VAL METRICS FOR XGBOOST MODEL ===")
print_metrics(metrics)

# =================== TEST ===================
test_df, test_metrics = test_model(trained_model, scaler, test_df, target_col=target_col)
print("=== TEST METRICS FOR XGBOOST MODEL ===")
print_metrics(test_metrics)
