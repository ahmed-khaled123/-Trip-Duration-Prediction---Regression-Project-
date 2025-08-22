import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def train_model(model, X_train, y_train):
    """
    Train any sklearn-like model.
    
    Returns:
        model : trained model
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_val, y_val, selected_metrics=None):
    """
    Evaluate a trained model on train and validation sets and return metrics + predictions.
    
    Parameters:
        model : trained sklearn estimator
        X_train, y_train : training data
        X_val, y_val : validation data
        selected_metrics : list of str, optional
            Metrics to calculate. Options: ["MAE", "RMSE", "R2"].
            If None, calculates all.
            
    Returns:
        metrics : dict of train/val metrics
        y_pred_train : predictions on training set
        y_pred_val : predictions on validation set
    """
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Define all metrics
    all_metrics = {
        "MAE": mean_absolute_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score
    }

    if selected_metrics is None:
        selected_metrics = all_metrics.keys()
    
    metrics = {"train": {}, "val": {}}
    for m in selected_metrics:
        if m not in all_metrics:
            raise ValueError(f"Invalid metric '{m}'. Choose from {list(all_metrics.keys())}")
        func = all_metrics[m]
        metrics["train"][m] = func(y_train, y_pred_train)
        metrics["val"][m] = func(y_val, y_pred_val)

    return metrics, y_pred_train, y_pred_val



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def test_model(model, scaler, test_df, target_col="log_trip_duration"):
    """
    Evaluate a trained model on the test set.
    """
    # Prepare test features
    X_test = test_df.copy()
    X_test = X_test[model.feature_names_in_]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Predict log values
    y_test_pred_log = model.predict(X_test_scaled)
    y_test_pred_log = np.clip(y_test_pred_log, a_min=None, a_max=15)

    if "log_trip_duration" not in test_df.columns:
        test_df["log_trip_duration"] = np.log1p(test_df["trip_duration"])
    # ضيف predictions
    test_df['pred_log_trip_duration'] = y_test_pred_log   # الاسم يبقى واضح

    # لو العمود target_col مش موجود مش هتحسب metrics
    metrics = {}
    if target_col in test_df.columns:
        y_true = test_df[target_col]
        mae = mean_absolute_error(y_true, y_test_pred_log)
        rmse = np.sqrt(mean_squared_error(y_true, y_test_pred_log))
        r2 = r2_score(y_true, y_test_pred_log)
        metrics = {"test": {"MAE": mae, "RMSE": rmse, "R2": r2}}

    return test_df, metrics



def print_metrics(metrics):
    """ Print metrics in a readable format. """
    for split in metrics:
        print(f"{split.capitalize()} Metrics:")
        for metric, value in metrics[split].items():
            print(f"  {metric}: {value:.4f}")
        print()     