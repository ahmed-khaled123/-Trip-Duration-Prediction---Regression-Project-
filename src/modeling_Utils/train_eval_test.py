import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(model, X_train, y_train):
    """
    Train any sklearn-like model.

    Parameters:
        model : sklearn estimator
        X_train : pd.DataFrame or np.ndarray
        y_train : pd.Series or np.ndarray

    Returns:
        model : trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, selected_metrics=None):
    """
    Evaluate a trained model on train and validation sets.

    Parameters:
        model : trained sklearn estimator
        X_train, y_train : training data
        X_val, y_val : validation data
        selected_metrics : list of str, optional
            Metrics to calculate. Options: ["MAE", "RMSE", "R2"].
            If None, calculates all.

    Returns:
        metrics : dict with train/val metrics
        y_pred_train : np.ndarray
        y_pred_val : np.ndarray
    """
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Define metrics
    all_metrics = {
        "MAE": mean_absolute_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score,
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


def test_model(model, scaler, test_df, target_col="log_trip_duration", clip_max=15):
    """
    Evaluate a trained model on the test set.

    Parameters:
        model : trained sklearn estimator
        scaler : fitted scaler
        test_df : pd.DataFrame
        target_col : str (default: "log_trip_duration")
        clip_max : float (default: 15) -> upper limit for clipping logs

    Returns:
        test_df : DataFrame with predictions
        metrics : dict with test metrics
    """
    # Prepare test features
    X_test = test_df[model.feature_names_in_]
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Predict log values
    y_test_pred_log = np.clip(model.predict(X_test_scaled), a_min=None, a_max=clip_max)

    # Ensure target exists
    if "log_trip_duration" not in test_df.columns:
        test_df["log_trip_duration"] = np.log1p(test_df["trip_duration"])

    # Clip target values for fair comparison
    test_df["log_trip_duration"] = np.clip(test_df["log_trip_duration"], a_min=None, a_max=clip_max)

    # Save predictions
    test_df["pred_log_trip_duration"] = y_test_pred_log

    # Compute metrics
    metrics = {}
    if target_col in test_df.columns:
        y_true = test_df[target_col]
        metrics = {
            "test": {
                "MAE": mean_absolute_error(y_true, y_test_pred_log),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_test_pred_log)),
                "R2": r2_score(y_true, y_test_pred_log),
            }
        }

    return test_df, metrics


def print_metrics(metrics):
    """Pretty-print metrics dict."""
    for split, vals in metrics.items():
        print(f"{split.capitalize()} Metrics:")
        for metric, value in vals.items():
            print(f"  {metric}: {value:.4f}")
        print()
