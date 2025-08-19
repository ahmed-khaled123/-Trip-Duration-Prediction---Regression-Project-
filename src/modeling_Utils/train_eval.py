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

def print_metrics(metrics):
    """ Print metrics in a readable format. """
    for split in metrics:
        print(f"{split.capitalize()} Metrics:")
        for metric, value in metrics[split].items():
            print(f"  {metric}: {value:.4f}")
        print()     