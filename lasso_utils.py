
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# -----------------------------
# Lasso: Calibration
# -----------------------------
def fit_lasso_full_path(Xdf, y, alpha=1e-5):
    """
    Fits a Lasso regression model on the given data (Xdf, y).
    
    Parameters:
    - Xdf: DataFrame containing the features.
    - y: Target variable.
    - alpha: Regularization strength for the Lasso regression.

    Returns:
    - A dictionary containing:
        - "reg": The fitted Lasso regression model.
        - "scaler": The scaler used to standardize the features.
        - "yhat_train": The predicted values on the training data.
        - "mse_train": The mean squared error of the predictions on the training data.
    """
    # Convert input DataFrame to numpy array
    X = Xdf.values
    # Standardize the features (mean=0, std=1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_s = scaler.fit_transform(X)
    # Fit a Lasso regression model
    reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    reg.fit(X_s, y)
    # Predict values using the trained model
    yhat = reg.predict(X_s)
    # Compute the mean squared error on the training data
    mse_train = mean_squared_error(y, yhat)

    return {
        "reg": reg,
        "scaler": scaler,
        "yhat_train": yhat,
        "mse_train": mse_train
    }

def apply_lasso_model(Xdf, y, reg, scaler):
    """
    Applies the trained Lasso regression model to new data (Xdf) to make predictions.
    
    Parameters:
    - Xdf: DataFrame containing the features.
    - y: Target variable.
    - reg: The trained Lasso regression model.
    - scaler: The scaler used to standardize the features before fitting.

    Returns:
    - mse: Mean squared error of the predictions on the test data.
    - yhat: Predicted values on the test data.
    """
    # Convert input DataFrame to numpy array
    X = Xdf.values
    # Standardize the features using the same scaler
    X_s = scaler.transform(X)
    # Make predictions using the fitted model
    yhat = reg.predict(X_s)
    # Compute the mean squared error on the test data
    mse = mean_squared_error(y, yhat)
    return mse, yhat
