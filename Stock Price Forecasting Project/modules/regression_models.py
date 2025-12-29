import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_regression_streamlit(input_data, features=None, target="Close", test_size=0.2):
    """
    Streamlit-ready Linear Regression pipeline.

    Returns:
        dict: {
            'model': trained model,
            'X_test': test features,
            'y_test': test target,
            'y_pred': predictions,
            'mae': MAE,
            'rmse': RMSE,
            'fig': matplotlib figure
        }
    """
    # ---------- Load Data ----------
    if hasattr(input_data, "read"):  # Streamlit file uploader
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # ---------- Features ----------
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
        if target in features:
            features.remove(target)

    X = df[features]
    y = df[target]

    # ---------- Train/Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ---------- Model ----------
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------- Metrics ----------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(len(y_test)), y_test, label="Actual")
    ax.plot(range(len(y_pred)), y_pred, label="Predicted")
    ax.set_title("Linear Regression: Actual vs Predicted")
    ax.set_xlabel("Test Samples")
    ax.set_ylabel(target)
    ax.legend()
    plt.tight_layout()

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'fig': fig
    }
