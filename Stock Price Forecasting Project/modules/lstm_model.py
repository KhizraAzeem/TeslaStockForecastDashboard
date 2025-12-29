import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def run_lstm(input_data, n_steps=60, epochs=30, batch_size=32):
    """
    Streamlit-ready LSTM pipeline.

    Returns:
        dict: {
            'y_test': actual prices,
            'y_pred': predicted prices,
            'mae': MAE,
            'rmse': RMSE,
            'history': training history,
            'fig': matplotlib figure
        }
    """
    # ---------- Load Data ----------
    if hasattr(input_data, "read"):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, n_steps)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---------- Model ----------
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test)
    # ---------- Rescale ----------
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), df.shape[1]-1))], axis=1)
    )[:,0]
    y_pred_rescaled = scaler.inverse_transform(
        np.concatenate([y_pred, np.zeros((len(y_pred), df.shape[1]-1))], axis=1)
    )[:,0]

    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(y_test_rescaled, label="Actual Price")
    ax.plot(y_pred_rescaled, label="Predicted Price")
    ax.set_title("LSTM Stock Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    return {
        'y_test': y_test_rescaled,
        'y_pred': y_pred_rescaled,
        'mae': mae,
        'rmse': rmse,
        'history': history,
        'fig': fig
    }



