# modules/ann_model_streamlit.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

def run_ann(df, epochs=50, batch_size=32):
    """
    Train and evaluate ANN model on stock dataset to predict 'Close'.
    Streamlit-ready: returns figures and metrics instead of plt.show().
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        dict: {
            'model': trained ANN model,
            'y_test': actual values,
            'y_pred': predicted values,
            'mae': mean absolute error,
            'rmse': root mean squared error,
            'fig_loss': matplotlib figure of training loss
        }
    """

    # Drop Date if exists
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # Features and target
    X = df.drop(columns=['Close'])
    y = df['Close']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build ANN Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train Model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Predict & Evaluate
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Plot training vs validation loss
    fig_loss = plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("ANN Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    # Return everything for Streamlit display
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'fig_loss': fig_loss
    }

# ================================
# Run directly for testing
# ================================
if __name__ == "__main__":
    df = pd.read_csv("data/TeslaStock_Dataset_Cleaned.csv")
    results = run_ann(df)
    print(f"MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    results['fig_loss'].show()
