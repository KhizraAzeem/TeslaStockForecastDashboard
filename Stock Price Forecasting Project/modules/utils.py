# modules/utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# ==========================
# 1. Plotting Functions
# ==========================
def plot_line(x, y, title="", xlabel="", ylabel="", color="blue", return_fig=False):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, y, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if return_fig:
        return fig
    else:
        plt.show()

def plot_scatter(actual, predicted, title="Actual vs Predicted", xlabel="Actual", ylabel="Predicted", return_fig=False):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(actual, predicted, alpha=0.6, color='blue', label="Predicted vs Actual")
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label="Perfect Prediction")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if return_fig:
        return fig
    else:
        plt.show()

def plot_histogram(data, bins=30, color="purple", title="", xlabel="", ylabel="Frequency", return_fig=False):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(data, bins=bins, kde=True, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if return_fig:
        return fig
    else:
        plt.show()

def plot_heatmap(df, title="Correlation Heatmap", return_fig=False):
    numeric_df = df.select_dtypes(include='number')  # Only numeric columns
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    if return_fig:
        return fig
    else:
        plt.show()

# ==========================
# 2. Scaling Functions
# ==========================
def standard_scale(train_data, test_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)
    X_test_scaled = scaler.transform(test_data)
    return X_train_scaled, X_test_scaled, scaler

def minmax_scale(data, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# ==========================
# 3. Evaluation Metrics
# ==========================
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    return acc, prec, rec, f1

# ==========================
# 4. Utility Functions
# ==========================
def load_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}, shape: {df.shape}")
    return df

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Saved DataFrame to {file_path}")
