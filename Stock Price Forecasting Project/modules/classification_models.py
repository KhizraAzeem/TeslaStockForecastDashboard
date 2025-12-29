# modules/classification_models_streamlit.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def run_classification(df, features=None, target_col="Price_Up", test_size=0.2):
    """
    Train Logistic Regression and Random Forest on stock price movement (up/down).
    Streamlit-ready: returns metrics and figures for dashboard display.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        features (list): Feature column names. Defaults to numeric columns excluding target.
        target_col (str): Target column for classification.
        test_size (float): Train-test split ratio.

    Returns:
        dict: {
            'log_model', 'rf_model',
            'X_test', 'y_test',
            'y_pred_log', 'y_pred_rf',
            'metrics': metrics dictionary,
            'fig_cm_log': Confusion matrix figure for Logistic Regression,
            'fig_cm_rf': Confusion matrix figure for Random Forest
        }
    """
    # Create target: tomorrow's price up/down
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df[target_col] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    df = df.dropna()

    # Select features
    if features is None:
        features = df.select_dtypes(include='number').columns.drop([target_col, 'Tomorrow_Close']).tolist()

    X = df[features]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate metrics
    metrics = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, y_pred_log),
            "Precision": precision_score(y_test, y_pred_log),
            "Recall": recall_score(y_test, y_pred_log),
            "F1 Score": f1_score(y_test, y_pred_log),
        },
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, y_pred_rf),
            "Precision": precision_score(y_test, y_pred_rf),
            "Recall": recall_score(y_test, y_pred_rf),
            "F1 Score": f1_score(y_test, y_pred_rf),
        }
    }

    # Confusion matrix figures
    from sklearn.metrics import confusion_matrix

    def plot_cm(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        return fig

    fig_cm_log = plot_cm(y_test, y_pred_log, "Logistic Regression CM")
    fig_cm_rf = plot_cm(y_test, y_pred_rf, "Random Forest CM")

    return {
        'log_model': log_model,
        'rf_model': rf_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_log': y_pred_log,
        'y_pred_rf': y_pred_rf,
        'metrics': metrics,
        'fig_cm_log': fig_cm_log,
        'fig_cm_rf': fig_cm_rf
    }


# ================================
# Run directly for testing
# ================================
if __name__ == "__main__":
    df = pd.read_csv("data/TeslaStock_Dataset_Cleaned.csv")
    results = run_classification(df)
    print("📊 Metrics:")
    for model_name, vals in results['metrics'].items():
        print(f"\n{model_name}:")
        for k, v in vals.items():
            print(f"{k}: {v:.4f}")
    results['fig_cm_log'].show()
    results['fig_cm_rf'].show()


