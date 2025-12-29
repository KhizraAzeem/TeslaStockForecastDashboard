import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load trained regression model
model = joblib.load("models/regression_model.pkl")

# Load test data
X_test = pd.read_csv("data/X_test.csv")

# Create SHAP explainer
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# 1️⃣ Summary Plot (Global Explanation)
shap.summary_plot(shap_values, X_test)
plt.show()

# 2️⃣ Waterfall Plot (Single Prediction)
shap.waterfall_plot(shap_values[0])
plt.show()
