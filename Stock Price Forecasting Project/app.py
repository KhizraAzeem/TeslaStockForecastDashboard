import streamlit as st
import pandas as pd
from modules import utils
from modules import regression_models
from modules import ann_model
from modules import lstm_model
from modules import classification_models
from modules import  clustering_pca
from modules import nlp_module

st.set_page_config(page_title="Stock Price Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

# --- Sidebar Modules ---
module = st.sidebar.selectbox(
    "Select Module",
    [
        "Exploratory Data Analysis",
        "Linear Regression",
        "ANN Model",
        "LSTM Model",
        "Classification",
        "Clustering & PCA",
        "NLP Preprocessing"
    ]
)

if uploaded_file and df is not None:

    # -------------------- EDA --------------------
    if module == "Exploratory Data Analysis":
        st.subheader("🔍 Exploratory Data Analysis")

        st.write(df.head())

        if 'Close' in df.columns:
            fig_hist = utils.plot_histogram(df['Close'], bins=30, title="Distribution of Close Price", return_fig=True)
            st.pyplot(fig_hist)

        try:
            fig_corr = utils.plot_heatmap(df, title="Correlation Heatmap", return_fig=True)
            st.pyplot(fig_corr)
        except Exception as e:
            st.warning(f"Heatmap not available: {e}")

        st.success("✅ EDA Completed")

    # -------------------- Linear Regression --------------------
    elif module == "Linear Regression":
        st.subheader("📊 Linear Regression")

        features = st.multiselect(
            "Select Features",
            options=[col for col in df.select_dtypes(include='number').columns if col != 'Close']
        )
        try:
            if features:
                results = regression_models.run_regression_streamlit(df, features=features)
                st.write(f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
                st.pyplot(results['fig'])
        except Exception as e:
            st.error(f"Regression failed: {e}")

    # -------------------- ANN Model --------------------
    elif module == "ANN Model":
        st.subheader("🤖 Artificial Neural Network")
        epochs = st.slider("Select Epochs", min_value=10, max_value=200, value=50)
        try:
            results = ann_model.run_ann(df, epochs=epochs)
            st.write(f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
            st.pyplot(results['fig'])
        except Exception as e:
            st.error(f"ANN Model failed: {e}")

    # -------------------- LSTM Model --------------------
    elif module == "LSTM Model":
        st.subheader("🧠 LSTM Model")
        n_steps = st.slider("Select Sequence Length (n_steps)", 10, 120, 60)
        try:
            results = lstm_model.run_lstm(df, n_steps=n_steps)
            st.write(f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
            st.pyplot(results['fig'])
        except Exception as e:
            st.warning(f"LSTM model could not run: {e}")
            st.dataframe(df.tail(10))

    # -------------------- Classification --------------------
    elif module == "Classification":
        st.subheader("📈 Classification Models (Price Up/Down)")
        try:
            results = classification_models.run_classification(df)
            for model_name, metrics in results['metrics'].items():
                st.write(f"**{model_name} Metrics:**", metrics)
        except Exception as e:
            st.warning(f"Classification failed: {e}")
            st.dataframe(df.tail(10))

    # -------------------- Clustering & PCA --------------------
    elif module == "Clustering & PCA":
        st.subheader("🔍 KMeans Clustering & PCA")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        try:
            results = clustering_pca.run_clustering_streamlit(df, n_clusters=n_clusters)
            st.pyplot(results['fig'])
            st.write("Cluster counts:")
            st.write(pd.Series(results['clusters']).value_counts())
        except Exception as e:
            st.warning(f"Clustering & PCA failed: {e}")
            st.dataframe(df.select_dtypes(include='number').head())

    # -------------------- NLP Preprocessing --------------------
    elif module == "NLP Preprocessing":
        st.subheader("📝 NLP Preprocessing")
        max_features = st.slider("Max TF-IDF Features", 5, 50, 10)
        try:
            results = nlp_module.run_nlp(df, max_features=max_features)
            st.dataframe(results['df_final'].head())
            st.write("TF-IDF Features:", results['tfidf_columns'])
        except Exception as e:
            st.warning(f"NLP preprocessing failed: {e}")
            st.dataframe(df.head())

else:
    st.info("Please upload a CSV file to start analysis.")



