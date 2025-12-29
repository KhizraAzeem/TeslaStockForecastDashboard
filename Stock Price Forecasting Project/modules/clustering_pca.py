# modules/clustering_pca.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def run_clustering(input_data, n_clusters=3):
    """
    Streamlit-ready KMeans clustering + PCA.
    Accepts CSV path, UploadedFile, or DataFrame.
    """
    # Load data
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[numeric_cols].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title(f"PCA 2D Clustering (K={n_clusters})")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    sns.despine()

    return {
        'df': df,
        'clusters': clusters,
        'X_pca': X_pca,
        'fig': fig
    }

