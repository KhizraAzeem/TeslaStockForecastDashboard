# modules/eda_streamlit.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda_streamlit(df):
    """
    Perform Streamlit-ready EDA on stock data.
    Returns dictionary of matplotlib figures for interactive display.
    """

    figs = {}

    # ------------------------------
    # 1. Closing Price Over Time
    # ------------------------------
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df['Date'], df['Close'], color='blue')
    ax1.set_title("Closing Price Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price ($)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    figs['Closing Price'] = fig1

    # ------------------------------
    # 2. Trading Volume Over Time
    # ------------------------------
    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.bar(df['Date'], df['Volume'], color='orange')
    ax2.set_title("Trading Volume Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.tick_params(axis='x', rotation=45)
    figs['Volume'] = fig2

    # ------------------------------
    # 3. High vs Low Prices
    # ------------------------------
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(df['Date'], df['High'], label='High', color='green')
    ax3.plot(df['Date'], df['Low'], label='Low', color='red')
    ax3.set_title("High vs Low Prices")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price ($)")
    ax3.legend()
    ax3.grid(True)
    ax3.tick_params(axis='x', rotation=45)
    figs['High vs Low'] = fig3

    # ------------------------------
    # 4. Close Price Distribution
    # ------------------------------
    fig4, ax4 = plt.subplots(figsize=(8,5))
    sns.histplot(df['Close'], bins=30, kde=True, color='purple', ax=ax4)
    ax4.set_title("Distribution of Closing Prices")
    figs['Close Distribution'] = fig4

    # ------------------------------
    # 5. Correlation Heatmap
    # ------------------------------
    fig5, ax5 = plt.subplots(figsize=(8,6))
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Correlation Between Features")
    figs['Correlation Heatmap'] = fig5

    print("✅ Streamlit-ready EDA completed successfully.")
    return figs

