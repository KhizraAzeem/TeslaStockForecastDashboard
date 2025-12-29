import pandas as pd

def clean_data(input_data):
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif hasattr(input_data, "read"):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, str):
        df = pd.read_csv(input_data)
    else:
        raise ValueError("input_data must be file, path, or DataFrame")

    df = df.drop_duplicates()
    df = df.fillna(method="ffill")

    return df




