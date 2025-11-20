import pandas as pd

def normalize_column(df, column):
    """
    Scales values in the specified column between 0 and 1.
    """
    if column not in df.columns:
        raise KeyError(f"Column {column} not found in DataFrame")

    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()

    if max_val - min_val == 0:
        df_copy[column] = 0.0
    else:
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)

    return df_copy
