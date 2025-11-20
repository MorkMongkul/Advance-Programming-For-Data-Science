import pandas as pd

def clean_data(df):
    """
    Removes duplicates and null values from the DataFrame.
    """
    df_clean = df.drop_duplicates()
    df_clean = df_clean.dropna()
    return df_clean
