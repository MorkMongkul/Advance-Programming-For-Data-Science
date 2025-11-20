import pandas as pd
import pytest
from cleaning import clean_data

def test_clean_data():
    # Create sample data with duplicates and nulls
    data = {
        "id": [1, 2, 2, 3, 4],
        "value": [10, 20, 20, None, 40]
    }
    df = pd.DataFrame(data)

    # Apply cleaning
    df_cleaned = clean_data(df)

    # Assertions
    assert df_cleaned.shape[0] == 3, "Should have 3 rows after cleaning (removed 1 duplicate and 1 null)"
    assert df_cleaned.isnull().sum().sum() == 0, "Should have no null values"
    assert df_cleaned.duplicated().sum() == 0, "Should have no duplicates"
    assert 2 in df_cleaned["id"].values, "ID 2 should still exist"
    assert 3 not in df_cleaned["id"].values, "ID 3 (with null value) should be removed"
