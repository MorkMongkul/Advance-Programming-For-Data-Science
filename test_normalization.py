import pandas as pd
import pytest
# We import the function even though it might not exist yet (TDD process)
# In a real TDD cycle, this import would fail first.
try:
    from normalization import normalize_column
except ImportError:
    pass

def test_normalize_column():
    df = pd.DataFrame({"score": [10, 20, 30]})

    # Test normalization
    df_norm = normalize_column(df, "score")

    assert df_norm["score"].min() == 0.0
    assert df_norm["score"].max() == 1.0
    assert len(df_norm) == 3

def test_invalid_column():
    df = pd.DataFrame({"score": [10, 20, 30]})
    with pytest.raises(KeyError):
        normalize_column(df, "invalid_col")
