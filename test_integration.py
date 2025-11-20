import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evaluation import evaluate_model

# Mock functions for the pipeline
def load_data():
    # Returns a simple dataframe
    data = {
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [10, 20, 30, 40, 50, 60],
        "target": [0, 0, 0, 1, 1, 1]
    }
    return pd.DataFrame(data)

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def test_ml_pipeline():
    # 1. Load Data
    df = load_data()
    assert not df.empty
    assert "target" in df.columns

    # 2. Train Model
    X = df[["feature1", "feature2"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = train_model(X_train, y_train)
    assert model is not None

    # 3. Evaluate Model
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1.0
    print(f"Integration Test Metrics: {metrics}")
