import pytest
from evaluation import evaluate_model

def test_evaluate_model_perfect():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 1]
    metrics = evaluate_model(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert "accuracy" in metrics
    assert "f1_score" in metrics

def test_evaluate_model_wrong():
    y_true = [1, 0, 1]
    y_pred = [0, 1, 0]
    metrics = evaluate_model(y_true, y_pred)

    assert metrics["accuracy"] == 0.0
    assert metrics["f1_score"] == 0.0
