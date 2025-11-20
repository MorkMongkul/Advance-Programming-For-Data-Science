from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Returns a dictionary with accuracy and F1 score.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {"accuracy": acc, "f1_score": f1}
