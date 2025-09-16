# tests/test_logistic.py

from src.data.loader import load_data, preprocess
from src.models.logistic import train_model, evaluate_model

def test_train_and_evaluate_basic():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    model = train_model(X_train, y_train, C=0.5, random_state=1)
    metrics = evaluate_model(model, X_test, y_test)
    assert 0.8 <= metrics["accuracy"] <= 1.0
    assert "precision" in metrics and "recall" in metrics