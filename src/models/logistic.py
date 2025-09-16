# src/models/logistic.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train, random_state=42, C=1.0):
    """
    Train a Logistic Regression classifier within a scaling pipeline.
    Args:
      X_train (array-like): scaled training features
      y_train (array-like): training labels
      random_state (int): seed for reproducibility
      C (float): inverse regularization strength
    Returns:
      Pipeline: fitted sklearn Pipeline (scaler + LogisticRegression)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  
        ("clf", LogisticRegression(random_state=random_state, C=C))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print key metrics.
    Args:
      model (Pipeline): fitted model pipeline
      X_test (array-like): scaled test features
      y_test (array-like): test labels
    Returns:
      dict: accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    return metrics