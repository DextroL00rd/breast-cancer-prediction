# src/data/loader.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    """
    Load the Breast Cancer Wisconsin dataset from scikit-learn.
    Returns:
      X (pd.DataFrame): feature matrix with column names
      y (pd.Series): target labels (0 = malignant, 1 = benign)
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def preprocess(X, y, test_size=0.2, random_state=42):
    """
    Scale features and split into train/test sets.
    Args:
      X (pd.DataFrame): raw feature matrix
      y (pd.Series): target labels
      test_size (float): fraction of data for testing
      random_state (int): seed for reproducibility
    Returns:
      X_train, X_test, y_train, y_test
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )