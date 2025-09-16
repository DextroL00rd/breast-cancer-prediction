# tests/test_loader.py

import pandas as pd
import numpy as np
import pytest
from src.data.loader import load_data, preprocess

def test_load_data_types_and_shapes():
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0] == 569
    assert X.shape[1] == 30

def test_preprocess_splits_and_scaling():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(
        X, y, test_size=0.3, random_state=0
    )
    total = X.shape[0]
    assert X_train.shape[0] == pytest.approx(0.7 * total, rel=0.05)
    assert X_test.shape[0] == pytest.approx(0.3 * total, rel=0.05)
    assert isinstance(X_train, np.ndarray)