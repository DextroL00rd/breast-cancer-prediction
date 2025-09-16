# tests/test_plotting.py

import tempfile
from src.data.loader import load_data, preprocess
from src.models.logistic import train_model
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve

def test_plot_functions_run(tmp_path):
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    model = train_model(X_train, y_train)
    cm_path = tmp_path / "cm.png"
    roc_path = tmp_path / "roc.png"
    plot_confusion_matrix(model, X_test, y_test, output_path=str(cm_path))
    plot_roc_curve(model, X_test, y_test, output_path=str(roc_path))
    assert cm_path.exists()
    assert roc_path.exists()