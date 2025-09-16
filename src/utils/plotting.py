# src/utils/plotting.py

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_confusion_matrix(model, X_test, y_test, output_path=None):
    """
    Plot and save the confusion matrix.
    """
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap="Blues", normalize=None
    )
    plt.title("Confusion Matrix")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()

def plot_roc_curve(model, X_test, y_test, output_path=None):
    """
    Plot and save the ROC curve.
    """
    disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()