import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_confusion_matrix(model, X_test, y_test, output_path="confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    """
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap="Blues", normalize=None
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

def plot_roc_curve(model, X_test, y_test, output_path="roc_curve.png"):
    """
    Plot and save the ROC curve.
    """
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()