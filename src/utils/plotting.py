import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from src.data import load_data

def plot_confusion_matrix(model_type: str, output_path: str = "results/confusion_matrix.png"):
    """
    model_type: "logistic" or "tree"
    output_path: where to save the PNG
    """
    # Load the trained model
    model = joblib.load(f"models/{model_type}_model.joblib")

    # Load your test set
    X_test, y_test = load_data(split="test")

    # Plot & save confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
        normalize=None
    )
    plt.title(f"{model_type.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def plot_roc_curve(model_type: str, output_path: str = "results/roc_curve.png"):
    """
    model_type: "logistic" or "tree"
    output_path: where to save the PNG
    """
    # Load the trained model
    model = joblib.load(f"models/{model_type}_model.joblib")

    # Load your test set
    X_test, y_test = load_data(split="test")

    # Plot & save ROC curve
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
    plt.title(f"{model_type.capitalize()} ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()