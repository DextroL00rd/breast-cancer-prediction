# scripts/visualize.py
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize confusion matrix & ROC for a chosen model"
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "tree"],
        default="logistic",
        help="Which model to visualize: 'logistic' or 'tree'"
    )
    return parser.parse_args()
from src.data.loader import load_data, preprocess
from src.models.logistic import train_model
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)

    model = train_model(X_train, y_train)

    plot_confusion_matrix(model, X_test, y_test, output_path="confusion_matrix.png")
    plot_roc_curve(model, X_test, y_test, output_path="roc_curve.png")

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model

    # Example: adjust your filenames or logic based on model_type
    cm_path = f"results/{model_type}_confusion_matrix.png"
    roc_path = f"results/{model_type}_roc_curve.png"

    # Then call your existing plot functions, passing model_type if needed
    plot_confusion_matrix(model_type, output_path=cm_path)
    plot_roc_curve(model_type, output_path=roc_path)

    print(f"Saved plots for {model_type}:")
    print(f"  • {cm_path}")
    print(f"  • {roc_path}")