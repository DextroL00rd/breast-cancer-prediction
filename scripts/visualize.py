# scripts/visualize.py

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
    main()