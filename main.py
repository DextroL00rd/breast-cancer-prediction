# main.py

import argparse
import joblib

from src.data.loader import load_data, preprocess
from src.models.logistic import train_model as train_logistic, evaluate_model as eval_logistic
from src.models.tree import train_model as train_tree, evaluate_model as eval_tree

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, save, and evaluate a classification model"
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "tree"],
        default="logistic",
        help="Choose model: logistic or tree"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_type = args.model

    # 1. Load & preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # 2. Train the chosen model
    if model_type == "logistic":
        model = train_logistic(X_train, y_train)
    else:
        model = train_tree(X_train, y_train)
    print(f"✅ {model_type.capitalize()} model training complete.")

    # 3. Save the trained model artifact
    model_path = f"models/{model_type}_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Saved trained {model_type} model to {model_path}")

    # 4. Evaluate the model
    if model_type == "logistic":
        metrics = eval_logistic(model, X_test, y_test)
    else:
        metrics = eval_tree(model, X_test, y_test)
    print(f"✅ {model_type.capitalize()} metrics: {metrics}")

if __name__ == "__main__":
    main()