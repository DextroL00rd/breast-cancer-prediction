from src.data.loader import load_data, preprocess
from src.models.logistic import train_model, evaluate_model

def main():
    # 1. Load & preprocess
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # 2. Train
    model = train_model(X_train, y_train)
    print("✅ Model training complete.")

    # 3. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"✅ Metrics: {metrics}")

if __name__ == "__main__":
    main()