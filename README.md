# 🧠 Breast Cancer Prediction System

This project uses scikit-learn to predict whether a breast tumor is benign or malignant based on medical features. It includes data preprocessing, model training, evaluation, and visualization.

## 📦 Features
- Logistic Regression classifier
- Scaled train/test split
- Confusion matrix and ROC curve plots
- Automated testing with pytest
- GitHub Actions CI pipeline

## 🚀 Getting Started

To install dependencies and run the main script:

```bash
uv sync
uv run python main.py

```markdown
## 📊 Visualize Results

To generate and display the confusion matrix and ROC curve:

```bash
uv run python -m scripts.visualize
