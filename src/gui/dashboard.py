# src/gui/dashboard.py

from textual.app import App
from textual.widgets import Button, Static, Header, Footer

# Import model training functions
from src.models.logistic import train_model as train_logistic_model
from src.models.tree import train_model as train_tree_model

# Import plotting functions
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve

class DashboardApp(App):
    def compose(self):
        yield Header()
        yield Static("Breast Cancer Prediction Dashboard", id="title")
        yield Button("Train Logistic",       id="train-logistic")
        yield Button("Train Decision Tree",  id="train-tree")
        yield Button("Show Confusion Matrix", id="show-conf")
        yield Button("Show ROC Curve",        id="show-roc")
        yield Static("", id="status")       # status messages
        yield Footer()

    async def on_button_pressed(self, event):
        status = self.query_one("#status", Static)
        btn = event.button.id

        if btn == "train-logistic":
            status.update("Training logistic model…")
            try:
                train_logistic_model()
                status.update("✅ Logistic training complete.")
            except Exception as e:
                status.update(f"❌ Error: {e}")

        elif btn == "train-tree":
            status.update("Training decision tree model…")
            try:
                train_tree_model()
                status.update("✅ Decision tree training complete.")
            except Exception as e:
                status.update(f"❌ Error: {e}")

        elif btn == "show-conf":
            status.update("Generating confusion matrix…")
            try:
                plot_confusion_matrix(model_type="logistic", output_path="results/logistic_confusion_matrix.png")
                status.update("✅ Confusion matrix saved to results/")
            except Exception as e:
                status.update(f"❌ Error: {e}")

        elif btn == "show-roc":
            status.update("Generating ROC curve…")
            try:
                plot_roc_curve(model_type="logistic", output_path="results/logistic_roc_curve.png")
                status.update("✅ ROC curve saved to results/")
            except Exception as e:
                status.update(f"❌ Error: {e}")

if __name__ == "__main__":
    DashboardApp().run()