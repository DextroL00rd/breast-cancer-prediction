print("✅ Dashboard script started")
from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer, Static
from textual.containers import Vertical, Horizontal
import subprocess

class Dashboard(App):
    CSS_PATH = "dashboard.css"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Horizontal(
                Button("▶ Run main.py", id="run_main"),
                Static("Runs the full prediction pipeline", classes="note"),
            ),
            Horizontal(
                Button("🧪 Run Tests", id="run_tests"),
                Static("Executes all unit tests using pytest", classes="note"),
            ),
            Horizontal(
                Button("📊 Visualize", id="run_visualize"),
                Static("Generates and saves confusion matrix & ROC curve", classes="note"),
            ),
            Horizontal(
                Button("🖼 Open Plots", id="open_plots"),
                Static("Opens saved plot images in default viewer", classes="note"),
            ),
            Horizontal(
                Button("🚀 Run All", id="run_all"),
                Static("Runs pipeline + visualization + opens plots", classes="note"),
            ),
            Static(id="output", expand=True),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        output = self.query_one("#output", Static)
        btn_id = event.button.id

        def run_command(cmd):
            return subprocess.run(cmd, capture_output=True, text=True)

        if btn_id == "run_main":
            result = run_command(["python", "main.py"])
        elif btn_id == "run_tests":
            result = run_command(["pytest"])
        elif btn_id == "run_visualize":
            result = run_command(["python", "scripts/visualize.py"])
        elif btn_id == "open_plots":
            subprocess.run(["start", "confusion_matrix.png"], shell=True)
            subprocess.run(["start", "roc_curve.png"], shell=True)
            output.update("✅ Plots opened.")
            return
        elif btn_id == "run_all":
            result_main = run_command(["python", "main.py"])
            result_vis = run_command(["python", "scripts/visualize.py"])
            subprocess.run(["start", "confusion_matrix.png"], shell=True)
            subprocess.run(["start", "roc_curve.png"], shell=True)
            combined_output = result_main.stdout + "\n\n" + result_vis.stdout
            output.update("✅ All steps completed.\n\n" + combined_output)
            return
        else:
            result = None

        if result:
            output.update(result.stdout or result.stderr)