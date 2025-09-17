# launch.py
import os
import subprocess

project_root = r"C:\Users\ayman\Desktop\Projects\AI_Course\breast-cancer-prediction"
os.chdir(project_root)

venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
dashboard_script = os.path.join(project_root, "src", "gui", "dashboard.py")

subprocess.run([venv_python, dashboard_script])