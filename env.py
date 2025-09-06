import platform
import subprocess
import os

def run_in_venv(script: str):
    if platform.system() == 'Windows':
        python_exe = os.path.join('.genv', 'Scripts', 'python.exe')
    else:
        python_exe = os.path.join('.genv', 'bin', 'python')

    subprocess.run([python_exe, script])

# Example usage
# Suppose you have a script `test.py`
run_in_venv('gizmo.py')
