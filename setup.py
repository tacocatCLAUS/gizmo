import subprocess
import sys
import os

# Define the virtual environment name and the library to install
VENV_NAME = "genv"
requirements_file = "requirements.txt"

def create_and_install():
    # 1. Create the virtual environment
    print(f"Creating virtual environment: {VENV_NAME}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", VENV_NAME], check=True)
        print(f"Virtual environment '{VENV_NAME}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

    # 2. Determine the pip executable path within the virtual environment
    if sys.platform == "win32":
        pip_executable = os.path.join(VENV_NAME, "Scripts", "pip.exe")
    else: # Linux/macOS
        pip_executable = os.path.join(VENV_NAME, "bin", "pip")

    # 3. Install the library
    print(f"Installing '{LIBRARY_TO_INSTALL}' into '{VENV_NAME}'...")
    try:
        subprocess.run([pip_executable, "install", "-r", requirements_file], check=True)
        print(f"Packages installed successfully in virtual environment:'{VENV_NAME}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing libraries: {e}")
        sys.exit(1)

if __name__ == "__main__":
print("                ██████╗ ██╗███████╗███╗   ███╗ ██████╗ ")
print(" ʕ 0 ᴥ 0ʔ      ██╔════╝ ██║╚══███╔╝████╗ ████║██╔═══██╗")
print(" |      |      ██║  ███╗██║  ███╔╝ ██╔████╔██║██║   ██║")
print(" |      |      ██║   ██║██║ ███╔╝  ██║╚██╔╝██║██║   ██║")
print(" |      |      ╚██████╔╝██║███████╗██║ ╚═╝ ██║╚██████╔╝")
print("+--------+      ╚═════╝ ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ")
    create_and_install()
    print("\nPIP package install complete. To activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"  .\\{VENV_NAME}\\Scripts\\activate")
    else:
        print(f"  source ./{VENV_NAME}/bin/activate")
    print("")