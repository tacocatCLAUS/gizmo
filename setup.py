import subprocess
import sys
import os
from Libraries.config_manager import set_openai, set_hackclub, set_ollama, set_rag_model, set_mcp_config_path, set_openai_api_key, get_openai_api_key, enable_voice, enable_devmode, set_db_clear, enable_mcp, get_config, update_config
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
    print("Its...")
    print("                ██████╗ ██╗███████╗███╗   ███╗ ██████╗ ")
    print(" ʕ 0 ᴥ 0ʔ      ██╔════╝ ██║╚══███╔╝████╗ ████║██╔═══██╗")
    print(" |      |      ██║  ███╗██║  ███╔╝ ██╔████╔██║██║   ██║")
    print(" |      |      ██║   ██║██║ ███╔╝  ██║╚██╔╝██║██║   ██║")
    print(" |      |      ╚██████╔╝██║███████╗██║ ╚═╝ ██║╚██████╔╝")
    print("+--------+      ╚═════╝ ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ")
    print("Welcome to the gizmo installer! We're going to get you set up in a jiffy.")
    print("To run this program, please confirm that you have a capable system, you have the option to run the script using an api, but if you want to use either the voice, ollama, or the rag you will NEED a capable system.")
    print("I have a 4070 super and it gets very slow if ollama and tts are running.")
    print("Can your system handle large ai workloads? If you say no RAG, ollama, and voice will be disable by default. Y/N")
    while True:
        canithandleit = input()
        if canithandleit == "Y":
            print("Got it. What model do you want to by default?")
            print("1. OpenAI")
            print("2. Hack Club (NO PERSONAL USE)")
            print("3. Ollama")
            break
        elif canithandleit == "N":
            print("Got it. What model do you want to by default?")
            print("1. OpenAI")
            print("2. Hack Club (NO PERSONAL USE)")
            print("3. Ollama (NOT RECCOMENDED)")
            break
    else:
        print("Input not Y/N, try again.")

    while True:
        model = input("Which one? (1-3):")
        if model == "1":
            model = "openai"
            print("What is your OpenAI key?")
            openai_api_key = input("It is:")
            set_openai(True, "gpt-4", openai_api_key)
            break
        elif model == "2":
            model = "hc"
            set_hackclub(True)
            break
        elif model == "3":
            model = "ollama"
            break
        else:
            print("Please enter a number between 1 & 3.")
    print("This config can be editted later in the config.json file.") 
    print("Creating and installing pip packages in virtual environment...")
    create_and_install()
    print("\nPIP package install complete. To if you need to activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"  .\\{VENV_NAME}\\Scripts\\activate")
    else:
        print(f"  source ./{VENV_NAME}/bin/activate")