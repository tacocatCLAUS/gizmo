import subprocess
import platform
import sys
import os
from Libraries.config_manager import set_openai, set_hackclub, set_ollama, set_rag_model, set_openai_api_key, get_openai_api_key, enable_voice, enable_devmode, set_db_clear, enable_mcp, get_config, update_config
from Libraries.filepicker import select_config_dir
from model.modelbuilder import build
# Define the virtual environment name and the library to install
VENV_NAME = ".genv"
requirements_file = "model/requirements.txt"

if sys.platform == "win32":
    pip_executable = os.path.join(VENV_NAME, "Scripts", "pip.exe")
else:  # Linux/macOS
    pip_executable = os.path.join(VENV_NAME, "bin", "pip")

openai_api_key = ""

def should_install_bitsandbytes():
    system = platform.system()      # 'Darwin', 'Linux', 'Windows'
    machine = platform.machine()    # 'arm64', 'x86_64', etc.
    return not (system == 'Darwin' and machine == 'arm64')

def create_and_install():
    # 1. Create the virtual environment
    print(f"Creating virtual environment: {VENV_NAME}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", VENV_NAME], check=True)
        print(f"✅ Virtual environment '{VENV_NAME}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        sys.exit(1)

    # 2. Determine the pip executable path within the virtual environment
    if sys.platform == "win32":
        pip_executable = os.path.join(VENV_NAME, "Scripts", "pip.exe")
    else:  # Linux/macOS
        pip_executable = os.path.join(VENV_NAME, "bin", "pip")

    # 3. Install from requirements.txt
    print(f"📦 Installing packages from '{requirements_file}'...")
    try:
        subprocess.run([pip_executable, "install", "-r", requirements_file], check=True)
        subprocess.run([pip_executable, "install", "langchain-openai"], check=True)
        print(f"✅ Packages from '{requirements_file}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

    # 4. Conditionally install bitsandbytes
    if should_install_bitsandbytes():
        print("🧠 Detected supported system for bitsandbytes. Installing...")
        try:
            subprocess.run([pip_executable, "install", "bitsandbytes>0.37.0"], check=True)
            print("✅ bitsandbytes installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install bitsandbytes: {e}")
    else:
        print("⚠️ Skipping bitsandbytes: Not supported on Apple Silicon macOS.")

if __name__ == "__main__":
    print("Its...")
    print("                ██████╗ ██╗███████╗███╗   ███╗ ██████╗ ")
    print(" ʕ 0 ᴥ 0ʔ      ██╔════╝ ██║╚══███╔╝████╗ ████║██╔═══██╗")
    print(" |      |      ██║  ███╗██║  ███╔╝ ██╔████╔██║██║   ██║")
    print(" |      |      ██║   ██║██║ ███╔╝  ██║╚██╔╝██║██║   ██║")
    print(" |      |      ╚██████╔╝██║███████╗██║ ╚═╝ ██║╚██████╔╝")
    print("+--------+      ╚═════╝ ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ")
    print("Welcome to the gizmo installer! We're going to get you set up in a jiffy.")
    print("First, do you have ollama inswtalled? This is requried to run the script.")
    while True:
        ollama = input("Y/N:").upper()
        if ollama == "Y":
            print("Great! Let's continue.")
            break
        elif ollama == "N":
            if platform.system() == 'Windows':
                import webbrowser
                webbrowser.open('https://ollama.com/download')
                print("Please install ollama from ollama.com and run this script again.")
            if platform.system() == 'Darwin':
                print("Please install ollama from ollama.org and run this script again.")
                import webbrowser
                webbrowser.open('https://ollama.com/download')
            if platform.system() == 'Linux':
                print("Installing Ollama... This may take a while...")
                subprocess.run(['bash', '-c', 'curl -sSL https://ollama.com/install.sh | sh'], check=True)
                print("Ollama installed.")
            break
    else:
        print("Input not Y/N, try again.")
    
    print("To run this program, please confirm that you have a capable system, you have the option to run the script using an api, but if you want to use either the voice, ollama, or the rag you will NEED a capable system.")
    print("I have a 4070 super and it gets very slow if ollama and tts are running.")
    print("Can your system handle large ai workloads? If you say no, RAG, ollama, and voice will be disabled.")
    while True:
        canithandleit = input("Y/N:").upper()
        if canithandleit == "Y":
            print("Got it. What model do you want to by default?")
            print("1. OpenAI")
            print("2. Hack Club (NO PERSONAL USE)")
            print("3. Ollama")
            break
        elif canithandleit == "N":
            set_rag_model("none")
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
            os.environ["OPENAI_API_KEY"] = openai_api_key
            set_hackclub(False)
            break
        elif model == "2":
            model = "hc"
            set_hackclub(True)
            set_openai(False)
            break
        elif model == "3":
            model = "ollama"
            set_hackclub(False)
            set_openai(False)
            set_ollama()
            print("If you dont have crazy hardware, your pc will slow down when you use ollama. After the script is quit, there is a five minute cooldown before speedup.")
            print("Pulling  and training WizardLM2 7B model, this may take a while...")
            subprocess.run(['ollama', 'pull', 'wizardlm2:7b'], check=True)
            print("wizardlm2:7b pulled.")
            print("Building gizmo model...")
            build(SYSTEM_FILE='model/system.txt', SKILLS_FILE='model/skills.txt', MODELFILE='model/Modelfile', MODEL_NAME='gizmo', BASE_MODEL='wizardlm2:7b')
            print("gizmo model built.")
            break
        else:
            print("Please enter a number between 1 & 3.")
    print("Okay! You are almost done. Would you like to use rag? This will create a local database of files you upload. Use this for large pdfs or text documents.")
    use_rag = input("ollama/openai/none:")
    if use_rag == "openai":
        set_rag_model("openai")
        print("RAG set.")
        if openai_api_key == "":
            print("What is your OpenAI key?")
            openai_api_key = input("It is:")
            set_openai_api_key(openai_api_key)
            os.environ["OPENAI_API_KEY"] = openai_api_key
    elif use_rag == "ollama":
        set_rag_model("ollama")
        print("RAG set, installing nomic-embed-text:latest...")
        subprocess.run(['ollama', 'pull', 'nomic-embed-text:latest'], check=True)
        print("nomic-embed-text:latest installed.")
    else:
        set_rag_model("none")
        print("RAG disabled.")
    print("Would you like to use MCP? This is still in development but it works pretty well. Just use mcp.json (Y/N)")
    while True:
        use_mcp = input("Y/N:").upper()
        if use_mcp == "Y":
            enable_mcp(True)
            print("MCP enabled.")
            break
        elif use_mcp == "N":
            enable_mcp(False)
            print("MCP disabled.")
            break
    else:
        print("Input not Y/N, try again.")
    print("Now for the crazy one. Would you like to enable voice? (Y/N) I have a 4070 super and running voice and ollama gets a good 1fps.")
    while True:
        use_voice = input("Y/N:").upper()
        if use_voice == "Y":
            enable_voice(True)
            print("Voice enabled. You should probably change this later. Or go crazy i guess.")
            break
        elif use_voice == "N":
            enable_voice(False)
            print("Voice disabled.")
            break
    else:
        print("Input not Y/N, try again.")
    print("These settings can be editted later in the config.json file.")
    print("Creating and installing pip packages in virtual environment...")
    create_and_install()
    if should_install_bitsandbytes():
        print("🧠 Detected supported system for bitsandbytes. Installing...")
        try:
            subprocess.run([pip_executable, "install", "bitsandbytes>0.37.0"], check=True)
            print("✅ bitsandbytes installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install bitsandbytes: {e}")
    else:
        print("⚠️ Skipping bitsandbytes: Not supported on Apple Silicon macOS.")
    print("\nIf you want to activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"  .\\{VENV_NAME}\\Scripts\\activate")
    else:
        print(f"  source ./{VENV_NAME}/bin/activate")
    print("But it should be done automatically when you run gizmo.py")
    print("RUNNING GIZMO...")
    subprocess.run([sys.executable, "gizmo.py"], check=True)
    