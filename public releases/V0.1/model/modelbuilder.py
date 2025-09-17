#!/usr/bin/env python3
"""
Ollama Model Builder - Create custom Ollama model from system.txt and skills.txt
"""

import os
import sys
import subprocess

# Configuration
SYSTEM_FILE = "system.txt"
SKILLS_FILE = "skills.txt"
MODELFILE = "Modelfile"
MODEL_NAME = "gizmo"
BASE_MODEL = "wizardlm2:7b"

def build(SYSTEM_FILE='system.txt', SKILLS_FILE='skills.txt', MODELFILE='Modelfile', MODEL_NAME='gizmo', BASE_MODEL='wizardlm2:7b'):
    """Build Ollama model from system and skills files"""
    
    # Ensure required files exist
    for file in [SYSTEM_FILE, SKILLS_FILE]:
        if not os.path.exists(file):
            print(f"Error: '{file}' not found.")
            sys.exit(1)
    
    # Read the content of system.txt and skills.txt
    try:
        with open(SYSTEM_FILE, 'r', encoding='utf-8') as f:
            system_content = f.read().strip()
        
        with open(SKILLS_FILE, 'r', encoding='utf-8') as f:
            skills_content = f.read().strip()
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)
    
    # Generate the Modelfile
    modelfile_content = f"""FROM {BASE_MODEL}

# Optional: tweak parameters below:
PARAMETER temperature 0
# PARAMETER num_ctx 32768

SYSTEM \"\"\"
{system_content}

{skills_content}
\"\"\"
"""
    
    try:
        with open(MODELFILE, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        print(f"Generated '{MODELFILE}'")
    except Exception as e:
        print(f"Error writing Modelfile: {e}")
        sys.exit(1)
    
    # Build/create the custom model
    try:
        result = subprocess.run([
            "ollama", "create", MODEL_NAME, "--file", MODELFILE
        ], check=True, capture_output=True, text=True)
        
        print(f"Model '{MODEL_NAME}' created or updated.")
        if result.stdout:
            print(f"Ollama output: {result.stdout}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error creating Ollama model: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Make sure Ollama is installed and in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    build()