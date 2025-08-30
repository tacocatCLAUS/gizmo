#!/usr/bin/env python3
"""
F5-TTS Inference Script
Executes F5-TTS text-to-speech inference using local installation
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the local F5-TTS source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src'))

gentext = "I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."

def run_f5_tts_inference(gentext):
    """
    Execute F5-TTS inference using local installation
    """
    
    # Use the local F5-TTS installation
    local_infer_script = os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src', 'f5_tts', 'infer', 'infer_cli.py')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(__file__)
    ref_audio_path = os.path.join(script_dir, 'dataset', '12secondtraining.wav')
    
    # Construct the command to use local script with proper model name
    command = f'''python3 "{local_infer_script}" --model "F5TTS_Base" --ref_audio "{ref_audio_path}" --ref_text "Picture a world just like ours, except the people are a fair bit smarter. In this world, Einstein isn't one in a million, he's one in a thousand. In fact, here he is now." --speed 0.8 --remove_silence --gen_text "{gentext}"'''
    
    # Execute the command
    exit_code = os.system(command)
    
    return exit_code

def play_wav():
    tests_folder = Path("tests")
    if not tests_folder.exists():
        return
    wav_files = list(tests_folder.glob("*.wav"))
    if not wav_files:
        return
    wav_file = wav_files[0]
    if os.name == 'nt':
        subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{wav_file}").PlaySync()'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        if os.uname().sysname == 'Darwin':
            subprocess.run(['afplay', str(wav_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(['aplay', str(wav_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def f5(gentext="I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."):
    run_f5_tts_inference(gentext)
    play_wav()