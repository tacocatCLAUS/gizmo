#!/usr/bin/env python3
"""
F5-TTS Inference Script with Output Suppression
Executes F5-TTS text-to-speech inference while suppressing most output
Only shows progress bars and "Generating audio" messages
"""

import os
import sys
import subprocess
from pathlib import Path
from contextlib import contextmanager
import re
import gc

# Add the local F5-TTS source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src'))

gentext = "I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."

@contextmanager
def selective_stdout_suppression():
    """
    Context manager that suppresses stdout except for progress bars and specific messages
    """
    class SelectiveStdout:
        def __init__(self, original_stdout):
            self.original_stdout = original_stdout
            self.buffer = ""
            
        def write(self, text):
            # Keep progress bars and "Generating audio" messages
            if any(keyword in text for keyword in [
                "Generating audio",
                "100%|", "|", "█", "/", "[", "]", "%", "it/s", "s/it"
            ]):
                self.original_stdout.write(text)
                self.original_stdout.flush()
            # Suppress everything else
            
        def flush(self):
            self.original_stdout.flush()
            
        def __getattr__(self, name):
            return getattr(self.original_stdout, name)
    
    original_stdout = sys.stdout
    sys.stdout = SelectiveStdout(original_stdout)
    try:
        yield
    finally:
        sys.stdout = original_stdout

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def run_f5_tts_inference(gentext):
    """
    Execute F5-TTS inference with selective output suppression and resource management
    Only shows progress bars and "Generating audio" messages
    """
    
    # Use the local F5-TTS installation
    local_infer_script = os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src', 'f5_tts', 'infer', 'infer_cli.py')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(__file__)
    ref_audio_path = os.path.join(script_dir, 'dataset', '12secondtraining.wav')
    
    # Construct the command
    command = [
        'python3', local_infer_script,
        '--model', 'F5TTS_Base',
        '--ref_audio', ref_audio_path,
        '--ref_text', 'Picture a world just like ours, except the people are a fair bit smarter. In this world, Einstein isn\'t one in a million, he\'s one in a thousand. In fact, here he is now.',
        '--speed', '0.8',
        '--remove_silence',
        '--gen_text', gentext
    ]
    
    try:
        # Force garbage collection before starting
        gc.collect()
        
        # Run the process with aggressive resource limits
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Completely suppress stderr
            universal_newlines=True,
            bufsize=0,  # Unbuffered for immediate output
            preexec_fn=None,
            close_fds=True  # Close unused file descriptors
        )
        
        # Process output with memory management
        output_buffer = []
        for line in process.stdout:
            # Only keep progress-related lines
            if any(keyword in line for keyword in [
                "Generating audio",
                "100%|", "|", "█", "/", "[", "]", "it/s", "s/it", "%|"
            ]):
                print(line, end='')
                sys.stdout.flush()
            
            # Prevent memory buildup - clear buffer periodically
            output_buffer.append(line)
            if len(output_buffer) > 100:  # Clear every 100 lines
                output_buffer.clear()
                gc.collect()  # Force garbage collection
        
        # Wait for process to complete with timeout
        try:
            return_code = process.wait(timeout=300)  # 5 minute timeout
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print("Process timed out and was terminated")
            return 1
            
        # Clean up
        process.stdout.close()
        del output_buffer
        gc.collect()
        
        return return_code
        
    except Exception as e:
        print(f"Error running F5-TTS: {e}")
        # Clean up on error
        if 'process' in locals():
            try:
                process.kill()
                process.wait()
            except:
                pass
        gc.collect()
        return 1

def run_f5_tts_inference_alternative(gentext):
    """
    Alternative method using the original os.system approach with output filtering
    """
    
    # Use the local F5-TTS installation
    local_infer_script = os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src', 'f5_tts', 'infer', 'infer_cli.py')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(__file__)
    ref_audio_path = os.path.join(script_dir, 'dataset', '12secondtraining.wav')
    
    # Construct the command with output redirection
    command = f'''python3 "{local_infer_script}" --model "F5TTS_Base" --ref_audio "{ref_audio_path}" --ref_text "Picture a world just like ours, except the people are a fair bit smarter. In this world, Einstein isn't one in a million, he's one in a thousand. In fact, here he is now." --speed 0.8 --remove_silence --gen_text "{gentext}" 2>/dev/null | grep -E "(Generating audio|100%|█|\\[|\\]|it/s|s/it)"'''
    
    # Execute the command (Unix/Linux/Mac only due to grep)
    exit_code = os.system(command)
    return exit_code

def play_wav():
    """Play the generated WAV file"""
    tests_folder = Path("tests")
    if not tests_folder.exists():
        return
    wav_files = list(tests_folder.glob("*.wav"))
    if not wav_files:
        return
    wav_file = wav_files[0]
    
    # Play silently (suppress audio player output too)
    with suppress_stderr():
        if os.name == 'nt':
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{wav_file}").PlaySync()'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if os.uname().sysname == 'Darwin':
                subprocess.run(['afplay', str(wav_file)], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(['aplay', str(wav_file)], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def f5(gentext="I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."):
    print("Starting F5-TTS generation...")
    run_f5_tts_inference(gentext)
    print("Playback starting...")
    play_wav()
    print("Complete!")

# For complete silence (no output at all)
def f5_completely_silent(gentext="I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."):
    """
    Run F5-TTS with absolutely no output
    """
    with suppress_stderr():
        process = subprocess.Popen([
            'python3', os.path.join(os.path.dirname(__file__), 'F5-TTS', 'src', 'f5_tts', 'infer', 'infer_cli.py'),
            '--model', 'F5TTS_Base',
            '--ref_audio', os.path.join(os.path.dirname(__file__), 'dataset', '12secondtraining.wav'),
            '--ref_text', 'Picture a world just like ours, except the people are a fair bit smarter. In this world, Einstein isn\'t one in a million, he\'s one in a thousand. In fact, here he is now.',
            '--speed', '0.8',
            '--remove_silence',
            '--gen_text', gentext
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.wait()
    
    play_wav()

def f5(gentext="I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."):
    """
    Run F5-TTS with only progress information displayed
    """
    print("Starting F5-TTS generation...")
    run_f5_tts_inference(gentext)
    print("Playback starting...")
    play_wav()
    print("Complete!")

if __name__ == "__main__":
    # Test with selective output (shows progress only)
    f5("Hello, this is a test of the silent F5-TTS system.")
    
    # Uncomment for completely silent operation:
    # f5_completely_silent("This will run completely silently.")