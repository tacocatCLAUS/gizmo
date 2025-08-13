from rich.console import Console
from rich.text import Text
import time
import os
import subprocess
import sys
from pathlib import Path

console = Console()

def check_python():
    """Check if Python is properly installed"""
    try:
        version = sys.version.split()[0]
        return True, f"Python {version}"
    except:
        return False, "Python not found"

def check_pip():
    """Check if pip is available"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "pip available"
        else:
            return False, "pip not found"
    except:
        return False, "pip not available"

def check_and_install_requirements():
    """Check and install requirements from model/requirements.txt"""
    requirements_path = Path("model/requirements.txt")
    
    if not requirements_path.exists():
        return True, "No requirements.txt found - skipping"
    
    try:
        # Read requirements
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not requirements:
            return True, f"Requirements file empty - {len(requirements)} packages"
        
        # Try to install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return True, f"Libraries installed - {len(requirements)} packages"
        else:
            return False, f"Failed to install requirements: {result.stderr[:100]}"
            
    except FileNotFoundError:
        return True, "No requirements.txt found - skipping"
    except Exception as e:
        return False, f"Error checking requirements: {str(e)[:100]}"

def check_ollama():
    """Check if Ollama is installed and accessible"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return True, f"Ollama {version}"
        else:
            return False, "Ollama not responding"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, f"Ollama error: {str(e)[:50]}"

def post_sequence():
    """Simulate IBM Netfinity 7000 POST sequence with real checks"""
    # Static messages first
    static_messages = [
        ("Gizmo 7000 MCP Version 12", 1.2),
        ("(c) MIT License CLAUS 2025", 0.8),
        ("Requirements Test in progress...", 1.5),
    ]
    
    for message, delay in static_messages:
        post_text = Text(message)
        post_text.stylize("bold green")
        console.print(post_text)
        time.sleep(delay)
    
    # Dynamic checks
    checks = [
        ("001: Python", check_python),
        ("002: pip", check_pip),
        ("003: Libraries", check_and_install_requirements),
        ("00F: Ollama", check_ollama),
    ]
    
    for check_id, check_func in checks:
        # Show checking message
        checking_text = Text(f"  {check_id}: Checking...")
        checking_text.stylize("bold yellow")
        console.print(checking_text)
        
        # Perform actual check
        success, message = check_func()
        
        # Clear the checking line and show result
        console.print(f"\033[1A\033[2K", end="")  # Move up and clear line
        
        if success:
            result_text = Text(f"  {check_id}: {message} - OK")
            result_text.stylize("bold green")
            console.print(result_text)
        else:
            result_text = Text(f"  {check_id}: {message} - FAILED")
            result_text.stylize("bold red")
            console.print(result_text)
            
            # Exit on critical failures
            if check_func in [check_python, check_ollama]:
                error_text = Text(f"\nCRITICAL ERROR: {message}")
                error_text.stylize("bold red")
                console.print(error_text)
                console.print(Text("Please install the missing component and run again.", "bold red"))
                sys.exit(1)
        
        time.sleep(0.8)
    
    # Continue with remaining static messages
    remaining_messages = [
        ("  010: Hard Disk 0: IBM DCAS-34330 (4.3 GB) - OK", 0.5),
        ("  011: CD-ROM Drive Detected", 0.4),
        ("  012: Network Controller: Intel EtherExpress Pro/100 - OK", 0.5),
        ("Boot Device Priority: SCSI ID 0", 0.8),
        ("Loading Operating System...", 1.2),
        ("Yacana Boot Agent v4.02", 0.6),
        ("Press F to quit.", 0.4),
        ("Starting CUDA 21.3.3 kernel...", 1.0),
        ("CPU0: Initializing MMU... done", 0.3),
        ("CPU1: Initializing MMU... done", 0.3),
        ("SCSI subsystem initialized", 0.4),
        ("eth0: link up, 100Mbps full duplex", 0.4),
        ("Mounting root filesystem (/dev/sda1)... done", 0.6),
        ("Starting inetd, syslogd, and sshd... done", 0.5),
        ("System ready.", 1.0),
        ("", 0.8)  # Empty line for spacing
    ]
    
    for message, delay in remaining_messages:
        if message:  # Only print non-empty messages
            post_text = Text(message)
            post_text.stylize("bold green")
            console.print(post_text)
        else:
            console.print()  # Print empty line
        time.sleep(delay)

def welcome_screen():
    ascii_art = [
        " ███            █████████   ███                                     ",
        "░░░███         ███░░░░░███ ░░░                                      ",
        "  ░░░███      ███     ░░░  ████   █████████ █████████████    ██████ ",
        "    ░░░███   ░███         ░░███  ░█░░░░███ ░░███░░███░░███  ███░░███",
        "     ███░    ░███    █████ ░███  ░   ███░   ░███ ░███ ░███ ░███ ░███",
        "   ███░      ░░███  ░░███  ░███    ███░   █ ░███ ░███ ░███ ░███ ░███",
        " ███░         ░░█████████  █████  █████████ █████░███ █████░░██████ ",
        "░░░            ░░░░░░░░░  ░░░░░  ░░░░░░░░░ ░░░░░ ░░░ ░░░░░  ░░░░░░  "
    ]
    
    # Define gradient colors (from top to bottom)
    colors = ["#cca7df", "#b68cd4", "#b78cd7", "#9174b6", "#684d8f", "#503973", "#302d54", "#1c1444"]
    
    for i, line in enumerate(ascii_art):
        text = Text(line)
        text.stylize(f"bold {colors[i]}")
        console.print(text)
    
    # Print welcome message with gradient too
    welcome_text = "\nTo get started"
    print(welcome_text)
    
    separator_text = "--------------------------------"
    print(separator_text)

def main():
    # First run the POST sequence with real checks
    post_sequence()
    
    # Wait for a moment to simulate boot time
    time.sleep(4)
    
    # Clear the screen after boot sequence
    os.system('cls' if os.name == 'nt' else 'clear')
    time.sleep(2)
    
    # Then show the welcome screen
    welcome_screen()

if __name__ == "__main__":
    main()