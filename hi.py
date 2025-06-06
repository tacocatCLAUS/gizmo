from pathlib import Path
import ollama
import threading
import time
import sys

# Animated spinner class (for initial load only)
class Spinner:
    def __init__(self):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.stop_running = False
        self.spinner_thread = None
    
    def spin(self):
        while not self.stop_running:
            for char in self.spinner_chars:
                sys.stdout.write(f"\rLoading system prompt... {char}")
                sys.stdout.flush()
                time.sleep(0.1)
                if self.stop_running:
                    break
    
    def start(self):
        self.stop_running = False
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()
    
    def stop(self):
        self.stop_running = True
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write("\r" + " " * 40 + "\r")  # Clear the spinner line
        sys.stdout.flush()

# Read from system.txt
system_prompt_path = Path("system.txt")
system_prompt = system_prompt_path.read_text()

# Read from skills.txt
skills_prompt_path = Path("skills.txt")
skills = skills_prompt_path.read_text()

# Combine both texts
combined_prompt = system_prompt + "\n\n" + skills
print(combined_prompt)

# Show spinner only for initial system prompt load
spinner = Spinner()
spinner.start()

# Initialize the chat with system prompt
try:
    response = ollama.chat(
        model='mistral:7b',
        messages=[{'role': 'system', 'content': combined_prompt}]
    )
finally:
    spinner.stop()

print("\nSystem ready! Type 'quit' to exit.\n")

# Continuous conversation loop (without spinner)
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['quit', 'exit']:
        break
        
    # Get response without spinner
    response = ollama.chat(
        model='mistral:7b',
        messages=[{'role': 'user', 'content': user_input}],
        stream=True
    )
    
    # Stream the response
    print("Gizmo: ", end='', flush=True)
    for chunk in response:
        content = chunk['message']['content']
        print(content, end='', flush=True)
    print()  # New line after response