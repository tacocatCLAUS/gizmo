import tkinter as tk
from tkinter import filedialog

def select_file():
    """Opens a file dialog and returns the selected file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Text files", "*.txt", "*.md"), ("Documents", "*.pdf"), ("All files", "*.*")]
    )
    root.destroy()  # Close the hidden Tkinter window
    return file_path

def select_config_dir():
    """Opens a file dialog and returns the selected file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    directory = filedialog.askdirectory(title='Select a Directory')
    root.destroy()  # Close the hidden Tkinter window
    return directory

if __name__ == "__main__":
    selected_file = select_file()
    if selected_file:
        print(f"Selected file: {selected_file}")
        # You can now use the 'selected_file' path to open, read, or process the file
    else:
        print("No file selected.")