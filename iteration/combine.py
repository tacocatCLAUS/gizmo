from pathlib import Path

# Read from system.txt
system_prompt_path = Path("system.txt")
system_prompt = system_prompt_path.read_text()

# Read from skills.txt
skills_prompt_path = Path("skills.txt")
skills = skills_prompt_path.read_text()

# Combine both texts
combined_prompt = system_prompt + "\n\n" + skills  # Optional spacing between the two

# Now `combined_prompt` holds the merged content
print(combined_prompt) 