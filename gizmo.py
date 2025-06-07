from yacana import Task, OllamaAgent
from pathlib import Path

# Read from system.txt
system_prompt_path = Path("system.txt")
system_prompt = system_prompt_path.read_text()

# Read from skills.txt
skills_prompt_path = Path("skills.txt")
skills = skills_prompt_path.read_text()

# Combine both texts
combined_prompt = system_prompt + "\n\n" + skills  # Optional spacing between the two

ollama_agent = OllamaAgent("Gizmo", "mistral:7b", system_prompt=combined_prompt)

# Create a task to tell a joke
message = Task("Tell me joke but in reverse.", ollama_agent).solve()

print(message.content)
# ? SAD BOOK MATH THE WAS WHY

# Chain a second task to tell the same joke but in uppercase
message = Task("Tell it again but ALL CAPS LIKE YOU ARE SCREAMING !", ollama_agent).solve()

print(message.content)
# !PROBLEMS MANY TOO HAD IT BECAUSE