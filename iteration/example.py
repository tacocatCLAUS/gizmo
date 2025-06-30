content = "し | Latest news about iPhone | recent iPhone updates | new iOS features | June 4, 2025"

# Split the content on the pipe symbol and strip any extra whitespace
parts = [part.strip() for part in content.split("|")]

# Assign variables based on their position
search_1 = parts[1]
primary_search = parts[2]
search_2 = parts[3]

print("search_1:", search_1)
print("primary_search:", primary_search)
print("search_2:", search_2)
