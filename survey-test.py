# Import the correct modules to avoid circular import
from survey import routines
from termcolor import cprint

# Use the correct function calls
cprint('•ᴗ• You', 'yellow', attrs=["bold"])
test = routines.input()
print(test)

