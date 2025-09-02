from termcolor import colored, cprint
string = "Hello World"
x = 12
goofy = True

print("\033[92mHello \033[91mWorld\033[0m")
if goofy == True:
    cprint("boom " * x, 'magenta')
