from yacana import LoggerManager
devmode = False
def manager(message=None):
    if devmode == False:
        LoggerManager.set_log_level(None)
    else:
        print(message)
        return