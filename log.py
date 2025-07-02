from yacana import LoggerManager
devmode = False
def manager(message=None):
    """
    If devmode is False, set the log level to None (no logs).
    If devmode is True, print the given message if it is not None.
    """
    if devmode == False:
        LoggerManager.set_log_level(None)
    else:
        if message is not None:
            print(message)
