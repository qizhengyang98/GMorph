LOG_FILE_LOC = '/home/opt.log'

def get_log_file_loc() -> str:
    return LOG_FILE_LOC

def set_log_file_loc(loc: str) -> None:
    global LOG_FILE_LOC
    LOG_FILE_LOC = loc