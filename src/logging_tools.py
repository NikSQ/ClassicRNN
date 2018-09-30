import logging

log_formatter = logging.Formatter("[%(name)s][%(level)s] %(message)s")
logfile_name = "rnn.log"
default_log_level = logging.DEBUG


def set_logfile_name(filename):
    global logfile_name
    logfile_name = filename


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(default_log_level)
    logger.propagate = False
    return logger

