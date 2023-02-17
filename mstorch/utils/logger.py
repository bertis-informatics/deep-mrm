import logging

LOG_MSG_FORMAT = '[%(asctime)s] [%(levelname)7s] [%(name)s] %(message)s'
LOG_DATE_FORMAT = '%m-%d-%Y %H:%M:%S'


def get_logger(name="mstorch", logger=None, level=logging.DEBUG):
    """ configure logger object and return it

    :param logger: user specified logger
    :param debug: bool, set logger level to DEBUG and override user specified logger's level
    :return: logger object
    """
    if logger is not None:
        if logger.getLevel() != level:
            logger.setLevel(level)
        return logger

    # Remove top-level handlers to avoid duplicated log records
    root_logger = logging.getLogger()
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)

    # check if there is logger
    _logger = logging.getLogger(name)
    _logger.setLevel(level)

    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_MSG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    else:
        # make sure there is only one handler
        for hdlr in _logger.handlers[1:]:
            _logger.removeHandler(hdlr)

    return _logger