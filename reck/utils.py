import os
from pathlib import Path
import functools
import inspect
import logging

DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
DEFAULT_DATE = '%H:%M:%S'


def get_logger(name, level=None):
    if not level:
        level = DEFAULT_LEVEL
    logger = logging.getLogger(name)
    try:
        import coloredlogs

        coloredlogs.DEFAULT_LOG_FORMAT = DEFAULT_FORMAT
        coloredlogs.DEFAULT_DATE_FORMAT = DEFAULT_DATE
        coloredlogs.install(level=level, logger=logger)
    except:
        if not len(logger.handlers):
            formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE)
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)

    return logger


_logger = get_logger(__name__)


def root_path():
    return str(Path(os.path.dirname(__file__), "..").absolute())


def check_path_or_empty(*args):
    p = Path(*args)
    if p.exists():
        return str(p.absolute())
    else:
        _logger.warning(f"Expect {p.absolute()} to exist, but not")
        return ""


class VarDim:
    def __init__(self, max=None, min=None):
        self.max = max or "?"
        self.min = min or "?"

    def __repr__(self) -> str:
        return f"{self.min}~{self.max}"

    def __str__(self) -> str:
        return self.__repr__()


def dict2list_table(d: dict):
    return [(k, repr(v)) for k, v in d.items()]


def strip_str(s: str):
    return s.strip().strip("\n")
