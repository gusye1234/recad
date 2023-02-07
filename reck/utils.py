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


# def configurable(init_func):
#     assert init_func is not None, "@configurable is only used from __init__"
#     assert (
#         inspect.isfunction(init_func) and init_func.__name__ == "__init__"
#     ), "@configurable is only used from __init__"

#     @functools.wraps(init_func)
#     def wrapped(self, *args, **kwargs):
#         try:
#             from_config_func = type(self).from_config
#         except AttributeError as e:
#             raise AttributeError(
#                 "Class with @configurable must have a 'from_config' classmethod."
#             ) from e
#         if not inspect.ismethod(from_config_func):
#             raise TypeError(
#                 "Class with @configurable must have a 'from_config' classmethod."
#             )
#         config = from_config_func()
#         if _called_with_cfg(*args, **kwargs):
#             explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
#             init_func(self, **explicit_args)
#         else:
#             init_func(self, *args, **kwargs)

#     return wrapped
