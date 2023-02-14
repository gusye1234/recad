import os
import torch
from torch import optim
from pathlib import Path
import functools
import inspect
import logging
from collections.abc import Iterable

DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
DEFAULT_DATE = '%H:%M:%S'


def get_default_logging_level():
    return DEFAULT_LEVEL


def get_logger(name, level=None):
    if not level:
        level = get_default_logging_level()
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


def check_dir_or_make(*args):
    p = Path(*args)
    if not p.exists():
        p.mkdir(parents=True)
    return str(p.absolute())


def use_dir(*args):
    p = Path(*args)
    return str(p.absolute())


class VarDim:
    def __init__(self, max=None, min=None, comment=""):
        self.max = max or "?"
        self.min = min or "0"
        self.comment = comment

    def __repr__(self) -> str:
        return f"{self.comment}[{self.min}~{self.max}]"

    def __str__(self) -> str:
        return self.__repr__()


def dict2list_table(d: dict):
    return [(k, v) for k, v in d.items()]


def strip_str(s: str):
    return s.strip().strip("\n")


def type_if_long(o):
    s = str(o)
    if len(s) < 30:
        return s
    elif isinstance(o, torch.Tensor):
        return f"{o.dtype}, {tuple(o.shape)}"
    return type(o)


def user_side_check(args, kwargs):
    args = parse_args(args)
    for a in args:
        if a not in kwargs:
            return False
    return True


def parse_args(args):
    if isinstance(args, str):
        args = args.split(",")
        args = [strip_str(s) for s in args]
    elif isinstance(args, Iterable):
        args = args
    else:
        raise ValueError("Args should be either string or iterable")
    return args


def pick_optim(which):
    if which.lower() == 'adam':
        return optim.Adam
    else:
        if hasattr(optim, which):
            _logger.info(f"load {which} from torch.optim")
            return getattr(optim, which)
        else:
            raise ValueError("optimizer not supported")
