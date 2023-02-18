import os
import torch
from torch import optim
from pathlib import Path
from functools import wraps
import numpy as np
import logging
from collections.abc import Iterable

DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
DEFAULT_DATE = '%H:%M:%S'

AUTO_INSTANTIATE = False


class NotInstantiatedError(Exception):
    pass


def set_auto_instantiate(value):
    global AUTO_INSTANTIATE
    AUTO_INSTANTIATE = value


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
        return f"torch {o.dtype}, {tuple(o.shape)}"
    elif isinstance(o, np.ndarray):
        return f"numpy {o.dtype}, {tuple(o.shape)}"
    elif isinstance(o, (dict, list)):
        return f"{type(o)} {len(o)}"
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


# lazy instantiate support


def lazy_init_func(init_func):
    @wraps(init_func)
    def empty_init(self, *arg, init_args_kwargs_ready=False, **kwargs):
        if init_args_kwargs_ready or AUTO_INSTANTIATE:
            init_func(self, *arg, **kwargs)
            self._is_instantiate = True
        else:
            self._not_allowed_lazy_arg = arg
            self._not_allowed_lazy_kwargs = kwargs
            self._is_instantiate = False

    return empty_init


def enable_func(func):
    @wraps(func)
    def enable_or_empty(self, *arg, **kwargs):
        if self._is_instantiate:
            return func(self, *arg, **kwargs)
        else:
            raise NotInstantiatedError(
                f"{type(self).__name__}.{func.__name__} is not enabled since no instantiated"
            )

    return enable_or_empty


def is_self_bound(func):
    return func.__code__.co_varnames[0] == "self"


def instantiate(self, **kwargs):
    if self._is_instantiate:
        return self
    config = self._not_allowed_lazy_kwargs
    config.update(kwargs)
    return type(self)(
        *self._not_allowed_lazy_arg, init_args_kwargs_ready=True, **config
    )


def lazy_init(cls):
    if not hasattr(cls, "_wrap_lazy_init"):
        setattr(cls, "_wrap_lazy_init", False)
    assert (
        cls._wrap_lazy_init or "instantiate" not in cls.__dict__
    ), f"lazy_init will bound a method called instantiate, which is already existed in {cls}"
    if getattr(cls, "_wrap_lazy_init"):
        return cls
    for attr in cls.__dict__:  # there's propably a better way to do this
        attr_o = getattr(cls, attr)
        if attr == "__init__":
            setattr(cls, attr, lazy_init_func(attr_o))
            continue
        if callable(attr_o) and is_self_bound(attr_o):
            setattr(cls, attr, enable_func(attr_o))
    setattr(cls, "_wrap_lazy_init", True)
    setattr(cls, "I", instantiate)
    return cls
