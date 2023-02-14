import torch
from abc import ABC, abstractmethod
from functools import partial
from copy import copy
from ..default import WORKFLOW
from ..utils import get_logger, parse_args, user_side_check
from tabulate import tabulate

logger = get_logger(__name__)


class BaseFlow(ABC):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    @abstractmethod
    def from_config(cls, name, arg_string, user_args, user_config):
        assert name in WORKFLOW, f"{name} is not on the default workflows"
        if not user_side_check(user_args, user_config):
            raise TypeError(f"Expect for user arguments [{user_args}]")
        # come from default
        args = parse_args(arg_string)
        user_args = parse_args(user_args)
        try:
            default_config = {k: copy(WORKFLOW[name][k]) for k in args}
        except:
            error_message = f"{args} contain non-legal keys for {cls}!, expect in {list(WORKFLOW[name])}"
            raise KeyError(error_message)
        filter_user_config = {}
        for k, v in user_config.items():
            if k in args or k in user_args:
                filter_user_config[k] = v
            else:
                logger.debug(f"Unexpected key [{k}] for {cls}")
        default_config.update(filter_user_config)
        return cls(**default_config)

    @abstractmethod
    def input_describe(self):
        """return the types of datasets and models this workflow needs"""
        raise NotImplementedError

    @abstractmethod
    def execute(self):
        """return the types of datasets and models this workflow needs"""
        raise NotImplementedError
