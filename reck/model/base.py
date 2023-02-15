from tabulate import tabulate
from functools import partial
from abc import ABC, abstractmethod
from collections.abc import Iterable

from torch.nn import Module
from ..default import MODEL
from ..utils import strip_str, get_logger, user_side_check, parse_args
from copy import copy

logger = get_logger(__name__)


class BaseModel(Module, ABC):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    @abstractmethod
    def from_config(cls, scope, name, arg_string, user_args, user_config):
        assert name in MODEL[scope], f"{name} is not on the default {scope} models"
        if not user_side_check(user_args, user_config):
            raise TypeError(f"Expect for user arguments [{user_args}]")
        # come from default
        args = parse_args(arg_string)
        user_args = parse_args(user_args)
        try:
            default_config = {k: copy(MODEL[scope][name][k]) for k in args}
        except:
            error_message = f"{args} contain non-legal keys for {cls}!, expect in {list(MODEL[scope][name])}"
            raise KeyError(error_message)
        filter_user_config = {}
        for k, v in user_config.items():
            if k in args or k in user_args:
                filter_user_config[k] = v
            else:
                logger.debug(f"Unexpected key [{k}] for {cls}")
        default_config.update(filter_user_config)

        instan = object.__new__(cls)
        instan._init_config = default_config
        instan._model_name = name
        instan.__init__(**default_config)
        return instan

    # TODO change describe method to classmethod
    @abstractmethod
    def input_describe(self):
        """Input description for method forward and object_forward"""
        raise NotImplementedError

    @abstractmethod
    def output_describe(self):
        """Output description for method forward and object_forward"""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, **config):
        """Unlike the forward method, this method should return the results relative to the object function"""
        raise NotImplementedError

    @property
    def model_name(self):
        if hasattr(self, "_model_name"):
            return self._model_name
        return type(self).__name__

    def print_help(self, **kwargs) -> str:
        input_des = self.input_describe()
        assert isinstance(input_des, dict), "Wrong return type of batch_describe"

        info = f"Input:\n"
        for keys in input_des:
            headers = [["name", "type", "shape"]]
            for k, v in input_des[keys].items():
                assert len(v) == 2
                headers.append((k, str(v[0]), str(v[1])))
            info = info + f"{keys}:\n{BaseModel.fmt_tab(headers)}\n"

        output_des = self.output_describe()
        assert isinstance(output_des, dict), "Wrong return type of batch_describe"

        info = info + f"Output:\n"
        for keys in output_des:
            headers = [["name", "type", "shape"]]
            for k, v in output_des[keys].items():
                assert len(v) == 2
                headers.append((k, str(v[0]), str(v[1])))
            info = info + f"{keys}:\n{BaseModel.fmt_tab(headers)}\n"
        print(info)

    def reset(self, **kwargs):
        # inject may change some files
        if hasattr(self, "_init_config"):
            config = copy(self._init_config)
            for k, v in kwargs.items():
                if k in config:
                    config[k] = v
                else:
                    raise ValueError(f"reset arg {k} should be in {list(config)}")
            return type(self).from_config(**config)
        raise ValueError("reset method is only for datasets instantiated from_config")
