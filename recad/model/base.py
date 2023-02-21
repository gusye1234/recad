from functools import partial
from abc import ABC, abstractmethod
from collections.abc import Iterable

from torch.nn import Module
from ..default import MODEL
from ..utils import (
    strip_str,
    get_logger,
    user_side_check,
    parse_args,
    lazy_init,
    type_if_long,
)
from pprint import pprint
from copy import copy

logger = get_logger(__name__)


class BaseModel(Module, ABC):
    def __new__(cls, *args, **kwargs):
        cls = lazy_init(cls, record_attrs=['_init_config', '_model_name'])
        instan = object.__new__(cls)
        return instan

    @classmethod
    @abstractmethod
    def from_config(cls, scope, name, arg_string, user_args, user_config):
        assert name in MODEL[scope], f"{name} is not on the default {scope} models"
        # if not user_side_check(user_args, user_config):
        #     print("will not I")
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

        instan = BaseModel.__new__(cls)
        instan._init_config = default_config
        instan._model_name = name
        instan.__init__(**default_config)
        return instan

    # TODO change describe method to classmethod
    def input_describe(self):
        """Input description for method forward and object_forward"""
        raise NotImplementedError

    def output_describe(self):
        """Output description for method forward and object_forward"""
        raise NotImplementedError

    def info_describe(self):
        return {
            "input_describe": self.input_describe(),
            "output_describe": self.output_describe(),
        }

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
        info_des = self.info_describe()
        assert isinstance(
            info_des, dict
        ), "Wrong return type of info_describe, expected Dict"
        info_des['model_name'] = self.model_name
        for k, v in info_des.items():
            if k in ['input_describe', 'output_describe']:
                continue
            v = type_if_long(v)
            info_des[k] = v
        pprint(info_des)

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
