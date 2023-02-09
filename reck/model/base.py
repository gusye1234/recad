from tabulate import tabulate
from functools import partial
from abc import ABC, abstractmethod
from torch.nn import Module
from ..default import MODEL
from ..utils import strip_str
from copy import copy


class BaseModel(Module, ABC):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    @abstractmethod
    def from_config(cls, name, arg_string, user_config):
        assert name in MODEL, f"{name} is not on the default datasets"
        args = arg_string.split(",")
        args = [strip_str(arg) for arg in args if strip_str(arg)]
        try:
            default_config = {k: copy(MODEL[name][k]) for k in args}
        except:
            error_message = f"{args} contain non-legal keys for {cls}!, expect in {list(MODEL[name])}"
            raise KeyError(error_message)
        for k in user_config.keys():
            assert k in args, f"Unexpected key [{k}] for {cls}, expected in {args}"
        default_config.update(user_config)

        return cls(**default_config)

    @abstractmethod
    def input_describe(self):
        raise NotImplementedError

    @abstractmethod
    def output_describe(self):
        raise NotImplementedError

    def help_info(self, **kwargs) -> str:
        input_des = self.input_describe()
        assert isinstance(input_des, dict), "Wrong return type of batch_describe"

        headers = [["name", "type", "shape"]]
        for k, v in input_des.items():
            assert len(v) == 2
            headers.append((k, str(v[0]), str(v[1])))
        info = f"Input:\n{BaseModel.fmt_tab(headers)}\n"

        output_des = self.output_describe()
        assert isinstance(output_des, dict), "Wrong return type of batch_describe"
        headers = [["name", "type", "shape"]]
        for k, v in output_des.items():
            assert len(v) == 2
            headers.append((k, str(v[0]), str(v[1])))
        info = info + f"Output:\n{BaseModel.fmt_tab(headers)}\n"
        return info
