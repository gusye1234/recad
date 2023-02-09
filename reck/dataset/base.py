from tabulate import tabulate
from functools import partial
from typing import Dict, Generator
from abc import ABC, abstractmethod
from ..default import DATASET
from ..utils import strip_str
from copy import copy


class BaseDataset(ABC):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    @abstractmethod
    def from_config(cls, name, arg_string, user_config):
        assert name in DATASET, f"{name} is not on the default datasets"
        args = arg_string.split(",")
        args = [strip_str(arg) for arg in args if strip_str(arg)]
        try:
            default_config = {k: copy(DATASET[name][k]) for k in args}
        except:
            error_message = f"{args} contain non-legal keys for {cls}!, expect in {list(DATASET[name])}"
            raise KeyError(error_message)
        for k in user_config.keys():
            assert k in args, f"Unexpected key [{k}] for {cls}, expected in {args}"
        default_config.update(user_config)

        instan = object.__new__(cls)
        instan._dataset_name = name
        instan.__init__(**default_config)
        return instan

    @abstractmethod
    def batch_describe(self) -> Dict:
        """return the data description of this dataset
        for example:
            {
                "X": (torch.float32, (-1, 224, 224, 3)),
                "Y": (torch.float32, (-1, 965)),
            }
        :raise: NotImplementedError

        :return: a dict of each data field
        """
        raise NotImplementedError

    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self) -> Generator:
        raise NotImplementedError

    @abstractmethod
    def switch_mode(self, mode):
        raise NotImplementedError

    @property
    def dataset_name(self):
        if hasattr(self, "_dataset_name"):
            return self._dataset_name
        return "Unknown"

    def help_info(self, **kwargs) -> str:
        batch_des = self.batch_describe()
        assert isinstance(batch_des, dict), "Wrong return type of batch_describe"

        headers = [["name", "type", "shape"]]
        for k, v in batch_des.items():
            assert len(v) == 2
            headers.append((k, str(v[0]), str(v[1])))
        return BaseDataset.fmt_tab(headers)
