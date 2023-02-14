from tabulate import tabulate
from functools import partial
from typing import Dict, Generator
from abc import ABC, abstractmethod
from collections.abc import Iterable
from ..default import DATASET
from ..utils import strip_str, type_if_long, parse_args, get_logger
from copy import copy

logger = get_logger(__name__)


class BaseDataset(ABC):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    @abstractmethod
    def from_config(cls, scope, name, arg_string, user_config):
        assert name in DATASET[scope], f"{name} is not on the default datasets"
        args = parse_args(arg_string)
        try:
            default_config = {k: copy(DATASET[scope][name][k]) for k in args}
        except:
            error_message = f"{args} contain non-legal keys for {cls}!, expect in {list(DATASET[scope][name])}"
            raise KeyError(error_message)
        for k in user_config.keys():
            if k not in args:
                logger.debug(f"Unexpected key [{k}] for {cls}, expected in {args}")
        default_config.update(user_config)

        instan = object.__new__(cls)
        instan._dataset_name = name
        instan._init_config = default_config
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
    def info_describe(self) -> Dict:
        """return the information description of this dataset
        for example:
            {
                "n_users": 1000,
                "n_items": 500,
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

    @abstractmethod
    def inject_data(self, mode, data):
        """inject the attack data

        :param mode: in which mode, default train mode
        :param data: array(fake_users X n_items), each element is a rating

        :raise: _description_

        .. note::
            _description_
        """
        raise NotImplementedError

    @property
    def dataset_name(self):
        if hasattr(self, "_dataset_name"):
            return self._dataset_name
        return type(self).__name__

    def reset(self, **kwargs):
        # inject may change some files
        if hasattr(self, "_init_config"):
            config = copy(self._init_config)
            for k, v in kwargs.items():
                if k in config:
                    config[k] = v
                else:
                    raise ValueError(f"reset arg {k} should be in {list(config)}")
            return type(self).from_config(self.dataset_name, **config)
        raise ValueError("reset method is only for datasets instantiated from_config")

    def print_help(self, **kwargs) -> str:
        info_des = self.info_describe()
        assert isinstance(
            info_des, dict
        ), "Wrong return type of info_describe, expected Dict"
        cols = [(k, type_if_long(v)) for k, v in info_des.items()]
        print("Information:")
        print(BaseDataset.fmt_tab(cols))

        batch_des = self.batch_describe()
        assert isinstance(
            batch_des, dict
        ), "Wrong return type of batch_describe, expected Dict"

        headers = [["name", "type", "shape"]]
        for k, v in batch_des.items():
            assert len(v) == 2
            headers.append((k, str(v[0]), str(v[1])))
        print("Batch data:")
        print(BaseDataset.fmt_tab(headers))
