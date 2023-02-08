from tabulate import tabulate
from functools import partial
from abc import ABC, abstractmethod
from torch.nn import Module

class BaseModel(Module, ABC):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    @abstractmethod
    def from_config(cls, default_config, user_config):
        for k in user_config.keys():
            assert k in default_config, f"Unexpected key [{k}] for {cls}"
        default_config.update(user_config)
        return cls(**default_config)
    
    @abstractmethod
    def batch_describe(self):
        raise NotImplementedError
    