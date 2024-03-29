from ..base import BaseModel
from abc import abstractmethod


class BaseDefender(BaseModel):
    @classmethod
    def from_config(cls, name, arg_string, user_args, user_config):
        return super().from_config("defender", name, arg_string, user_args, user_config)
    
    @abstractmethod
    def defense_step(self, **kwargs):
        raise NotImplementedError
