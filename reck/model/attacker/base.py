from ..base import BaseModel
from abc import abstractmethod


class BaseAttacker(BaseModel):
    @classmethod
    def from_config(cls, name, arg_string, user_args, user_config):
        return super().from_config("attacker", name, arg_string, user_args, user_config)

    @abstractmethod
    def generate_fake(self, **kwargs):
        raise NotImplementedError
