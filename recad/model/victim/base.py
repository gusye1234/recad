from ..base import BaseModel


class BaseVictim(BaseModel):
    @classmethod
    def from_config(cls, name, arg_string, user_args, user_config):
        return super().from_config("victim", name, arg_string, user_args, user_config)
