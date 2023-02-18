"""dataset's returned information should have the common name:
- n_users: the number of the users in dataset
- n_items: the number of the items in dataset
- train_interactions: the interactions number in the dataste
TODO: finish the common fields
.. note::
    _description_
"""
from . import explicit, implicit
from .base import BaseData


factories = {"implicit": implicit.ImplicitData, "explicit": explicit.ExplicitData}


def from_config(scope, *args, **kwargs):
    return factories[scope].from_config(*args, **kwargs)
