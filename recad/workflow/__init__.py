from .normal import Normal
from .defense import Defense

factories = {"no defense": Normal, "defense": Defense}

def from_config(name, **kwargs):
    return factories[name].from_config(**kwargs)
