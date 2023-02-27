from .normal import Normal


factories = {"no defense": Normal}


def from_config(name, **kwargs):
    return factories[name].from_config(**kwargs)
