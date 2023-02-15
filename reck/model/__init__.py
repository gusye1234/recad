from . import victim, attacker

factories = {
    "victim": {"lightgcn": victim.LightGCN},
    "attacker": {"random": attacker.RandomAttack, "average": attacker.AverageAttack},
}


def from_config(scope, name, **kwargs):
    return factories[scope][name].from_config(**kwargs)
