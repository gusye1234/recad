from . import victim, attacker

factories = {
    "victim": {"lightgcn": victim.LightGCN},
    "attacker": {
        "random": attacker.RandomAttack,
        "average": attacker.AverageAttack,
        "segment": attacker.SegmentAttack,
        "bandwagon": attacker.BandwagonAttack,
    },
}


def from_config(scope, name, **kwargs):
    return factories[scope][name].from_config(**kwargs)
