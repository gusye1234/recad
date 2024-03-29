from . import victim, attacker, defense

factories = {
    "victim": {"lightgcn": victim.LightGCN, 'mf': victim.MF, 'ncf': victim.NCF},
    "attacker": {
        "random": attacker.RandomAttack,
        "average": attacker.AverageAttack,
        "segment": attacker.SegmentAttack,
        "bandwagon": attacker.BandwagonAttack,
        "aush": attacker.Aush,
        "aia": attacker.AIA,
        "aushplus": attacker.AushPlus,
        "uba": attacker.UBA,
    },
    "defender":{
        "PCASelectUsers":defense.PCASelectUsers
    }
}

def from_config(scope, name, **kwargs):
    return factories[scope][name].from_config(**kwargs)
