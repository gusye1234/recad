import sys

sys.path.append("..")
import reck
import logging

dataset_name = 'ml1m'

# attack_data.print_help()
data = reck.dataset.from_config(
    "implicit", dataset_name, need_graph=False, if_cache=True, sample='pointwise'
)

attack_data = reck.dataset.from_config("explicit", dataset_name).partial_sample(
    user_ratio=0.2
)
data.print_help()

rec_model = reck.model.from_config("victim", "ncf", dataset=data)

attack_model = reck.model.from_config("attacker", "random", dataset=attack_data)

config = {
    "victim_data": data,
    "attack_data": attack_data,
    "victim": rec_model,
    "attacker": attack_model,
    "rec_epoch": 1,
}
workflow = reck.workflow.Normal.from_config(**config)

workflow.execute()
