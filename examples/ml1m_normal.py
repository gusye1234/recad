from torch.optim import Adam
import sys

sys.path.append("..")
import reck
import logging

dataset_name = 'ml1m'

data = reck.dataset.from_config("implicit", dataset_name, need_graph=True)
data.print_help()
attack_data = reck.dataset.from_config("explicit", dataset_name).partial_sample(
    user_ratio=0.2
)
attack_data.print_help()

rec_model = reck.model.from_config("victim", "lightgcn", dataset=data)

attack_model = reck.model.from_config("attacker", "average", dataset=attack_data)

config = {
    "victim_data": data,
    "attack_data": attack_data,
    "victim": rec_model,
    "attacker": attack_model,
    "rec_epoch": 0,
}
workflow = reck.workflow.Normal.from_config(**config)

workflow.execute()
