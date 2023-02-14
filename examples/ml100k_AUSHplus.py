from torch.optim import Adam
import sys

sys.path.append("..")
import reck
import logging

dataset_name = 'ml1m'

data = reck.dataset.victim_dataset.NPYDataset.from_config(
    dataset_name, need_graph=True, logging_level=logging.DEBUG, if_cache=True
)
data.print_help()
attack_data = reck.dataset.attack_dataset.CSVDataset.from_config(dataset_name)

rec_model = reck.model.victim.LightGCN.from_config(dataset=data)

attack_model = reck.model.attacker.RandomAttack.from_config(dataset=attack_data)


config = {
    "victim_data": data,
    "attack_data": attack_data,
    "victim": rec_model,
    "attacker": attack_model,
    "rec_epoch": 1,
}
workflow = reck.workflow.Normal.from_config(**config, logging_level=logging.DEBUG)

workflow.execute()
