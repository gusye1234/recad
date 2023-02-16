import sys

sys.path.append("..")
import reck
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run reck")
    parser.add_argument(
        '--data', type=str, default="ml1m", choices=['ml1m', 'yelp', 'game']
    )
    parser.add_argument(
        '--victim', type=str, default="lightgcn", choices=['lightgcn', 'mf', 'ncf']
    )
    parser.add_argument('--attack', type=str, default="random")
    parser.add_argument('--rec_epoch', type=int, default=100)
    parser.add_argument('--attack_epoch', type=int, default=100)
    return parser.parse_args()


ARG = parse_args()
print("Receiving", ARG)

dataset_name = ARG.data

attack_data = reck.dataset.from_config("explicit", dataset_name).partial_sample(
    user_ratio=0.2
)
# attack_data.print_help()
data = reck.dataset.from_config(
    "implicit", dataset_name, need_graph=True, if_cache=True
)
# data.print_help()

rec_model = reck.model.from_config("victim", ARG.victim, dataset=data)

attack_model = reck.model.from_config("attacker", ARG.attack, dataset=attack_data)

config = {
    "victim_data": data,
    "attack_data": attack_data,
    "victim": rec_model,
    "attacker": attack_model,
    "rec_epoch": ARG.rec_epoch,
    "attack_epoch": ARG.attack_epoch,
}
workflow = reck.workflow.Normal.from_config(**config)

workflow.execute()
