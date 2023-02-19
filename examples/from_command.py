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

config = {
    "victim_data": reck.dataset.from_config(
        "implicit", dataset_name, need_graph=True, if_cache=True
    ),
    "attack_data": reck.dataset.from_config(
        "explicit", dataset_name, if_cache=True
    ).partial_sample(
        user_ratio=0.2,
    ),
    "victim": reck.model.from_config("victim", ARG.victim),
    "attacker": reck.model.from_config("attacker", ARG.attack),
    "rec_epoch": ARG.rec_epoch,
    "attack_epoch": ARG.attack_epoch,
}
workflow = reck.workflow.Normal.from_config(**config)
workflow.print_help()
workflow.execute()
