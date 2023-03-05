import sys

sys.path.append("..")
import recad
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run recad")
    parser.add_argument(
        '--data', type=str, default="ml1m", choices=['ml1m', 'yelp', 'game']
    )
    parser.add_argument(
        '--victim', type=str, default="lightgcn", choices=['lightgcn', 'mf', 'ncf']
    )
    parser.add_argument('--attack', type=str, default="random")
    parser.add_argument('--rec_epoch', type=int, default=100)
    parser.add_argument('--attack_epoch', type=int, default=100)
    parser.add_argument('--attack_ratio', type=float, default=0.2)
    parser.add_argument('--tqdm', type=int, default=1)
    parser.add_argument('--cuda_id', type=int, default=0)
    return parser.parse_args()


ARG = parse_args()
print("Receiving", ARG)
recad.utils.TQDM = bool(ARG.tqdm)
recad.default.set_device_id(ARG.cuda_id)

dataset_name = ARG.data
sampling = "pointwise" if ARG.victim in ['mf', 'ncf'] else "pairwise"
need_graph = ARG.victim in ['lightgcn']

config = {
    "victim_data": recad.dataset.from_config(
        "implicit", dataset_name, need_graph=need_graph, if_cache=True, sample=sampling
    ),
    "attack_data": recad.dataset.from_config(
        "explicit", dataset_name, if_cache=True
    ).partial_sample(
        user_ratio=ARG.attack_ratio,
    ),
    "victim": recad.model.from_config("victim", ARG.victim),
    "attacker": recad.model.from_config("attacker", ARG.attack),
    "rec_epoch": ARG.rec_epoch,
    "attack_epoch": ARG.attack_epoch,
}
workflow = recad.workflow.from_config("no defense", **config)
workflow.execute()
