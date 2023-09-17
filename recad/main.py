import argparse
from . import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run recad")
    parser.add_argument(
        '--data', type=str, default="ml1m", choices=list(default._Remote_Data_Urls)
    )
    parser.add_argument(
        '--victim', type=str, default="lightgcn", choices=['lightgcn', 'mf', 'ncf']
    )
    parser.add_argument('--attack', type=str, default="random")
    parser.add_argument('--rec_epoch', type=int, default=100)
    parser.add_argument('--attack_epoch', type=int, default=100)
    parser.add_argument('--attack_ratio', type=float, default=0.2)
    parser.add_argument('--rec_ratio', type=float, default=1)
    parser.add_argument('--tqdm', type=int, default=1)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--filler_num', type=int, default=36)
    return parser.parse_args()


def main():
    ARG = parse_args()
    print("Receiving", ARG)
    utils.TQDM = bool(ARG.tqdm)
    default.set_device_id(ARG.cuda_id)

    dataset_name = ARG.data
    sampling = "pointwise" if ARG.victim in ['mf', 'ncf'] else "pairwise"
    need_graph = ARG.victim in ['lightgcn']

    config = {
        "victim_data": dataset.from_config(
            "implicit",
            dataset_name,
            need_graph=need_graph,
            if_cache=True,
            sample=sampling,
        ).partial_sample(
            user_ratio=ARG.rec_ratio,
        ),
        "attack_data": dataset.from_config(
            "explicit", dataset_name, if_cache=True
        ).partial_sample(
            user_ratio=ARG.attack_ratio,
        ),
        "victim": model.from_config("victim", ARG.victim),
        "attacker": model.from_config(
            "attacker", ARG.attack, filler_num=ARG.filler_num
        ),
        "rec_epoch": ARG.rec_epoch,
        "attack_epoch": ARG.attack_epoch,
    }
    workflow_inst = workflow.from_config("no defense", **config)
    workflow_inst.execute()
