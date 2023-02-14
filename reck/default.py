from .utils import (
    root_path,
    get_logger,
    check_path_or_empty,
    dict2list_table,
    check_dir_or_make,
    use_dir,
)
import torch
from tabulate import tabulate
import logging

_logger = get_logger(__name__)

DATA_ROOT = root_path()
_logger.info(f"data dir located at {DATA_ROOT}")
SEED = 2023


def _decorate_config(D, k, v, depth):
    if depth == 1:
        D[k] = v
        return
    else:
        for p in D.values():
            _decorate_config(p, k, v, depth - 1)


DATASET = {
    "victim_dataset": {
        name: {
            "path_train": check_path_or_empty(
                DATA_ROOT, "data", name, "training_dict.npy"
            ),
            "path_valid": check_path_or_empty(
                DATA_ROOT, "data", name, "validation_dict.npy"
            ),
            "path_test": check_path_or_empty(
                DATA_ROOT, "data", name, "testing_dict.npy"
            ),
            "bpr_batch_size": 2048,
            "test_batch_size": 400,
            "A_split": False,
            "A_n_fold": 100,
            "device": torch.device('cuda' if torch.cuda.is_available() else "cpu"),
            "sample": "bpr",
            "need_graph": True,
        }
        for name in ['ml1m', 'yelp']
    },
    "attack_dataset": {
        name: {
            "path_train": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_partial_train.csv"
            ),
            "path_test": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_partial_test.csv"
            ),
            # "path_full"
            'header': None,
            'sep': ',',
            'threshold': 4,
            'verbose': False,
        }
        for name in ['ml1m', 'yelp']
    },
}

_decorate_config(DATASET, "logging_level", logging.INFO, 3)
_decorate_config(DATASET, "train_dict", None, 3)
_decorate_config(DATASET, "if_cache", False, 3)
_decorate_config(DATASET, "cache_dir", use_dir(".", "generated"), 3)


def dataset_print_help(name):
    found = None
    for k, v in DATASET.items():
        if name in v:
            found = k
            break
    assert found is not None, f"dataset {name} is not on the list"
    print(tabulate(dict2list_table(DATASET[found][name]), disable_numparse=True))


def show_datasets():
    return list(DATASET)


MODEL = {
    "victim": {
        "lightgcn": {
            "latent_dim_rec": 128,
            "lightGCN_n_layers": 3,
            "A_split": False,
            "pretrain": False,
            "keep_prob": 0.6,
            "dropout": 0.0,
            "lambda": 0.0001,
        }
    },
    "attacker": {"random": {"attack_num": 50, "filler_num": 36}},
}

_decorate_config(MODEL, "logging_level", logging.INFO, 3)


def model_print_help(scope, name):
    assert name in MODEL[scope], f"model {name} is not on the {scope} models"
    print(dict2list_table(MODEL[scope][name]))
    print(tabulate(dict2list_table(MODEL[scope][name]), disable_numparse=True))


def show_models():
    return list(MODEL)


WORKFLOW = {
    "normal": {
        "cache_dir": check_dir_or_make(".", "workflows_results", "normal"),
        "rec_epoch": 400,
        "rec_lr": 0.001,
        "rec_optim": "adam",
        "attack_epoch": 100,
        "attack_lr": 0.001,
        "attack_optim": "adam",
        "target_items": [
            0,
        ],
        "filter_num": 4,
    }
}

_decorate_config(WORKFLOW, "logging_level", logging.INFO, 2)
