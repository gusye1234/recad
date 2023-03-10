import os
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

DATA_ROOT = os.environ.get("RECAD_DIR", root_path())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
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
    "implicit": {
        name: {
            "path_train": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_train.csv"
            ),
            "path_valid": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_valid.csv"
            ),
            "path_test": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_test.csv"
            ),
            "test_batch_size": 400,
            "A_split": False,
            "A_n_fold": 100,
            "pairwise_batch_size": 1024,
            "pointwise_batch_size": 1024,
            "sample": "pairwise",
            "negative_ratio": 4,
            "need_graph": True,
            "rating_filter": 4,
        }
        for name in ['ml1m', 'yelp', "game", "dev"]
    },
    "explicit": {
        name: {
            "path_train": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_train.csv"
            ),
            "path_test": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_test.csv"
            ),
            "path_valid": check_path_or_empty(
                DATA_ROOT, "data", name, f"{name}_test.csv"
            ),
            "batch_size": 256,
            'header': None,
            'sep': ',',
            'threshold': 4,
            "sample": "row",
        }
        for name in ['ml1m', 'yelp', "game", "dev"]
    },
}

_decorate_config(DATASET, "logging_level", logging.INFO, 3)
_decorate_config(DATASET, "train_dict", None, 3)
_decorate_config(DATASET, "valid_dict", None, 3)
_decorate_config(DATASET, "test_dict", None, 3)
_decorate_config(DATASET, "remap_enable", False, 3)  # for partial sample
_decorate_config(DATASET, "device", DEVICE, 3)
_decorate_config(DATASET, "if_cache", False, 3)
_decorate_config(DATASET, "cache_dir", use_dir(".", "generated"), 3)


def dataset_print_help(name):
    for k, v in DATASET.items():
        if name in v:
            print(k)
            print(tabulate(dict2list_table(DATASET[k][name]), disable_numparse=True))


def print_datasets():
    print({k: list(v) for k, v in DATASET.items()})


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
            "optim": 'adam',
            "lr": 0.001,
        },
        "mf": {
            "factor_num": 3,
            "embedding_size": 128,
            "dropout": 0,
            "optim": 'adam',
            "lr": 0.001,
        },
        "ncf": {
            "factor_num": 32,
            "num_layers": 5,
            "dropout": 0,
            "model": "NeuMF-end",
            "GMF_model": None,
            "MLP_model": None,
            "optim": 'adam',
            "lr": 0.001,
        },
    },
    "attacker": {
        "random": {"attack_num": 50, "filler_num": 36},
        "average": {"attack_num": 50, "filler_num": 36},
        "segment": {
            "attack_num": 50,
            "filler_num": 36,
            "selected_ids": [
                1153,
                2201,
                1572,
                836,
                523,
                849,
                1171,
                344,
                857,
                1213,
                1535,
            ],
        },
        "bandwagon": {
            "attack_num": 50,
            "filler_num": 36,
            "selected_ids": [],
        },
        "aush": {
            "attack_num": 50,
            "filler_num": 36,
            "lr_g": 0.01,
            "lr_d": 0.001,
            "optim_g": 'adam',
            "optim_d": 'adam',
            "selected_ids": [62],
            "ZR_ratio": 0.2,
        },
        "aia": {
            "attack_num": 50,
            "filler_num": 36,
            "lr_g": 0.01,
            "lr_d": 0.001,
            "optim_g": 'adam',
            "optim_d": 'adam',
            "surrogate_model": "WMF",
            "epoch_s": 50,
            "unroll_steps_s": 1,
            "hidden_dim_s": 16,
            "lr_s": 1e-2,
            "weight_decay_s": 1e-5,
            "batch_size_s": 16,
            "weight_pos_s": 1.0,
            "weight_neg_s": 0.0,
            "selected_ids": [62],
        },
        "aushplus": {
            "attack_num": 50,
            "pretrain_epoch_g": 1,
            "pretrain_epoch_d": 5,
            "epoch_gan_d": 5,
            "epoch_gan_g": 1,
            "epoch_surrogate": 50,
            "filler_num": 36,
            "lr_g": 0.01,
            "lr_d": 0.001,
            "optim_g": 'adam',
            "optim_d": 'adam',
            "surrogate_model": "WMF",
            "epoch_s": 50,
            "unroll_steps_s": 1,
            "hidden_dim_s": 16,
            "lr_s": 1e-2,
            "weight_decay_s": 1e-5,
            "batch_size_s": 16,
            "weight_pos_s": 1.0,
            "weight_neg_s": 0.0,
            "selected_ids": [62],
        },
    },
}

_decorate_config(MODEL, "logging_level", logging.INFO, 3)
_decorate_config(MODEL, "device", DEVICE, 3)


def model_print_help(name):
    for k, v in MODEL.items():
        if name in v:
            print(k)
            print(tabulate(dict2list_table(DATASET[k][name]), disable_numparse=True))


def print_models():
    print({k: list(v) for k, v in MODEL.items()})


WORKFLOW = {
    "normal": {
        "rec_epoch": 400,
        "attack_epoch": 100,
        "target_id_list": [
            0,
        ],
        "filter_num": 4,
        "topks": [10, 20, 50, 100],
    }
}

_decorate_config(WORKFLOW, "logging_level", logging.INFO, 2)
_decorate_config(WORKFLOW, "device", DEVICE, 2)
_decorate_config(WORKFLOW, "cache_dir", use_dir(".", "workflows_results"), 2)


def set_device_id(cuda_id):
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else "cpu")
    # TODO
    _decorate_config(MODEL, "device", device, 3)
    _decorate_config(WORKFLOW, "device", device, 2)
    _decorate_config(DATASET, "device", device, 3)
