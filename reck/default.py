from .utils import root_path, get_logger, check_path_or_empty, dict2list_table
from tabulate import tabulate

_logger = get_logger(__name__)

DATA_ROOT = root_path()
_logger.info(f"data dir located at {DATA_ROOT}")

SEED = 2023

DATASET = {
    name: {
        "path_train": check_path_or_empty(DATA_ROOT, "data", name, f"{name}_train.dat"),
        "path_test": check_path_or_empty(DATA_ROOT, "data", name, f"{name}_test.dat"),
        "selected_path": check_path_or_empty(
            DATA_ROOT, "data", name, f"{name}_selected_items"
        ),
        "target_path": check_path_or_empty(
            DATA_ROOT, "data", name, f"{name}_target_users"
        ),
        "verbose": False,
        "header": None,
        "sep": "\t",
        "threshold": 4,
        "batch_size": 500,
        "keep_rate_net": 1,
    }
    for name in ['ml100k', 'filmTrust', 'automotive']
}


def data_help_info(name):
    assert name in DATASET, f"dataset {name} is not on the list"
    print(tabulate(dict2list_table(DATASET[name])))
