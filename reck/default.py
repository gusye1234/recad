from .utils import root_path, get_logger, check_path_or_empty

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
        "verbose" : False,
        "header" : None,
        "sep" : "\t",
        "threshold" : 4

    }
    for name in ['ml100k', 'filmTrust', 'automotive']
}
