import torch
from .base import BaseData
import numpy as np
import pandas as pd
import os
from os.path import join
from ..utils import get_logger, VarDim, check_dir_or_make
from ..default import DATASET
from scipy.sparse import csr_matrix


class ExplicitData(BaseData):
    def __init__(
        self,
        path_train,
        path_test,
        path_valid,
        header,
        sep,
        threshold,
        logging_level,
        **config,
    ):
        self.path_train = path_train
        self.path_test = path_test
        self.path_valid = path_valid
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.logger = get_logger(f"{__name__}:{self.dataset_name}", level=logging_level)
        self._mode = "train"
        self.config = config
        self.remap_enable = config['remap_enable']
        self._load_data()

    def remap_kvr(self, kvr):
        for i in range(len(kvr)):
            kvr[i, 0] = self.user_map[kvr[i, 0]]
        return kvr

    def _load_data(self):
        self.n_users = 0
        self.n_items = 0

        for attr, default in zip(
            ['train_dict', 'valid_dict', 'test_dict'],
            [self.path_train, self.path_valid, self.path_test],
        ):
            if self.config[attr] is not None:
                self.logger.debug(f"Init {attr} from a passed dict")
                setattr(self, attr, self.config[attr])
                continue
            maybe_cache = join(
                self.config['cache_dir'],
                f"{self.dataset_name}_explicit_{attr}.npy",
            )
            if os.path.exists(maybe_cache) and self.config['if_cache']:
                self.logger.info(f"loading cached {attr}")
                setattr(self, attr, np.load(maybe_cache))
            else:
                setattr(self, attr, self.load_file_as_np(default))
                check_dir_or_make(self.config['cache_dir'])
                if self.config['if_cache']:
                    np.save(maybe_cache, getattr(self, attr))

        if self.remap_enable:
            unique_user = np.unique(
                np.concatenate(
                    [self.train_dict[:, 0], self.test_dict[:, 0], self.valid_dict[:, 0]]
                )
            )
            self.user_map = {unique_user[i]: i for i in range(len(unique_user))}
            self.train_dict = self.remap_kvr(self.train_dict)
            self.test_dict = self.remap_kvr(self.test_dict)
            self.valid_dict = self.remap_kvr(self.valid_dict)

        self.n_users = (
            max(
                self.train_dict[:, 0].max(),
                self.valid_dict[:, 0].max(),
                self.test_dict[:, 0].max(),
            )
            + 1
        )
        self.n_items = (
            max(
                self.train_dict[:, 1].max(),
                self.valid_dict[:, 1].max(),
                self.test_dict[:, 1].max(),
            )
            + 1
        )
        self.n_users = int(self.n_users)
        self.n_items = int(self.n_items)

        self.train_size = len(self.train_dict)
        self.valid_size = len(self.valid_dict)
        self.test_size = len(self.test_dict)

        self.train_mat = self.to_matrix(self.train_dict, self.n_users, self.n_items)

    def load_file_as_np(self, file):
        self.logger.debug("\nload data from %s ..." % file)

        train_data = pd.read_csv(file, engine='python', sep=self.sep)
        train_data = train_data.loc[:, self.header]

        # self.n_users = max(self.n_users, max(train_data.user_id.unique()))
        # self.n_items = max(self.n_items, max(train_data.item_id.unique()))

        train_data = train_data.to_numpy()
        return train_data

    def to_matrix(self, kv_array, n_users, n_items):
        row, col, rating = kv_array[:, 0], kv_array[:, 1], kv_array[:, 2]
        row = row.astype("int64")
        col = col.astype("int64")
        rating = rating.astype("float32")
        # implicit_rating = (rating >= self.threshold).astype("int")

        matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items)).toarray()
        return matrix

    @classmethod
    def from_config(cls, name, **user_config):
        args = list(DATASET['explicit'][name])
        return super().from_config("explicit", name, args, user_config)

    def batch_describe(self):
        if self.mode() == 'train':
            return {
                'users': (
                    torch.int64,
                    (VarDim(max=self.config['batch_size'], comment="batch size")),
                ),
                'users_mat': (
                    torch.float32,
                    (
                        VarDim(max=self.config['batch_size'], comment="batch size"),
                        self.n_items,
                    ),
                ),
            }

    def info_describe(self):
        infos = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "train_interactions": self.train_size,
            "valid_interactions": self.valid_size,
            "test_interactions": self.test_size,
            "train_kvr": self.train_dict,
            "train_mat": self.train_mat,
            "batch_describe": self.batch_describe(),
        }
        if self.remap_enable:
            infos['user_map'] = self.user_map
        return infos

    def mode(self) -> str:
        return self._mode

    def generate_batch(self, **config):
        user_filter = config.get("user_filter", None)
        if self.mode() == 'train':
            batch_size = self.config['batch_size']
            if user_filter is not None:
                available_idx = user_filter(train_mat=self.train_mat)
            else:
                available_idx = list(range(len(self.train_mat)))
            available_idx = np.random.permutation(available_idx)
            total_batch = (len(available_idx) + batch_size - 1) // batch_size
            for b in range(total_batch):
                batch_set_idx = available_idx[b * batch_size : (b + 1) * batch_size]
                real_profiles = self.train_mat[batch_set_idx, :].astype('float')

                yield {
                    "users": torch.tensor(batch_set_idx, dtype=torch.int64).to(
                        self.config['device']
                    ),
                    "users_mat": torch.tensor(real_profiles, dtype=torch.float32).to(
                        self.config['device']
                    ),
                }

    def switch_mode(self, mode):
        assert mode in ["train", "test", "validate"]
        self._mode = mode

    def inject_data(self, mode, data):
        # TODO add injection for explicit data
        return super().inject_data(mode, data)
    
    def delete_data(self, mode, user_id, data):
        # TODO add delete for explicit data
        return super().delete_data(mode, user_id, data)

    def partial_sample(self, **kwargs) -> 'BaseData':
        assert "user_ratio" in kwargs, "Expect to have [user_ratio]"
        user_ratio = kwargs['user_ratio']

        users = np.unique(self.train_dict[:, 0])
        np.random.shuffle(users)
        left_users = users[: int(len(users) * user_ratio)]

        left_index = np.zeros(len(self.train_dict), dtype="bool")
        for u in left_users:
            left_index = np.logical_or(left_index, self.train_dict[:, 0] == u)
        new_train_dict = self.train_dict[left_index]

        left_index = np.zeros(len(self.test_dict), dtype="bool")
        for u in left_users:
            left_index = np.logical_or(left_index, self.test_dict[:, 0] == u)
        new_test_dict = self.test_dict[left_index]

        left_index = np.zeros(len(self.valid_dict), dtype="bool")
        for u in left_users:
            left_index = np.logical_or(left_index, self.valid_dict[:, 0] == u)
        new_valid_dict = self.valid_dict[left_index]

        return self.reset(
            train_dict=new_train_dict,
            test_dict=new_test_dict,
            valid_dict=new_valid_dict,
            remap_enable=True,
            if_cache=False,
        )
