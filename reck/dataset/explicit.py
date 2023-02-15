import torch
from .base import BaseDataset
import numpy as np
import pandas as pd
from ..utils import get_logger, VarDim
from ..default import DATASET
from scipy.sparse import csr_matrix


class ExplicitData(BaseDataset):
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

        if self.config['train_dict'] is None:
            self.train_dict = self.load_file_as_dataFrame(self.path_train)
        else:
            self.train_dict = self.config['train_dict']

        if self.config['test_dict'] is None:
            self.test_dict = self.load_file_as_dataFrame(self.path_test)
        else:
            self.test_dict = self.config['test_dict']

        if self.config['valid_dict'] is None:
            self.valid_dict = self.load_file_as_dataFrame(self.path_valid)
        else:
            self.valid_dict = self.config['valid_dict']
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

        self.train_size = len(self.train_dict)
        self.valid_size = len(self.valid_dict)
        self.test_size = len(self.test_dict)

        self.train_mat = self.to_matrix(self.train_dict, self.n_users, self.n_items)

    def load_file_as_dataFrame(self, file):
        self.logger.debug("\nload data from %s ..." % file)

        train_data = pd.read_csv(file, engine='python', sep=self.sep)
        train_data = train_data.loc[:, self.header]

        # self.n_users = max(self.n_users, max(train_data.user_id.unique()))
        # self.n_items = max(self.n_items, max(train_data.item_id.unique()))

        train_data = train_data.to_numpy()
        return train_data

    def to_matrix(self, kv_array, n_users, n_items):
        row, col, rating = kv_array[:, 0], kv_array[:, 1], kv_array[:, 2]
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
        }
        if self.remap_enable:
            infos['user_map'] = self.user_map
        return infos

    def mode(self) -> str:
        return self._mode

    def generate_batch(self, **config):
        if self.mode() == 'train':
            filler_num = config['filler_num']
            selected_ids = config['selected_ids']
            target_id_list = config['target_id_list']
            batch_size = self.config['batch_size']
            mask_array = (self.train_mat > 0).astype(np.float)
            mask_array[:, selected_ids + target_id_list] = 0
            available_idx = np.where(np.sum(mask_array, 1) >= filler_num)[0]
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

    def partial_sample(self, **kwargs) -> 'BaseDataset':
        assert "user_ratio" in kwargs, "Expect to have [user_ratio]"
        user_ratio = kwargs['user_ratio']

        users = np.unique(self.train_dict[:, 0])
        np.random.shuffle(users)
        left_users = users[: int(len(users) * user_ratio)]

        left_index = np.zeros(len(self.train_dict), dtype=np.bool)
        for u in left_users:
            left_index = np.logical_or(left_index, self.train_dict[:, 0] == u)
        new_train_dict = self.train_dict[left_index]

        left_index = np.zeros(len(self.test_dict), dtype=np.bool)
        for u in left_users:
            left_index = np.logical_or(left_index, self.test_dict[:, 0] == u)
        new_test_dict = self.test_dict[left_index]

        left_index = np.zeros(len(self.valid_dict), dtype=np.bool)
        for u in left_users:
            left_index = np.logical_or(left_index, self.valid_dict[:, 0] == u)
        new_valid_dict = self.valid_dict[left_index]

        return self.reset(
            train_dict=new_train_dict,
            test_dict=new_test_dict,
            valid_dict=new_valid_dict,
            remap_enable=True,
        )
