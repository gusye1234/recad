# adapt from https://github.com/XMUDM/ShillingAttack/tree/master/Leg-UP
from .base import BaseDataset
from ..default import DATASET
from ..utils import VarDim, get_logger
import torch
from copy import copy
import pandas as pd
import numpy as np
import scipy
from numbers import Number
from scipy.sparse import csr_matrix


class PreBuilt(BaseDataset):
    def __init__(
        self,
        name,
        path_train,
        path_test,
        header,
        sep,
        threshold,
        verbose,
        batch_size,
        keep_rate_net,
    ):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose
        self.batch_size = batch_size
        self.keep_rate_net = keep_rate_net
        self.logger = get_logger(__name__ + f":{name}")

        self._init()
        self.total_batch = (self.n_items + self.batch_size - 1) // self.batch_size
        self._mode = "train"
        if verbose:
            self.logger.info(f"Batch info:\n{self.help_info()}")

    def _init(self):
        (
            train_data_df,
            test_data_df,
            self.n_users,
            self.n_items,
        ) = self.load_file_as_dataFrame()
        self.__train_mat = self.dataFrame_to_matrix(
            train_data_df, self.n_users, self.n_items
        )
        self.__test_mat = self.dataFrame_to_matrix(
            test_data_df, self.n_users, self.n_items
        )
        self.__train_data_array = self.__train_mat.toarray()
        self.__train_data_array_mask = scipy.sign(self.__train_data_array)

    @classmethod
    def from_config(cls, name, **kwargs):
        assert name in DATASET, f"{name} is not on the pre-built datasets"
        config = {
            k: copy(DATASET[name][k])
            for k in [
                'path_train',
                'path_test',
                'header',
                'sep',
                'threshold',
                'verbose',
                'batch_size',
                'keep_rate_net',
            ]
        }
        config['name'] = name
        return super().from_config(config, kwargs)

    def switch_mode(self, mode):
        assert mode.lower() in ['train', "test"]
        self._mode = mode

    def mode(self) -> str:
        return self._mode

    def generate_batch(self):
        if self.mode() == "train":
            idxs = np.random.permutation(self.n_items)  # shuffled ordering
            for i in range(self.total_batch):
                batch_set_idx = idxs[i * self.batch_size : (i + 1) * self.batch_size]
                rating_mat = self.__train_data_array[:, batch_set_idx]
                rating_mask = self.__train_data_array_mask[:, batch_set_idx]
                yield {
                    "rating_mat": torch.from_numpy(rating_mat),
                    "rating_mask": torch.from_numpy(rating_mask),
                    "keep_rate_net": torch.scalar_tensor(self.keep_rate_net),
                }

    def batch_describe(self):
        return {
            "rating_mat": (torch.float64, (self.n_users, VarDim(max=self.batch_size))),
            "rating_mask": (torch.float64, (self.n_users, VarDim(max=self.batch_size))),
            "keep_rate_net": (torch.float32, ()),
        }

    def load_file_as_dataFrame(self):
        # load data to pandas dataframe
        if self.verbose:
            self.logger.info("load data from %s ..." % self.path_train)

        train_data = pd.read_csv(
            self.path_train, sep=self.sep, names=self.header, engine='python'
        )
        train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]

        if self.verbose:
            self.logger.info("load data from %s ..." % self.path_test)
        test_data = pd.read_csv(
            self.path_test, sep=self.sep, names=self.header, engine='python'
        ).loc[:, ['user_id', 'item_id', 'rating']]
        test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]

        # data statics

        n_users = (
            max(max(test_data.user_id.unique()), max(train_data.user_id.unique())) + 1
        )
        n_items = (
            max(max(test_data.item_id.unique()), max(train_data.item_id.unique())) + 1
        )

        if self.verbose:
            self.logger.info(
                "Number of users : %d , Number of items : %d. " % (n_users, n_items),
            )
            self.logger.info(
                "Train size : %d , Test size : %d. "
                % (train_data.shape[0], test_data.shape[0])
            )

        return train_data, test_data, n_users, n_items

    def dataFrame_to_matrix(self, data_frame, n_users, n_items):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items))
        # matrix_implicit = csr_matrix(
        #     (implicit_rating, (row, col)), shape=(n_users, n_items)
        # )
        # return matrix, matrix_implicit
        return matrix
