from .base import BaseDataset
import torch
import pandas as pd
from ..utils import get_logger, VarDim
from ..default import DATASET
from scipy.sparse import csr_matrix


class CSVDataset(BaseDataset):
    def __init__(
        self,
        path_train,
        path_test,
        header,
        sep,
        threshold,
        verbose,
        logging_level,
        **kwargs,
    ):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose
        self.logger = get_logger(f"{__name__}:{self.dataset_name}", level=logging_level)

        self._mode = "train"

        (
            self.train_data_df,
            self.test_data_df,
            self.n_users,
            self.n_items,
        ) = self.load_file_as_dataFrame()

    def load_file_as_dataFrame(self):
        # load data to pandas dataframe
        self.logger.debug("\nload data from %s ..." % self.path_train)

        train_data = pd.read_csv(self.path_train, engine='python', sep=self.sep)
        train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]

        self.logger.debug("load data from %s ..." % self.path_test)
        test_data = pd.read_csv(self.path_test, engine='python', sep=self.sep).loc[
            :, ['user_id', 'item_id', 'rating']
        ]
        test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]
        # data statics

        n_users = (
            max(max(test_data.user_id.unique()), max(train_data.user_id.unique())) + 1
        )
        n_items = (
            max(max(test_data.item_id.unique()), max(train_data.item_id.unique())) + 1
        )

        self.logger.debug(
            "Number of users : %d , Number of items : %d. " % (n_users, n_items)
        )
        self.logger.debug(
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
        matrix_implicit = csr_matrix(
            (implicit_rating, (row, col)), shape=(n_users, n_items)
        )
        return matrix, matrix_implicit

    @classmethod
    def from_config(cls, name, **user_config):
        args = list(DATASET['attack_dataset'][name])
        return super().from_config("attack_dataset", name, args, user_config)

    def batch_describe(self):
        return {}

    def info_describe(self):
        infos = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "train_df": self.train_data_df,
            "test_df": self.test_data_df,
        }
        return infos

    def mode(self) -> str:
        return self._mode

    def generate_batch(self):
        pass

    def switch_mode(self, mode):
        assert mode in ["train", "test", "validate"]
        self._mode = mode

    def inject_data(self, mode, data):
        return super().inject_data(mode, data)
