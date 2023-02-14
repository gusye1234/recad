# adapt from https://github.com/XMUDM/ShillingAttack/tree/master/Leg-UP
from .base import BaseDataset
from ..default import DATASET
from ..utils import VarDim, get_logger, check_dir_or_make
import torch
from time import time
from copy import copy
import pandas as pd
import numpy as np
from os.path import dirname, join
import scipy.sparse as sp
from numbers import Number
from scipy.sparse import csr_matrix
from collections import defaultdict


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have ' 'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def minibatch(*tensors, **kwargs):
    batch_size = kwargs['batch_size']

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = dataset.traindataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.0
    sample_time2 = 0.0
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.n_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
    return np.array(S)


def csv2dict(file, filter=4):
    df = pd.read_csv(file)
    interactions = {}
    for entry in df.to_list():
        user_id, item_id, rate = entry
        if rate < filter:
            continue
        if user_id not in interactions:
            interactions[int(user_id)] = []
        if item_id not in interactions[int(user_id)]:
            interactions[int(user_id)].append(int(item_id))


def fake_array2dict(fake_array, n_users, filter_num=4):
    assert len(fake_array.shape) == 2, "Expect a user-item 2D rating matrix"
    uids, iids = np.where(fake_array > filter_num)
    uids += n_users
    result = defaultdict(list)
    for u, i in zip(uids, iids):
        result[u].append(i)
    return result


def npy2dict(file):
    return np.load(file, allow_pickle=True).item()


def convert2dict(file: str):
    if file.endswith(".csv"):
        return csv2dict(file)
    elif file.endswith(".npy"):
        return npy2dict(file)


class NPYDataset(BaseDataset):
    def __init__(self, **config):
        self.config = config
        self.logger = get_logger(
            f"{__name__}:{self.dataset_name}", level=config['logging_level']
        )
        self._mode = "train"
        self._load_data()
        self._init_data()

    def _load_data(self):
        train_file = self.config['path_train']
        valid_file = self.config['path_valid']
        test_file = self.config['path_test']

        if self.config.get("train_dict", None) is None:
            self.train_dict = convert2dict(train_file)
        else:
            self.logger.debug("Init train dict from a passed dict")
            self.train_dict = self.config['train_dict']
        self.valid_dict = convert2dict(valid_file)
        self.test_dict = convert2dict(test_file)

    def _init_data(self):
        self.n_user = 0
        self.m_item = 0

        self.split = self.config['A_split']
        self.folds = self.config['A_n_fold']

        (
            self.traindataSize,
            self.trainUniqueUsers,
            self.trainUser,
            self.trainItem,
        ) = self.read_data(self.train_dict)

        (
            self.validDataSize,
            self.validUniqueUsers,
            self.validUser,
            self.validItem,
        ) = self.read_data(self.valid_dict)

        (
            self.testDataSize,
            self.testUniqueUsers,
            self.testUser,
            self.testItem,
        ) = self.read_data(self.test_dict)

        self.m_item += 1
        self.n_user += 1

        self.Graph = None
        self.logger.debug(f"{self.traindataSize} interactions for training")
        self.logger.debug(f"{self.validDataSize} interactions for validation")
        self.logger.debug(f"{self.testDataSize} interactions for testing")
        self.logger.debug(
            f"{self.dataset_name} Sparsity : {(self.traindataSize + self.validDataSize + self.testDataSize) / self.n_users / self.n_items}"
        )

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.logger.debug(f"{self.dataset_name} is ready to go")

        if self.config['need_graph']:
            self.getSparseGraph()

    def read_data(self, data_dict):
        trainUniqueUsers, trainItem, trainUser = [], [], []
        traindataSize = 0
        for uid in data_dict.keys():
            if len(data_dict[uid]) != 0:
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(data_dict[uid]))
                trainItem.extend(data_dict[uid])
                self.m_item = max(self.m_item, max(data_dict[uid]))
                self.n_user = max(self.n_user, uid)
                traindataSize += len(data_dict[uid])

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        return (
            traindataSize,
            np.array(trainUniqueUsers),
            np.array(trainUser),
            np.array(trainItem),
        )

    def getSparseGraph(self):
        self.logger.debug("loading adjacency matrix")
        if self.Graph is None:
            if self.config.get('adj_mat', None):
                pre_adj_mat = sp.load_npz(self.config['adj_mat'])
                self.logger.warning(
                    f"successfully loaded adj_mat from {self.config['adj_mat']}, this could cause the inconsistency of the dataset"
                )
                norm_adj = pre_adj_mat
            else:
                self.logger.debug("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.n_items, self.n_users + self.n_items),
                    dtype=np.float32,
                )
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[: self.n_users, self.n_users :] = R
                adj_mat[self.n_users :, : self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                self.logger.debug(f"costing {end - s}s, saved norm_mat...")
                if self.config['if_cache']:
                    self.logger.debug(f"cache adj mat to {self.config['cache_dir']}")
                    check_dir_or_make(self.config['cache_dir'])
                    sp.save_npz(
                        join(
                            self.config['cache_dir'],
                            f"adj_mat_{self.dataset_name}_{self.n_users}_{self.n_items}.npz",
                        ),
                        norm_adj,
                    )

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                self.logger.debug("split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.config['device'])
                self.logger.debug("don't split the matrix")
        return self.Graph

    @property
    def n_users(self):
        return self.n_user

    @property
    def n_items(self):
        return self.m_item

    def num_items(self):
        return self.m_item

    def trainDict(self):
        return self.train_dict

    @property
    def testDict(self):
        return self.test_dict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end])
                .coalesce()
                .to(self.config['device'])
            )
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserValidItems(self, users):
        validItems = []
        for user in users:
            if user in self.valid_dict:
                validItems.append(self.valid_dict[user])
        return validItems

    @classmethod
    def from_config(cls, name, **user_config):
        args = list(DATASET['victim_dataset'][name])
        return super().from_config("victim_dataset", name, args, user_config)

    def batch_describe(self):
        if self.config['sample'] == 'bpr' and self.mode() == "train":
            return {
                "users": (
                    torch.int64,
                    (VarDim(max=self.config['bpr_batch_size'], comment="batch")),
                ),
                "positive_items": (
                    torch.int64,
                    (VarDim(max=self.config['bpr_batch_size'], comment="batch")),
                ),
                "negative_items": (
                    torch.int64,
                    (VarDim(max=self.config['bpr_batch_size'], comment="batch")),
                ),
            }
        elif self.mode() in ["validate", "test"]:
            return {
                "users": (
                    torch.int64,
                    (VarDim(max=self.config['test_batch_size'], comment='batch')),
                ),
                "positive_items": (
                    list,
                    (VarDim(max=self.config['test_batch_size'], comment='batch')),
                ),
                "ground_truth": (
                    list,
                    (VarDim(max=self.config['test_batch_size'], comment='batch')),
                ),
            }

    def info_describe(self):
        infos = {
            "n_users": self.n_user,
            "n_items": self.m_item,
            "train_interactions": self.traindataSize,
            "valid_interactions": self.validDataSize,
            "test_interactions": self.testDataSize,
        }
        if self.config['need_graph']:
            infos['graph'] = self.Graph
        return infos

    def mode(self) -> str:
        return self._mode

    def generate_batch(self):
        if self.config['sample'] == 'bpr' and self.mode() == 'train':
            S = UniformSample_original_python(self)
            users = torch.Tensor(S[:, 0]).long()
            posItems = torch.Tensor(S[:, 1]).long()
            negItems = torch.Tensor(S[:, 2]).long()

            users = users.to(self.config['device'])
            posItems = posItems.to(self.config['device'])
            negItems = negItems.to(self.config['device'])
            users, posItems, negItems = shuffle(users, posItems, negItems)
            # total_batch = (
            #     len(users) + self.config['bpr_batch_size'] - 1
            # ) // self.config['bpr_batch_size']
            for batch_users, batch_pos, batch_neg in minibatch(
                users, posItems, negItems, batch_size=self.config['bpr_batch_size']
            ):
                yield {
                    "users": batch_users.to(self.config['device']),
                    "positive_items": batch_pos.to(self.config['device']),
                    "negative_items": batch_neg.to(self.config['device']),
                }
        elif self.mode() == "train":
            raise NotImplementedError("Not implemented yet")
        elif self.mode() in ["validate", "test"]:
            if self.mode() == 'validate':
                testDict = self.valid_dict
            else:
                testDict = self.test_dict
            users = list(testDict.keys())
            for batch_users in minibatch(
                users, batch_size=self.config['test_batch_size']
            ):
                all_pos = self.getUserPosItems(batch_users)
                ground_true = [testDict[u] for u in batch_users]
                yield {
                    "users": torch.Tensor(batch_users).long().to(self.config['device']),
                    "positive_items": all_pos,
                    "ground_truth": ground_true,
                }

    def switch_mode(self, mode):
        assert mode in ["train", "test", "validate"]
        self._mode = mode

    def inject_data(self, mode, data, **kwargs):
        # TODO use copy to avoid the injection affact the original dataset
        if mode == 'train':
            new_train_dict = copy(self.train_dict)
            inject_dict = fake_array2dict(
                data, self.n_users, filter_num=kwargs['filter_num']
            )
            for k, v in inject_dict.items():
                assert (
                    k not in new_train_dict
                ), f"Injection to a exist user {k} is not allowed"
                new_train_dict[k] = v
            return self.reset(train_dict=new_train_dict)
        else:
            raise NotImplementedError(f"Injection not supported in {mode} mode")
