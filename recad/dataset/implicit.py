import os
from .base import BaseData
from ..default import DATASET
from ..utils import VarDim, get_logger, check_dir_or_make
import torch
from time import time
from copy import copy
import pandas as pd
import numpy as np
import random
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


def pairwise_sample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = dataset.traindataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

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
    return np.array(S)


def pointwise_sample(dataset: BaseData, negative_num_each_interaction):
    data = []
    train_dict = dataset.info_describe()['train_dict']
    n_items = dataset.info_describe()['n_items']

    full_items = set(range(n_items))
    for uid, iids in train_dict.items():
        data.extend([(uid, iid, 1) for iid in iids])

        left_set = list(full_items - set(iids))
        negs = np.random.choice(
            left_set, size=len(iids) * negative_num_each_interaction
        )
        data.extend([(uid, ni, 0) for ni in negs])
    return np.array(data)


def csv2dict(file, filter=4):
    df = pd.read_csv(file)
    interactions = {}
    df = df.sort_values("timestamp")
    df = df[df['rating'] >= filter]
    for u, i in zip(df['user_id'], df['item_id']):
        if u not in interactions:
            interactions[int(u)] = []
        if i not in interactions[u]:  # to keep order of time
            interactions[int(u)].append(int(i))
    return interactions


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


def convert2dict(file: str, filter_num):
    if file.endswith(".csv"):
        return csv2dict(file, filter=filter_num)
    elif file.endswith(".npy"):
        return npy2dict(file)
    else:
        raise ValueError(f"Expect data file ends witg [csv/npy], but got {file}")


class ImplicitData(BaseData):
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

        for attr, default in zip(
            ['train_dict', 'valid_dict', 'test_dict'],
            [train_file, valid_file, test_file],
        ):
            if self.config[attr] is not None:
                self.logger.debug(f"Init {attr} from a passed dict")
                setattr(self, attr, self.config[attr])
                continue
            maybe_cache = join(
                self.config['cache_dir'],
                f"{self.dataset_name}_implicit_{attr}.npy",
            )
            if os.path.exists(maybe_cache) and self.config['if_cache']:
                self.logger.info(f"loading cached {attr}")
                setattr(self, attr, np.load(maybe_cache, allow_pickle=True).item())
            else:
                setattr(self, attr, convert2dict(default, self.config['rating_filter']))
                if self.config['if_cache']:
                    check_dir_or_make(self.config['cache_dir'])
                    np.save(maybe_cache, getattr(self, attr))

    def _init_data(self):
        self.n_users = 0
        self.n_items = 0

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

        self.n_items += 1
        self.n_users += 1

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
            shape=(self.n_users, self.n_items),
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
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
                self.n_items = max(self.n_items, max(data_dict[uid]))
                self.n_users = max(self.n_users, uid)
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
            maybe_cache = join(
                self.config['cache_dir'],
                f"adj_mat_{self.dataset_name}_{self.n_users}_{self.n_items}.npz",
            )
            if os.path.exists(maybe_cache) and self.config['if_cache']:
                pre_adj_mat = sp.load_npz(maybe_cache)
                self.logger.info(
                    f"successfully loaded adj_mat from {maybe_cache}, this could cause the inconsistency of the dataset"
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
                d_inv = np.power(rowsum + 1e-14, -0.5).flatten()
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
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    @classmethod
    def from_config(cls, name, **user_config):
        args = list(DATASET['implicit'][name])
        return super().from_config("implicit", name, args, user_config)

    def batch_describe(self):
        if self.config['sample'] == 'pairwise' and self.mode() == "train":
            return {
                "users": (
                    torch.int64,
                    (VarDim(max=self.config['pairwise_batch_size'], comment="batch")),
                ),
                "positive_items": (
                    torch.int64,
                    (VarDim(max=self.config['pairwise_batch_size'], comment="batch")),
                ),
                "negative_items": (
                    torch.int64,
                    (VarDim(max=self.config['pairwise_batch_size'], comment="batch")),
                ),
            }
        if self.config['sample'] == 'pointwise' and self.mode() == "train":
            return {
                "users": (
                    torch.int64,
                    (VarDim(max=self.config['pointwise_batch_size'], comment="batch")),
                ),
                "items": (
                    torch.int64,
                    (VarDim(max=self.config['pointwise_batch_size'], comment="batch")),
                ),
                "labels": (
                    torch.int64,
                    (VarDim(max=self.config['pointwise_batch_size'], comment="batch")),
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
            "n_users": self.n_users,
            "n_items": self.n_items,
            "train_interactions": self.traindataSize,
            "valid_interactions": self.validDataSize,
            "test_interactions": self.testDataSize,
            "train_dict": self.train_dict,
            "valid_dict": self.valid_dict,
            "test_dict": self.test_dict,
            "batch_describe": self.batch_describe(),
        }
        if self.config['need_graph']:
            infos['graph'] = self.Graph
        return infos

    def mode(self) -> str:
        return self._mode

    def generate_batch(self, **config):
        if self.config['sample'] == 'pairwise' and self.mode() == 'train':
            S = pairwise_sample(self)
            users = torch.Tensor(S[:, 0]).long()
            posItems = torch.Tensor(S[:, 1]).long()
            negItems = torch.Tensor(S[:, 2]).long()

            users = users.to(self.config['device'])
            posItems = posItems.to(self.config['device'])
            negItems = negItems.to(self.config['device'])
            users, posItems, negItems = shuffle(users, posItems, negItems)
            # total_batch = (
            #     len(users) + self.config['pairwise_batch_size'] - 1
            # ) // self.config['pairwise_batch_size']
            for batch_users, batch_pos, batch_neg in minibatch(
                users, posItems, negItems, batch_size=self.config['pairwise_batch_size']
            ):
                yield {
                    "users": batch_users.to(self.config['device']),
                    "positive_items": batch_pos.to(self.config['device']),
                    "negative_items": batch_neg.to(self.config['device']),
                }
        elif self.config['sample'] == 'pointwise' and self.mode() == 'train':
            S = pointwise_sample(self, self.config['negative_ratio'])
            users = torch.Tensor(S[:, 0]).long()
            items = torch.Tensor(S[:, 1]).long()
            labels = torch.Tensor(S[:, 2]).long()

            users = users.to(self.config['device'])
            items = items.to(self.config['device'])
            labels = labels.to(self.config['device'])
            users, items, labels = shuffle(users, items, labels)
            # total_batch = (
            #     len(users) + self.config['pairwise_batch_size'] - 1
            # ) // self.config['pairwise_batch_size']
            for batch_users, batch_items, batch_labels in minibatch(
                users, items, labels, batch_size=self.config['pointwise_batch_size']
            ):
                yield {
                    "users": batch_users,
                    "items": batch_items,
                    "labels": batch_labels,
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

    def inject_data(self, data_mode, data, **kwargs):
        if data_mode == 'explicit':
            new_train_dict = copy(self.train_dict)
            inject_dict = fake_array2dict(
                data, self.n_users, filter_num=kwargs['filter_num']
            )
            for k, v in inject_dict.items():
                assert (
                    k not in new_train_dict
                ), f"Injection to a exist user {k} is not allowed"
                new_train_dict[k] = v
            return self.reset(train_dict=new_train_dict, if_cache=False)
        raise NotImplementedError(f"Injection not supported in {data_mode} mode")
    
    def delete_data(self, data_mode, user_id, data, **kwargs):
        if data_mode == 'explicit':
            new_train_dict = copy(self.train_dict)
            
            
            inject_dict = fake_array2dict(
                data, self.n_users, filter_num=kwargs['filter_num']
            )
            for k, v in inject_dict.items():
                assert (
                    k not in new_train_dict
                ), f"Injection to a exist user {k} is not allowed"
                new_train_dict[k] = v
            for key in list(new_train_dict.keys()):
                if key in user_id:
                    del new_train_dict[key]
            return self.reset(train_dict=new_train_dict, if_cache=False)
        raise NotImplementedError(f"Injection not supported in {data_mode} mode")

    def partial_sample(self, **kwargs) -> 'BaseData':
        # TODO return the sampled implicit feedback, but attacker may want ratings
        assert "user_ratio" in kwargs, "Expect to have [user_ratio]"
        user_ratio = kwargs['user_ratio']
        if abs(user_ratio - 1) < 1e-9:
            return self
        users = list(self.train_dict)
        random.shuffle(users)
        left_users = users[: int(len(users) * user_ratio)]
        new_dict = {u: self.train_dict[u] for u in left_users}
        return self.reset(train_dict=new_dict)
