from sklearn.metrics import classification_report
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
import scipy
from scipy.sparse import csr_matrix
from ...default import MODEL
from .base import BaseDefender
from ...utils import user_side_check, get_logger, VarDim
from ...default import MODEL
from ...dataset import BaseData
from ...utils import pick_optim
import torch
from torch import nn
import numpy as np


class PCASelectUsers(BaseDefender):
    def __init__(
        self, **config
    ):
        super(PCASelectUsers, self).__init__()

        self.dataset = config['dataset']
        # check this number is add 50
        self.user_num = self.dataset.info_describe()['n_users']
        self.item_num = self.dataset.info_describe()['n_items']
        self.k = config['kVals']

        if self.k >= min(self.user_num, self.item_num):
            self.k = 3
            print('k-vals is more than the number of user or item, so it is set to', self.k)

        self.attack_ratio = config['attack_num'] / self.user_num
        
    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['defender']['PCASelectUsers'])
        user_args = "dataset"
        return super().from_config("PCASelectUsers", args, user_args, kwargs)

    def forwad(self):
        pass

    def train_step(self, **config):
        pass
    
    def defense_step(self, **config):
        # dataArray = np.zeros([self.user_num, self.item_num], dtype=float)
        dataArray = torch.zeros(self.user_num, self.item_num, dtype=torch.float, device='cuda')
        self.testLabels = np.zeros(self.user_num)
        self.predLabels = np.zeros(self.user_num)
        # Speed Up
        for idx, dp in enumerate(self.dataset.generate_batch()):
            user_list = dp['users']
            user_matrix = dp['users_mat']
            for user_order, user in enumerate(user_list):
                item_list = user_matrix[user_order]
                dataArray[user, :] = item_list
        
        dataArray = dataArray.cpu().numpy()
        sMatrix = csr_matrix(dataArray)
        # z-scores
        sMatrix = preprocessing.scale(sMatrix, axis=0, with_mean=False)
        sMT = np.transpose(sMatrix)
        # cov
        covSM = np.dot(sMT, sMatrix)
        # eigen-value-decomposition
        vals, vecs = scipy.sparse.linalg.eigs(covSM, k=self.k, which='LM')

        newArray = np.dot(dataArray**2, np.real(vecs))

        distanceDict = {}
        userId = 0
        for user in newArray:
            distance = 0
            for tmp in user:
                distance += tmp
            distanceDict[userId] = float(distance)
            userId += 1

        self.disSort = sorted(distanceDict.items(), key=lambda d: d[1], reverse=False)

        spamList = []
        i = 0
        while i < self.attack_ratio * len(self.disSort):
            spam = self.disSort[i][0]
            spamList.append(spam)
            self.predLabels[spam] = 1
            i += 1
        
        return spamList

    def input_describe(self):
        # TODO change a Log
        return {
            "generate_fake": {
                "fake_profile": None
            }
        }


