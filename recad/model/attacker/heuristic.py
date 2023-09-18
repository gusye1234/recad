import torch
from .base import BaseAttacker
import numpy as np
import pandas as pd
from ...utils import user_side_check, get_logger, VarDim
from ...default import MODEL
from collections.abc import Iterable


class RandomAttack(BaseAttacker):
    def __init__(self, **config):
        super(RandomAttack, self).__init__()
        train_kv_array = config['dataset'].info_describe()['train_kvr']
        self.n_items = config['dataset'].info_describe()['n_items']

        self.global_mean = np.mean(train_kv_array[:, 2])
        self.global_std = np.std(train_kv_array[:, 2])
        self.attack_num = config['attack_num']
        self.filler_num = config['filler_num']
        self.logger = get_logger(__name__, level=config['logging_level'])

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['attacker']['random'])
        user_args = "dataset"
        return super().from_config("random", args, user_args, kwargs)

    def train_step(self, users, positive_items, negative_items, **config):
        pass

    def forward(self, users, items):
        pass

    def generate_fake(self, target_id_list, **config):
        # target_id_list =[1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        rate = int(self.attack_num / len(target_id_list))
        # padding target score
        self.logger.debug('every target item\'s num %s' % rate)
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = 5
        # padding fillers score
        self.logger.debug(self.n_items)
        filler_pool = list(set(range(self.n_items)) - set(target_id_list))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, self.filler_num])
                    for _ in range(self.attack_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]
        sampled_values = np.random.normal(
            loc=self.global_mean,
            scale=self.global_std,
            size=(self.attack_num * self.filler_num),
        )
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles

    def input_describe(self):
        return {
            "generate_fake": {
                "target_id_list": (Iterable, [VarDim()]),
            }
        }

    def output_describe(self):
        return {
            "generate_fake": {
                "fake_profile": (np.ndarray, (self.attack_num, self.n_items))
            }
        }


class AverageAttack(BaseAttacker):
    def __init__(self, **config):
        super().__init__()
        kvr = config['dataset'].info_describe()['train_kvr']
        self.n_items = config['dataset'].info_describe()['n_items']

        self.global_mean = np.mean(kvr[:, 2])
        self.global_std = np.std(kvr[:, 2])

        self.item_mean_dict = {}
        self.item_std_dict = {}
        for iid in np.unique(kvr[:, 1]):
            self.item_mean_dict[iid] = np.mean(kvr[kvr[:, 1] == iid][:, 2])
            self.item_std_dict[iid] = np.mean(kvr[kvr[:, 1] == iid][:, 2])
        self.attack_num = config['attack_num']
        self.filler_num = config['filler_num']
        self.logger = get_logger(__name__, level=config['logging_level'])

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['attacker']['average'])
        user_args = "dataset"
        return super().from_config("average", args, user_args, kwargs)

    def train_step(self, users, positive_items, negative_items, **config):
        pass

    def forward(self, users, items):
        pass

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        rate = int(self.attack_num / len(target_id_list))
        # padding target score
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = 5

        filler_pool = list(set(range(self.n_items)) - set(target_id_list))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, self.filler_num])
                    for _ in range(self.attack_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]
        # sampled_values = np.random.normal(loc=0, scale=1,
        #                                   size=(self.attack_num * self.filler_num))
        sampled_values = [
            np.random.normal(
                loc=self.item_mean_dict.get(iid, self.global_mean),
                scale=self.item_std_dict.get(iid, self.global_std),
            )
            for iid in sampled_cols
        ]
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles

    def input_describe(self):
        return {
            "generate_fake": {
                "target_id_list": (Iterable, [VarDim()]),
            }
        }

    def output_describe(self):
        return {
            "generate_fake": {
                "fake_profile": (np.ndarray, (self.attack_num, self.n_items))
            }
        }


class SegmentAttack(BaseAttacker):
    def __init__(self, **config):
        super().__init__()
        kvr = config['dataset'].info_describe()['train_kvr']
        self.n_items = config['dataset'].info_describe()['n_items']

        self.selected_ids = config['selected_ids']
        self.attack_num = config['attack_num']
        self.filler_num = config['filler_num']
        self.logger = get_logger(__name__, level=config['logging_level'])

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['attacker']['segment'])
        user_args = "dataset"
        return super().from_config("segment", args, user_args, kwargs)

    def train_step(self, users, positive_items, negative_items, **config):
        pass

    def forward(self, users, items):
        pass

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        rate = int(self.attack_num / len(target_id_list))
        # padding target score
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(
            set(range(self.n_items)) - set(target_id_list) - set(self.selected_ids)
        )
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, self.filler_num])
                    for _ in range(self.attack_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]
        sampled_values = np.ones_like(sampled_rows)
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles

    def input_describe(self):
        return {
            "generate_fake": {
                "target_id_list": (Iterable, [VarDim()]),
            }
        }

    def output_describe(self):
        return {
            "generate_fake": {
                "fake_profile": (np.ndarray, (self.attack_num, self.n_items))
            }
        }


class BandwagonAttack(BaseAttacker):
    def __init__(self, **config):
        super().__init__()
        kvr = config['dataset'].info_describe()['train_kvr']

        self.global_mean = np.mean(kvr[:, 2])
        self.global_std = np.std(kvr[:, 2])

        train_data_df = pd.DataFrame(kvr)
        train_data_df.columns = ['user_id', 'item_id', 'rating']
        self.n_items = config['dataset'].info_describe()['n_items']

        self.selected_ids = config['selected_ids']
        self.attack_num = config['attack_num']
        self.filler_num = config['filler_num']
        if len(self.selected_ids) == 0:
            sorted_item_pop_df = (
                train_data_df.groupby('item_id')
                .agg('count')
                .sort_values('user_id')
                .index[::-1]
            )
            self.selected_ids = sorted_item_pop_df[:11].to_list()
        self.logger = get_logger(__name__, level=config['logging_level'])

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['attacker']['bandwagon'])
        user_args = "dataset"
        return super().from_config("bandwagon", args, user_args, kwargs)

    def train_step(self, users, positive_items, negative_items, **config):
        pass

    def forward(self, users, items):
        pass

    def generate_fake(self, target_id_list, **config):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        # fake_profiles[:, self.target_id] = 5
        rate = int(self.attack_num / len(target_id_list))
        # padding target score
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(
            set(range(self.n_items)) - set(target_id_list) - set(self.selected_ids)
        )
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, self.filler_num])
                    for _ in range(self.attack_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(self.attack_num) for j in [i] * self.filler_num
        ]
        sampled_values = np.random.normal(
            loc=self.global_mean,
            scale=self.global_std,
            size=(self.attack_num * self.filler_num),
        )
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles

    def input_describe(self):
        return {
            "generate_fake": {
                "target_id_list": (Iterable, [VarDim()]),
            }
        }

    def output_describe(self):
        return {
            "generate_fake": {
                "fake_profile": (np.ndarray, (self.attack_num, self.n_items))
            }
        }
