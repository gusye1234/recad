import torch
from .base import BaseAttacker
import numpy as np
from ...utils import user_side_check, get_logger, VarDim
from ...default import MODEL
from collections.abc import Iterable


class RandomAttack(BaseAttacker):
    def __init__(self, **config):
        super(RandomAttack, self).__init__()
        train_df = config['dataset'].info_describe()['train_df']
        self.n_items = config['dataset'].info_describe()['n_items']

        self.global_mean = train_df.rating.mean()
        self.global_std = train_df.rating.std()
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

    def generate_fake(self, target_id_set):
        # target_id_set =[1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        rate = int(self.attack_num / len(target_id_set))
        # padding target score
        self.logger.debug('every target item\'s num %s' % rate)
        for i in range(len(target_id_set)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_set[i]] = 5
        # fake_profiles[:, self.target_id] = 5
        # padding fillers score
        self.logger.debug(self.n_items)
        filler_pool = list(set(range(self.n_items)) - set(target_id_set))
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
                "dataset": ("Attack dataset", []),
                "target_id_set": (Iterable, [VarDim()]),
            }
        }

    def output_describe(self):
        return {
            "generate_fake": {
                "fake_profile": (np.ndarray, (self.attack_num, self.n_items))
            }
        }
