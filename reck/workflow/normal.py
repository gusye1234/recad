from os import path
import torch
from .base import BaseFlow
from ..dataset import BaseDataset
from ..model.attacker import BaseAttacker
from ..model.victim import BaseVictim
from ..default import WORKFLOW
from ..utils import check_dir_or_make, pick_optim, get_logger
from copy import deepcopy
from tqdm import tqdm


class Normal(BaseFlow):
    def __init__(self, **config):
        self.c = config
        self.saving_dir = check_dir_or_make(
            config['cache_dir'],
            "_".join(
                [
                    config['victim_data'].dataset_name,
                    config['attack_data'].dataset_name,
                    config['victim'].model_name,
                    config['attacker'].model_name,
                ]
            ),
        )
        self.attacker: BaseAttacker = config['attacker']
        self.victim: BaseVictim = config['victim']
        self.victim_data: BaseDataset = config['victim_data']
        self.logger = get_logger(__name__, level=self.c['logging_level'])

    @classmethod
    def from_config(cls, **kwargs):
        args = list(WORKFLOW['normal'])
        user_args = "victim_data, attack_data, victim, attacker"
        return super().from_config("normal", args, user_args, kwargs)

    def input_describe(self):
        return super().input_describe()

    def normal_train(self, **config):
        optimizer = pick_optim(config['optim'])(
            config['model'].parameters(), lr=config['lr']
        )
        self.logger.info("training the recommendation model")

        progress = tqdm(range(config['epoch']))
        for ep in progress:
            for dp in config['dataset'].generate_batch():
                loss = config['model'].train_step(**dp, optimizer=optimizer)
                progress.set_description(f"loss:{loss:.4f}")

    def execute(self):
        self.logger.info("Step 1. training a recommender")
        self.normal_train(
            **{
                'optim': self.c['rec_optim'],
                'model': self.victim,
                'lr': self.c['rec_lr'],
                'epoch': self.c['rec_epoch'],
                'dataset': self.victim_data,
            }
        )

        self.logger.info("Step 2. training a attacker")
        if "train_step" in self.attacker.input_describe():
            self.normal_train(
                **{
                    'optim': self.c['attack_optim'],
                    'model': self.attacker,
                    'lr': self.c['attack_lr'],
                    'epoch': self.c['attack_epoch'],
                    'dataset': self.c['attack_data'],
                }
            )
        else:
            self.logger.info(
                f"Skip attacker training, since {self.attacker.model_name} didn't require it"
            )

        self.logger.info("Step 3. injecting fake data and re-train the recommender")
        fake_array = self.attacker.generate_fake(self.c['target_items'])
        self.logger.debug(f"{fake_array.shape}")
        fake_dataset = self.victim_data.inject_data(
            "train", fake_array, filter_num=self.c['filter_num']
        )
        self.logger.info("After injection of the fake data")
        fake_dataset.print_help()

        fake_victim = self.victim.reset()
        self.normal_train(
            **{
                'optim': self.c['rec_optim'],
                'model': fake_victim,
                'lr': self.c['rec_lr'],
                'epoch': self.c['rec_epoch'],
                'dataset': fake_dataset,
            }
        )

        # TODO add evaluate
