from os import path
import torch
import pandas as pd
import numpy as np
from .base import BaseWorkflow
from ..dataset import BaseData
from ..model.attacker import BaseAttacker
from ..model.victim import BaseVictim
from ..default import WORKFLOW
from ..utils import check_dir_or_make, pick_optim, get_logger, dict2list_table, fmt_tab
from collections import OrderedDict
import logging
from tqdm import tqdm


class Normal(BaseWorkflow):
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
        self.attacker: BaseAttacker = config['attacker'].I(
            dataset=config['attack_data']
        )
        self.victim: BaseVictim = config['victim'].I(dataset=config['victim_data'])
        self.victim_data: BaseData = config['victim_data']
        self.logger = get_logger(__name__, level=self.c['logging_level'])

    @classmethod
    def from_config(cls, **kwargs):
        args = list(WORKFLOW['normal'])
        user_args = "victim_data, attack_data, victim, attacker"
        return super().from_config("normal", args, user_args, kwargs)

    def input_describe(self):
        return {
            "victim": BaseVictim,
            "victim_data": BaseData,
            "attacker": BaseAttacker,
            "attack_data": BaseData,
        }

    def info_describe(self):
        return {
            'target_id_list': self.c['target_id_list'],
            'input_describe': self.input_describe(),
        }

    def user_item_model_generate(self, model, data_dict, topks, target_ids, device):
        preds = []
        users = []
        items = []
        with torch.no_grad():
            for u, iids in tqdm(data_dict.items(), desc="Inferring"):
                if not len(iids):
                    continue
                tensor_iids = torch.tensor(iids, dtype=torch.int64).to(device)
                us = torch.full_like(tensor_iids, u)
                pred = model(us, tensor_iids).cpu().numpy().tolist()

                preds.extend(pred)
                users.extend([u] * len(iids))
                items.extend(iids)
        pred_results = pd.DataFrame(
            {'user_id': users, 'item_id': items, 'rating': preds}
        )

        topks_array = np.zeros(
            [len(pred_results.user_id.unique()) * len(target_ids), len(topks) + 2]
        )
        pbar = tqdm(enumerate(pred_results.groupby('user_id')))
        for idx, (user_id, pred_result) in pbar:
            pbar.set_description(f"user {user_id}")
            for bias, target_id in enumerate(target_ids):
                pred_value = pred_result[
                    pred_result.item_id == target_id
                ].rating.values[0]
                sorted_recommend_list = pred_result.sort_values(
                    'rating', ascending=False
                ).item_id.values
                new_line = [user_id, pred_value] + [
                    1 if target_id in sorted_recommend_list[:k] else 0 for k in topks
                ]
                topks_array[idx + bias] = new_line
        return topks_array

    def normal_train(self, **config):
        progress = tqdm(range(config['epoch']))
        for ep in progress:
            loss = config['model'].train_step(
                **self.info_describe(), progress_bar=progress
            )
            out_des = config['model'].output_describe()['train_step']

            assert len(loss) == len(
                out_des
            ), f"The output describe is not aligned with the actual output of train_step for {config['model'].model_name}"
            loss_flag = ""
            for i, out in enumerate(out_des.keys()):
                loss_flag = loss_flag + f"{out}: {loss[i]:.4f}|"
            progress.set_description(f"{loss_flag}")

    def normal_evaluate(
        self,
        model: BaseVictim,
        model_fake: BaseVictim,
        dataset: BaseData,
        target_id_list,
        topks,
    ):
        train_dict = dataset.info_describe()['train_dict']
        n_users = dataset.info_describe()['n_users']
        n_items = dataset.info_describe()['n_items']
        for m in [model, model_fake]:
            assert (
                len(m.input_describe()['forward']) == 2
            ), "Expect forward only need two inputs"
            assert (
                "users" in m.input_describe()['forward']
            ), "Expect to have the users input in forward method"
            assert (
                "items" in m.input_describe()['forward']
            ), "Expect to have the items input in forward method"

        filtered_dict = {k: set(v) for k, v in train_dict.items()}
        full_items = set(range(n_items))
        for k in train_dict:
            existed = False
            for target in target_id_list:
                if target in filtered_dict[k]:
                    del filtered_dict[k]
                    existed = True
                    break
            if not existed:
                filtered_dict[k] = list(full_items - filtered_dict[k])
        self.logger.debug("Done filtered")
        pred_results = self.user_item_model_generate(
            model, filtered_dict, topks, target_id_list, self.c['device']
        )
        pred_results_fake = self.user_item_model_generate(
            model_fake, filtered_dict, topks, target_id_list, self.c['device']
        )
        assert np.allclose(
            pred_results[:, 0], pred_results_fake[:, 0]
        ), "Users are not aligned"

        results = OrderedDict()
        results['pred_shift'] = np.mean(pred_results_fake[:, 1] - pred_results[:, 1])
        for i, k in enumerate(topks):
            results[f"HR@{k}"] = np.mean(pred_results[:, 2 + i])
            results[f"HR@{k} after attack"] = np.mean(pred_results_fake[:, 2 + i])
        print(fmt_tab(dict2list_table(results)))

    def execute(self):
        self.logger.info(
            f"Normal attacking, with dataset {self.victim_data.dataset_name}, victim model {self.victim.model_name}, attack model {self.attacker.model_name}, on device {self.c['device']}"
        )

        self.victim = self.victim.to(self.c['device'])
        self.attacker = self.attacker.to(self.c['device'])

        self.logger.info("Step 1. training a recommender")
        self.normal_train(
            **{
                'model': self.victim,
                'epoch': self.c['rec_epoch'],
                'dataset': self.victim_data,
            }
        )

        self.logger.info("Step 2. training a attacker")
        if "train_step" in self.attacker.input_describe():
            self.normal_train(
                **{
                    'model': self.attacker,
                    'epoch': self.c['attack_epoch'],
                    'dataset': self.c['attack_data'],
                }
            )
        else:
            self.logger.info(
                f"Skip attacker training, since {self.attacker.model_name} didn't require it"
            )

        self.logger.info("Step 3. injecting fake data and re-train the recommender")
        fake_array = self.attacker.generate_fake(**self.info_describe())
        self.logger.debug(f"{fake_array.shape}")
        fake_dataset = self.victim_data.inject_data(
            "explicit", fake_array, filter_num=self.c['filter_num']
        )
        self.logger.debug("After injection of the fake data")
        if self.c['logging_level'] == logging.DEBUG:
            fake_dataset.print_help()

        fake_victim = self.victim.reset().to(self.c['device'])
        self.normal_train(
            **{
                'model': fake_victim,
                'epoch': self.c['rec_epoch'],
                'dataset': fake_dataset,
            }
        )

        # TODO add evaluate
        self.logger.debug(
            f"targets: {self.c['target_id_list']}, topks: {self.c['topks']}"
        )
        self.normal_evaluate(
            self.victim,
            fake_victim,
            self.victim_data,
            self.c['target_id_list'],
            topks=self.c['topks'],
        )
