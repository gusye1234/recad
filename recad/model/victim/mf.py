import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseVictim
from ...utils import VarDim, pick_optim
from ...default import MODEL


class MF(BaseVictim):
    def __init__(self, factor_num, embedding_size, dropout, **config):
        super(MF, self).__init__()
        num_users = config['dataset'].info_describe()['n_users']
        num_items = config['dataset'].info_describe()['n_items']
        self.dataset = config['dataset']
        self.config = config
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([factor_num]), False)
        self.dropout = nn.Dropout(dropout)

        self.optimizer = pick_optim(self.config['optim'])(
            self.parameters(), lr=self.config['lr']
        )
        self.loss_func = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['victim']['mf'])
        user_args = "dataset"
        return super().from_config("mf", args, user_args, kwargs)

    def forward(self, users, items):
        u_id = users
        i_id = items
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return self.dropout((U * I).sum(1) + b_u + b_i + self.mean)

    def train_step(self, **config):
        optim = self.optimizer
        total_loss = 0.0
        self.train()
        pbar = config.get('progress_bar', None)
        for idx, dp in enumerate(self.dataset.generate_batch()):
            user = dp['users']
            item = dp['items']
            label = dp['labels']

            pred = self(user, item)
            loss = self.loss_func(pred, label.float())

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if pbar:
                pbar.set_description(f"loss: {total_loss/(idx + 1):.4f}")
        return (total_loss / (idx + 1),)

    def input_describe(self):
        return {
            "train_step": {
                "users": (torch.int64, (VarDim(comment="batch"))),
                "items": (torch.int64, (VarDim(comment="batch"))),
                "labels": (torch.int64, (VarDim(comment="batch"))),
            },
            "forward": {
                "users": (torch.int64, (VarDim(comment="batch"))),
                "items": (torch.int64, (VarDim(comment="batch"))),
            },
        }

    def output_describe(self):
        return {
            "train_step": {
                "loss": (float, []),
            },
            "forward": {
                "unnormalized_scores": (torch.float32, [VarDim(comment="batch")])
            },
        }
