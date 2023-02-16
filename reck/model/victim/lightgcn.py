import torch
from .base import BaseVictim
from torch import nn
from ...utils import user_side_check, get_logger, VarDim
from ...default import MODEL
from ...dataset import BaseDataset
from ...utils import pick_optim


class LightGCN(BaseVictim):
    def __init__(self, **config):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BaseDataset = config['dataset']
        self.logger = get_logger(__name__, config['logging_level'])
        self.__init_weight()
        self.optimizer = pick_optim(self.config['optim'])(
            self.parameters(), lr=self.config['lr']
        )

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['victim']['lightgcn'])
        if kwargs.get("pretrain", False):
            user_args = "dataset user_emb, item_emb"
        else:
            user_args = "dataset"
        return super().from_config("lightgcn", args, user_args, kwargs)

    def __init_weight(self):
        self.num_users = self.dataset.info_describe()['n_users']
        self.num_items = self.dataset.info_describe()['n_items']
        self.Graph = self.dataset.info_describe()['graph']
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        if self.A_split:
            raise ValueError("A_split is not support in LightGCN yet")
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            self.logger.debug('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb'])
            )
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb'])
            )
            self.logger.debug('use pretarined data')
        self.f = nn.Sigmoid()
        self.logger.debug(f"lgn is already to go(dropout:{self.config['dropout']})")
        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def train_step(self, **config):
        optim = self.optimizer
        total_loss = 0.0
        self.train()
        pbar = config.get('progress_bar', None)
        for idx, dp in enumerate(self.dataset.generate_batch()):
            users = dp['users']
            pos = dp['positive_items']
            neg = dp['negative_items']
            (
                users_emb,
                pos_emb,
                neg_emb,
                userEmb0,
                posEmb0,
                negEmb0,
            ) = self.getEmbedding(users.long(), pos.long(), neg.long())
            reg_loss = (
                (1 / 2)
                * (
                    userEmb0.norm(2).pow(2)
                    + posEmb0.norm(2).pow(2)
                    + negEmb0.norm(2).pow(2)
                )
                / float(len(users))
            )
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)

            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

            final_loss = loss + self.config['lambda'] * reg_loss
            optim.zero_grad()
            final_loss.backward()
            optim.step()
            total_loss += final_loss.item()
            if pbar:
                pbar.set_description(f"loss {total_loss / (idx + 1):.5f}")
        return (total_loss / (idx + 1),)

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def input_describe(self):
        return {
            "train_step": {
                "users": (torch.int64, (VarDim(comment="batch"))),
                "positive_items": (torch.int64, (VarDim(comment="batch"))),
                "negative_items": (torch.int64, (VarDim(comment="batch"))),
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
