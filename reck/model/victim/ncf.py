from .base import BaseVictim
import torch
from torch import nn
from ...utils import pick_optim, VarDim
from ...default import MODEL


class NCF(BaseVictim):
    def __init__(
        self, factor_num, num_layers, dropout, model, GMF_model, MLP_model, **config
    ):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dataset = config['dataset']
        user_num = self.dataset.info_describe()['n_users']
        item_num = self.dataset.info_describe()['n_items']

        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

        self.optimizer = pick_optim(config['optim'])(self.parameters(), lr=config['lr'])
        self.loss_func = nn.BCEWithLogitsLoss()

    def _init_weight_(self):
        """We leave the weights initialization here."""
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(
                self.predict_layer.weight, a=1, nonlinearity='sigmoid'
            )

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat(
                [
                    self.GMF_model.predict_layer.weight,
                    self.MLP_model.predict_layer.weight,
                ],
                dim=1,
            )
            precit_bias = (
                self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias
            )

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['victim']['ncf'])
        user_args = "dataset"
        return super().from_config("ncf", args, user_args, kwargs)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

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
                pbar.set_description(f"loss: {total_loss/(idx+1):.5f}")
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
