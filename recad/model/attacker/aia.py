# TODO
from .base import BaseAttacker
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from ...default import MODEL
from ...utils import pick_optim, VarDim


def aia_attack_loss(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_probs * labels
    instance_data = labels.sum(1)
    instance_loss = loss.sum(1)
    # Avoid divide by zeros.
    res = instance_loss / (instance_data + 0.1)  # PSILON)
    return res


def partial_update(loss, optimizer, part=0):
    optimizer.zero_grad()
    grad_groups = torch.autograd.grad(
        loss, optimizer.param_groups[part]['params'], allow_unused=True
    )
    for para_, grad_ in zip(optimizer.param_groups[part]['params'], grad_groups):
        if para_.grad is None:
            para_.grad = grad_.clone()
        else:
            # print(grad_)
            para_.grad.data = grad_

    optimizer.step()


class AIA(BaseAttacker):
    def __init__(self, **config) -> None:
        super().__init__()

        self.selected_ids = config['selected_ids']
        self.attack_num = config['attack_num']
        self.filler_num = config['filler_num']
        self.dataset = config['dataset']
        self.n_users = config['dataset'].info_describe()['n_users']
        self.n_items = config['dataset'].info_describe()['n_items']
        self.device = config['device']
        self.train_array = config['dataset'].info_describe()['train_mat']
        self.surrogate = config["surrogate_model"]
        self.config = config
        self.build_network()

    def build_network(self):
        sampled_idx = np.random.choice(
            np.where(np.sum(self.train_array > 0, 1) >= self.filler_num)[0],
            self.attack_num,
        )
        templates = self.train_array[sampled_idx]
        for idx, template in enumerate(templates):
            fillers = np.where(template)[0]
            np.random.shuffle(fillers)
            for iid in fillers[self.filler_num :]:
                templates[idx][iid] = 0.0
        self.real_template = torch.tensor(templates).type(torch.float).to(self.device)
        self.netG = RecsysGenerator(self.device, self.real_template).to(self.device)

        self.G_optimizer = pick_optim(self.config['optim_g'])(
            self.parameters(), lr=self.config['lr_g']
        )

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['attacker']['aia'])
        user_args = "dataset"
        return super().from_config("aia", args, user_args, kwargs)

    def input_describe(self):
        return {"train_step": {"target_id_list": (list, VarDim())}}

    def output_describe(self):
        return {
            "train_step": {
                "g_losses": (float, []),
            }
        }

    def train_step(self, **config):
        target_id_list = config['target_id_list']
        self.netG.train()
        G_loss = torch.tensor(0.0).to(self.device)
        fake_tensor = self.netG(self.real_template)
        sur_predictions = self.get_sur_predictions(fake_tensor)
        for target_id in target_id_list:
            target_users = np.where(self.train_array[:, target_id] == 0)[0]
            attack_target = np.zeros((len(target_users), self.n_items))
            attack_target[:, target_id] = 1.0
            attack_target = (
                torch.from_numpy(attack_target).type(torch.float32).to(self.device)
            )
            higher_mask = (
                sur_predictions[target_users]
                >= (sur_predictions[target_users, target_id].reshape([-1, 1]))
            ).float()
            G_loss_sub = aia_attack_loss(
                logits=sur_predictions[target_users] * higher_mask,
                labels=attack_target,
            ).mean()
            G_loss += G_loss_sub
        G_loss = G_loss / 10

        partial_update(G_loss, self.G_optimizer)

        return (G_loss.cpu().item(),)

    def generate_fake(self, **config):
        target_id_list = config['target_id_list']
        with torch.no_grad():
            fake_tensor = self.netG(self.real_template)
            rate = int(self.attack_num / len(target_id_list))
            for i in range(len(target_id_list)):
                fake_tensor[i * rate : (i + 1) * rate, target_id_list[i]] = 5
        return fake_tensor.detach().cpu().numpy()

    def get_sur_predictions(self, fake_tensor):
        data_tensor = torch.cat(
            [
                torch.from_numpy(self.train_array).type(torch.float32).to(self.device),
                fake_tensor,
            ],
            dim=0,
        )

        surrogate = self.surrogate

        if surrogate == 'WMF':
            sur_trainer_ = WMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dim=self.config['hidden_dim_s'],
                # device=self.device,
                device=self.device,
                lr=self.config['lr_s'],
                weight_decay=self.config['weight_decay_s'],
                batch_size=self.config['batch_size_s'],
                weight_pos=self.config['weight_pos_s'],
                weight_neg=self.config['weight_neg_s'],
                verbose=False,
            )
            epoch_num_ = self.config['epoch_s']
            unroll_steps_ = self.config['unroll_steps_s']
        # elif surrogate == 'ItemAE':
        #     sur_trainer_ = ItemAETrainer(
        #         n_users=self.n_users + self.attack_num,
        #         n_items=self.n_items,
        #         hidden_dims=self.config.hidden_dim_s,
        #         device=self.device,
        #         lr=self.config.lr_s,
        #         l2=self.config.weight_decay_s,
        #         batch_size=self.config.batch_size_s,
        #         weight_pos=self.config.weight_pos_s,
        #         weight_neg=self.config.weight_neg_s,
        #         verbose=False)
        #     epoch_num_ = self.config.epoch_s
        #     unroll_steps_ = self.config.unroll_steps_s
        # elif surrogate == 'SVDpp':
        #     sur_trainer_ = SVDppTrainer(
        #         n_users=self.n_users + self.attack_num,
        #         n_items=self.n_items,
        #         hidden_dims=[128],
        #         device=self.device,
        #         lr=1e-3,
        #         l2=5e-2,
        #         batch_size=128,
        #         weight_alpha=20)
        #     epoch_num_ = 10
        #     unroll_steps_ = 1
        # elif surrogate == 'NMF':
        #     sur_trainer_ = NMFTrainer(
        #         n_users=self.n_users + self.attack_num,
        #         n_items=self.n_items,
        #         batch_size=128,
        #         device=self.device,
        #     )
        #     epoch_num_ = 50
        #     unroll_steps_ = 1
        # elif surrogate == 'PMF':
        #     sur_trainer_ = PMFTrainer(
        #         n_users=self.n_users + self.attack_num,
        #         n_items=self.n_items,
        #         hidden_dim=128,
        #         device=self.device,
        #         lr=0.0001,
        #         weight_decay=0.1,
        #         batch_size=self.config.batch_size_s,
        #         momentum=0.9,
        #         verbose=True)
        #     epoch_num_ = 50
        #     unroll_steps_ = 1
        else:
            raise ValueError(
                f'surrogate model error : {surrogate}',
            )

        sur_predictions = sur_trainer_.fit_adv(
            data_tensor=data_tensor, epoch_num=epoch_num_, unroll_steps=unroll_steps_
        )

        # sur_test_rmse = np.mean(
        #     (
        #         sur_predictions[: self.n_users][self.test_array > 0]
        #         .detach()
        #         .cpu()
        #         .numpy()
        #         - self.test_array[self.test_array > 0]
        #     )
        #     ** 2
        # )

        return sur_predictions


class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim):
        super(WeightedMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dim

        self.Q = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1)
        )
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.1)
        )
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())


class BaseTrainer(object):
    def __init__(self):
        self.args = None

        self.n_users = None
        self.n_items = None

        self.net = None
        self.optimizer = None
        self.metrics = None
        self.golden_metric = "Recall@50"

    @staticmethod
    def minibatch(*tensors, **kwargs):
        """Mini-batch generator for pytorch tensor."""
        batch_size = kwargs.get('batch_size', 128)

        if len(tensors) == 1:
            tensor = tensors[0]
            for i in range(0, len(tensor), batch_size):
                yield tensor[i : i + batch_size]
        else:
            for i in range(0, len(tensors[0]), batch_size):
                yield tuple(x[i : i + batch_size] for x in tensors)

    @staticmethod
    def mult_ce_loss(data, logits):
        """Multi-class cross-entropy loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs * data

        instance_data = data.sum(1)
        instance_loss = loss.sum(1)
        # Avoid divide by zeros.
        res = instance_loss / (instance_data + 0.1)  # PSILON)
        return res

    @staticmethod
    def weighted_mse_loss(data, logits, weight_pos=1, weight_neg=0):
        """Mean square error loss."""
        weights = torch.ones_like(data) * weight_neg
        weights[data > 0] = weight_pos
        res = weights * (data - logits) ** 2
        return res.sum(1)
    
    @staticmethod
    def weighted_regularized_mse_loss(U, I, data, logits, regu_coef=0.01, weight_pos=1, weight_neg=0):
        weights = torch.ones_like(data) * weight_neg
        weights[data > 0] = weight_pos
        res = (weights * (data - logits) ** 2).sum() + regu_coef * (U.norm()**2 + I.norm()**2)
        return res
    @staticmethod
    def _array2sparsediag(x):
        values = x
        indices = np.vstack([np.arange(x.size), np.arange(x.size)])

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = [x.size, x.size]

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    @property
    def _initialized(self):
        return self.net is not None

    def _initialize(self):
        """Initialize model and optimizer."""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        """Generate a top-k recommendation (ranked) list."""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def train_epoch(self, data):
        """Train model for one epoch"""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def train_epoch_wrapper(self, train_data, epoch_num):
        """Wrapper for train_epoch with some logs."""
        epoch_loss = self.train_epoch(train_data)

    def evaluate_epoch(self, train_data, test_data, epoch_num):
        """Evaluate model performance on test data."""

        n_rows = train_data.shape[0]
        n_evaluate_users = test_data.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len], dtype=np.float32)

        recommendations = self.recommend(train_data, top_k=100)

        valid_rows = list()
        for i in range(train_data.shape[0]):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = test_data[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            metric_results = list()
            for metric in self.metrics:
                result = metric(targets, recs)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)

        # Average evaluation results by user.
        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

    def fit(self, train_data, test_data):
        """Full model training loop."""
        if not self._initialized:
            self._initialize()

        if self.args.save_feq > self.args.epochs:
            raise ValueError(
                "Model save frequency should be smaller than" " total training epochs."
            )

        start_epoch = 1
        best_checkpoint_path = ""
        best_perf = 0.0
        for epoch_num in range(start_epoch, self.args.epochs + 1):
            # Train the model.
            self.train_epoch_wrapper(train_data, epoch_num)
            if epoch_num % self.args.save_feq == 0:
                result = self.evaluate_epoch(train_data, test_data, epoch_num)

        # Load best model and evaluate on test data.
        self.restore(best_checkpoint_path)
        self.evaluate_epoch(train_data, test_data, -1)
        return

    def restore(self, path):
        return


class WMFTrainer(BaseTrainer):
    def __init__(
        self,
        n_users,
        n_items,
        device,
        hidden_dim,
        lr,
        weight_decay,
        batch_size,
        weight_pos,
        weight_neg,
        verbose=False,
    ):
        super(WMFTrainer, self).__init__()
        self.device = device
        #
        self.n_users = n_users
        self.n_items = n_items
        #
        self.hidden_dim = hidden_dim
        #
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        #
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        #
        self.verbose = verbose

        pass

    def _initialize(self):
        self.net = WeightedMF(
            n_users=self.n_users, n_items=self.n_items, hidden_dim=self.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.dim = self.net.dim

    def fit_adv(self, data_tensor, epoch_num, unroll_steps):
        self._initialize()

        import higher

        if not data_tensor.requires_grad:
            raise ValueError(
                "To compute adversarial gradients, data_tensor "
                "should have requires_grad=True."
            )
        #
        data_tensor = data_tensor.to(self.device)
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)
        #
        model = self.net.to(self.device)
        #
        for i in range(1, epoch_num - unroll_steps + 1):
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0

            for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                # Compute loss

                # TODO detach()
                loss = self.weighted_mse_loss(
                    data=data_tensor[batch_idx].detach(),
                    logits=model(user_id=batch_idx),
                    weight_pos=self.weight_pos,
                    weight_neg=self.weight_neg,
                ).sum()
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with higher.innerloop_ctx(model, self.optimizer) as (fmodel, diffopt):
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                    # Compute loss
                    # ===========warning=================

                    loss = self.weighted_mse_loss(
                        data=data_tensor[batch_idx],
                        logits=fmodel(user_id=batch_idx),
                        weight_pos=self.weight_pos,
                        weight_neg=self.weight_neg,
                    ).sum()
                    # ====================================
                    epoch_loss += loss.item()
                    diffopt.step(loss)

            fmodel.eval()
            predictions = fmodel()
        return predictions  # adv_loss  # .item(), adv_grads[-n_fakes:, ]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in self.minibatch(
                idx_list, batch_size=self.args.valid_batch_size
            ):
                batch_data = data[batch_idx].toarray()

                preds = model(user_id=batch_idx)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations


class BaseGenerator(nn.Module):
    def __init__(self, device, input_dim):
        super(BaseGenerator, self).__init__()
        #
        self.input_dim = input_dim
        self.device = device

        """helper_tensor"""

        self.epsilon = torch.tensor(1e-4).to(self.device)  # 计算boundary
        self.helper_tensor = torch.tensor(2.5).to(device)
        pass

    def project(self, fake_tensor):
        fake_tensor.data = torch.round(fake_tensor)
        # fake_tensor.data = torch.where(fake_tensor < 1, torch.ones_like(fake_tensor).to(self.device), fake_tensor)
        fake_tensor.data = torch.where(
            fake_tensor < 0, torch.zeros_like(fake_tensor).to(self.device), fake_tensor
        )
        fake_tensor.data = torch.where(
            fake_tensor > 5, torch.tensor(5.0).to(self.device), fake_tensor
        )
        #
        return fake_tensor

    def forward(self, input):
        raise NotImplementedError


class RecsysGenerator(BaseGenerator):
    def __init__(self, device, init_tensor):
        super(RecsysGenerator, self).__init__(device, init_tensor.shape[1])
        """
        fake_parameter
        """
        fake_tensor = init_tensor.clone().detach().requires_grad_(True)
        self.fake_parameter = torch.nn.Parameter(fake_tensor, requires_grad=True)
        self.register_parameter("fake_tensor", self.fake_parameter)
        pass

    def forward(self, input=None):
        return self.project(self.fake_parameter * (input > 0))

class WRMFTrainer(BaseTrainer):
    def __init__(
        self,
        n_users,
        n_items,
        device,
        hidden_dim,
        regu_coef,
        lr,
        batch_size,
        weight_pos,
        weight_neg,
        verbose=False,
    ):
        super(WRMFTrainer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.hidden_dim = hidden_dim
        self.regu_coef = regu_coef
        self.lr = lr
        self.batch_size = batch_size
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.verbose = verbose

    def _initialize(self):
        self.net = WeightedMF(
            n_users=self.n_users, n_items=self.n_items, hidden_dim=self.hidden_dim
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr
        )
        
    def fit_adv(self, data_tensor, epoch_num, unroll_steps=1):
        self._initialize()
        
        import higher
        
        if not data_tensor.requires_grad:
            raise ValueError(
                "To compute adversarial gradients, data_tensor "
                "should have requires_grad=True."
            )
            
        data_tensor = data_tensor.to(self.device)
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)
        model = self.net.to(self.device)
        
        for i in range(1, epoch_num - unroll_steps + 1):
            np.random.shuffle(idx_list)
            model.train()
            for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                loss = self.weighted_regularized_mse_loss(
                    U=model.P,
                    I=model.Q,
                    data=data_tensor[batch_idx].detach(),
                    logits=model(user_id=batch_idx),
                    regu_coef=self.regu_coef,
                    weight_pos=self.weight_pos,
                    weight_neg=self.weight_neg,
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        #     # print(f"training surSys Epoch {i} loss: {loss.item()}")
        
        with higher.innerloop_ctx(model, self.optimizer) as (fmodel, diffopt):
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                np.random.shuffle(idx_list)
                fmodel.train()
                for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
        #             # Compute loss
        #             # ===========warning=================

                    loss = self.weighted_regularized_mse_loss(
                        U=model.P,
                        I=model.Q,
                        data=data_tensor[batch_idx],
                        logits=fmodel(user_id=batch_idx),
                        regu_coef=self.regu_coef,
                        weight_pos=self.weight_pos,
                        weight_neg=self.weight_neg,
                    )
        #             # ====================================
                    diffopt.step(loss)
                    
            fmodel.eval()
            predictions = fmodel()
        return predictions