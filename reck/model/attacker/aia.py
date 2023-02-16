# TODO
from .base import BaseAttacker
import math
import torch
from torch import nn
import numpy as np
from ...default import MODEL
from ...utils import pick_optim, VarDim


class Aush(BaseAttacker):
    def __init__(self, **config) -> None:
        super().__init__()

        self.selected_ids = config['selected_ids']
        self.attack_num = config['attack_num']
        self.filler_num = config['filler_num']
        self.dataset = config['dataset']
        self.n_items = config['dataset'].info_describe()['n_items']
        self.config = config
        self.device = config['device']
        self.ZR_ratio = config['ZR_ratio']
        self.train_array = config['dataset'].info_describe()['train_mat']
        self.build_network()

    def build_network(self):
        pass

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
                "d_losses": (float, []),
                "g_loss_rec_l": (float, []),
                "g_loss_shilling_l": (float, []),
                "g_loss_gan_l": (float, []),
            }
        }

    def train_step(self, **config):
        target_id_list = config['target_id_list']

        target_users = np.where(self.train_array[:, target_id_list] == 0)[0]
        attack_target = np.zeros((len(target_users), self.n_items))
        attack_target[:, target_id_list] = 1.0
        attack_target = (
            torch.from_numpy(attack_target).type(torch.float32).to(self.device)
        )

        for idx, dp in enumerate(
            self.dataset.generate_batch(
                filler_num=self.filler_num, selected_ids=self.selected_ids, **config
            )
        ):
            batch_set_idx = dp['users']
            real_profiles = dp['users_mat']

        return (
            np.mean(d_losses),
            np.mean(g_loss_rec_l),
            np.mean(g_loss_shilling_l),
            np.mean(g_loss_gan_l),
        )

    def generate_fake(self, **kwargs):
        pass


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
        time_st = time.time()
        epoch_loss = self.train_epoch(train_data)
        print(
            "Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - time_st, epoch_num, epoch_loss
            )
        )

    def evaluate_epoch(self, train_data, test_data, epoch_num):
        """Evaluate model performance on test data."""
        t1 = time.time()

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
        print("Loading best model checkpoint.")
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
            t1 = time.time()
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
            if self.verbose:
                print(
                    "Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                        time.time() - t1, i, epoch_loss
                    ),
                    flush=True,
                )

        with higher.innerloop_ctx(model, self.optimizer) as (fmodel, diffopt):
            if self.verbose:
                print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
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
                if self.verbose:
                    print(
                        "Training (higher mode) [{:.1f} s],"
                        " epoch: {}, loss: {:.4f}".format(
                            time.time() - t1, i, epoch_loss
                        ),
                        flush=True,
                    )
            #
            if self.verbose:
                print(
                    "Finished surrogate model training,"
                    " {} copies of surrogate model params.".format(
                        len(fmodel._fast_params)
                    ),
                    flush=True,
                )

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
