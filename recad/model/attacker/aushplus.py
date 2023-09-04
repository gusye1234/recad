import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .base import BaseAttacker
from .aia import AIA, RecsysGenerator, partial_update, aia_attack_loss
from ...default import MODEL
from ...utils import pick_optim, VarDim


class AushPlus(AIA):
    def __init__(self, **config) -> None:
        super().__init__(**config)
        self.pretrained = False

    @classmethod
    def from_config(cls, **kwargs):
        print(AushPlus.__bases__)
        args = list(MODEL['attacker']['aushplus'])
        user_args = "dataset"
        return super(AIA, cls).from_config("aushplus", args, user_args, kwargs)

    def build_network(self):
        sampled_idx = np.random.choice(range(self.n_users), self.attack_num)
        templates = self.train_array[sampled_idx]
        for idx, template in enumerate(templates):
            fillers = np.where(template)[0]
            np.random.shuffle(fillers)
            for iid in fillers[self.filler_num :]:
                templates[idx][iid] = 0.0
        self.real_template = torch.tensor(templates).type(torch.float).to(self.device)
        self.netG = DiscretGenerator_AE_1(self.device, p_dims=[self.n_items, 125]).to(
            self.device
        )
        self.G_optimizer = pick_optim(self.config['optim_g'])(
            self.netG.parameters(), lr=self.config['lr_g']
        )

        self.netD = Discriminator(input_dim=self.n_items).to(self.device)
        self.D_optimizer = pick_optim(self.config['optim_g'])(
            self.netD.parameters(), lr=self.config['lr_d']
        )

    def input_describe(self):
        return {"train_step": {"target_id_list": (list, VarDim())}}

    def output_describe(self):
        return {"train_step": {"g_adv": (float, []), "g_rec": (float, [])}}

    def train_G(self, adv=True, attack=True, target_id_list=[]):
        self.netG.train()
        fake_tensor_distribution, fake_tensor = self.netG(self.real_template)
        G_adv_loss = torch.tensor(0.0).to(self.device)
        if adv:
            G_adv_loss = nn.BCELoss(reduction='mean')(
                self.netD(fake_tensor),
                torch.ones(fake_tensor.shape[0], 1).to(self.device),
            )

        real_labels_flatten = self.real_template.flatten().type(torch.long)

        MSELoss = nn.MSELoss()(
            fake_tensor.flatten()[real_labels_flatten > 0],
            real_labels_flatten[real_labels_flatten > 0],
        )

        G_attack_loss = torch.tensor(0.0).to(self.device)
        if attack:
            sur_predictions = self.get_sur_predictions(fake_tensor)
            for target_id in target_id_list:
                target_users = np.where(self.train_array[:, target_id] == 0)[0]
                # target_users 所有对target id 没有评分的 users

                attack_target = np.zeros((len(target_users), self.n_items))
                # attack_target 张成的一个 矩阵 (0)
                attack_target[:, target_id] = 1.0
                # 另矩阵中全部的 target_id == 1.0
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
                G_attack_loss += G_loss_sub
            G_attack_loss = G_attack_loss / 10

        G_loss = G_adv_loss + G_attack_loss

        partial_update(G_loss, self.G_optimizer)
        self.netG.eval()

        mean_score = fake_tensor[fake_tensor > 0].mean().item()
        return (G_loss.item(), MSELoss.item(), G_attack_loss.item(), mean_score)

    def train_D(self):
        self.netD.train()

        _, fake_tensor = self.netG(self.real_template)
        fake_tensor = fake_tensor.detach()

        D_loss_list = []
        loss_bce = nn.BCELoss()
        for dp in self.dataset.generate_batch():
            real_tensor = dp['users_mat']
            real_tensor = real_tensor.to(self.device)[: self.attack_num]
            # forward
            self.D_optimizer.zero_grad()
            D_real = self.netD(real_tensor)
            D_fake = self.netD(fake_tensor)
            # loss
            D_real_loss = loss_bce(D_real, torch.ones_like(D_real).to(self.device))
            D_fake_loss = loss_bce(D_fake, torch.zeros_like(D_fake).to(self.device))
            D_loss = D_real_loss + D_fake_loss
            # backward
            D_loss.backward()
            self.D_optimizer.step()

            D_loss_list.append(D_loss.item())
            #
            # break
        self.netD.eval()

        return np.mean(D_loss_list)

    def pretrain_G(self):
        self.netG.train()
        G_loss_list = []
        loss_cross = nn.CrossEntropyLoss()
        loss_mse = nn.MSELoss()
        for dp in self.dataset.generate_batch():
            real_tensor = dp['users_mat']
            # forward
            fake_tensor_distribution, fake_tensor = self.netG(real_tensor)
            # crossEntropy loss
            real_labels_flatten = real_tensor.flatten().type(torch.long)
            fake_logits_flatten = fake_tensor_distribution.reshape([-1, 5])
            G_rec_loss = loss_cross(
                fake_logits_flatten[real_labels_flatten > 0],
                real_labels_flatten[real_labels_flatten > 0] - 1,
            )
            G_loss = G_rec_loss
            MSELoss = loss_mse(
                fake_tensor.flatten()[real_labels_flatten > 0],
                real_labels_flatten[real_labels_flatten > 0],
            )
            # backword
            partial_update(G_loss, self.G_optimizer)
            G_loss_list.append(G_loss.item())
        self.netG.eval()
        return np.mean(G_loss_list)

    def train_step(self, **config):
        target_id_list = config['target_id_list']
        if not self.pretrained:
            for i in range(self.config['pretrain_epoch_g']):
                self.pretrain_G()
            for i in range(self.config['pretrain_epoch_g']):
                self.train_D()
            self.pretrained = True
        for _ in range(self.config["epoch_gan_d"]):
            self.train_D()
        for _ in range(self.config["epoch_gan_g"]):
            _, _, G_adv_loss, _ = self.train_G(
                adv=True, attack=False, target_id_list=target_id_list
            )
            _, fake_tensor = self.netG(self.real_template)
        for epoch_surrogate in range(self.config['epoch_surrogate']):
            _, _, G_rec_loss, _ = self.train_G(
                adv=False, attack=True, target_id_list=target_id_list
            )
            print(epoch_surrogate)

        return (G_adv_loss, G_rec_loss)

    def generate_fake(self, **config):
        target_id_list = config['target_id_list']
        with torch.no_grad():
            _, fake_tensor = self.netG(self.real_template)
            rate = int(self.attack_num / len(target_id_list))
            for i in range(len(target_id_list)):
                fake_tensor[i * rate : (i + 1) * rate, target_id_list[i]] = 5
        return fake_tensor.detach().cpu().numpy()


class HeaviTanh(torch.autograd.Function):
    """
    Approximation of the heaviside step function as
    h(x,k) = \frac{1}{2} + \frac{1}{2} \text{tanh}(k x)
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x, k)

        def heaviside(data):
            """
            A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
            that truncates numbers <= 0 to 0 and everything else to 1.
            .. math::
                H[n]=\\begin{cases} 0, & n <= 0, \\ 1, & n \g 0, \end{cases}
            """
            return torch.where(
                data <= torch.zeros_like(data),
                torch.zeros_like(data),
                torch.ones_like(data),
            )

        return heaviside(x)  # 0.5 + 0.5 * torch.tanh(k * x)

    @staticmethod
    def backward(ctx, dy):
        (
            x,
            k,
        ) = ctx.saved_tensors
        dtanh = 1 - (x * k).tanh().pow(2)
        return dy * dtanh, None


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


class BaseDiscretGenerator_1(BaseGenerator):
    def __init__(self, device, input_dim):
        super(BaseDiscretGenerator_1, self).__init__(device, input_dim)

        # self.min_boundary_value = torch.nn.Parameter(torch.rand([self.input_dim]), requires_grad=True)
        self.min_boundary_value = torch.nn.Parameter(
            torch.ones([self.input_dim]), requires_grad=True
        )
        self.register_parameter("min_boundary_value", self.min_boundary_value)

        # self.interval_lengths = torch.nn.Parameter(torch.rand([self.input_dim, 3]), requires_grad=True)
        self.interval_lengths = torch.nn.Parameter(
            torch.ones([self.input_dim, 3]), requires_grad=True
        )
        self.register_parameter("interval_lengths", self.interval_lengths)

        pass

    def forward(self, input):
        # fake_tensor = (self.main(input) * self.helper_tensor) + self.helper_tensor
        # # project
        # fake_dsct_distribution, fake_dsct_value = self.project(fake_tensor)
        # return fake_dsct_value
        raise NotImplementedError

    def project_old(self, fake_tensor):
        boundary_values = self.get_boundary_values()

        fake_dsct_distribution = []

        for iid in range(self.input_dim):
            cur_dsct_distribution = []
            for rating_dsct in range(5):
                rating_prob = torch.ones(fake_tensor.shape[0]).to(self.device)
                for boundary_idx in range(4):
                    rating_prob *= self.is_in_interval(
                        rating_dsct,
                        boundary_idx,
                        fake_tensor[:, iid],
                        boundary_values[iid][boundary_idx],
                    )
                cur_dsct_distribution += [rating_prob]
            fake_dsct_distribution += [
                torch.cat([torch.unsqueeze(p, 1) for p in cur_dsct_distribution], 1)
            ]
        fake_dsct_distribution = torch.cat(
            [torch.unsqueeze(p, 1) for p in fake_dsct_distribution], 1
        )

        fake_dsct_value = torch.matmul(
            fake_dsct_distribution,
            torch.tensor(np.arange(1.0, 6.0)).type(torch.float32).to(self.device),
        )
        return fake_dsct_distribution, fake_dsct_value

    def project(self, fake_tensor):
        Heaviside = HeaviTanh.apply

        boundary_values = self.get_boundary_values()

        cnt_ratings = fake_tensor.flatten()
        iids = (
            np.expand_dims(np.arange(self.input_dim), 0)
            .repeat(fake_tensor.shape[0], axis=0)
            .flatten()
        )
        boundary_values_per_rating = boundary_values[iids]

        def _project_helper(ratings, boundary_values_input):
            def get_target_dst_rating_prob(
                target_dst_rating, input_cnt_rating, boundary_values, device
            ):
                # boundary_values = boundary_values.reshape([-1, 4])
                # input_cnt_rating = input_cnt_rating.reshape([-1])
                rating_prob = torch.ones(input_cnt_rating.shape[0]).to(self.device)
                for boundary_idx in range(4):
                    """
                    :param target_dst_rating: r_i_j
                    :param boundary_idx: k
                    :param input_cnt_rating: a_i_j
                    :param boundary_value: b_j_k
                    :return:
                    """
                    #
                    p_1 = torch.sign(
                        target_dst_rating - boundary_idx - torch.tensor(0.5).to(device)
                    )
                    #
                    p_2 = input_cnt_rating - boundary_values[:, boundary_idx]
                    #
                    rating_prob *= Heaviside(p_1 * p_2, torch.tensor(1.0).to(device))
                return rating_prob

            cur_dsct_distribution = []
            for rating_dsct in range(5):
                p = get_target_dst_rating_prob(
                    rating_dsct, ratings, boundary_values_input, self.device
                )
                cur_dsct_distribution += [p]
            dsct_distribution = torch.cat(
                [torch.unsqueeze(p, 1) for p in cur_dsct_distribution], 1
            )
            return dsct_distribution

        fake_dsct_distribution = _project_helper(
            cnt_ratings, boundary_values_per_rating
        ).reshape([-1, self.input_dim, 5])

        fake_dsct_value = torch.matmul(
            fake_dsct_distribution,
            torch.tensor(np.arange(1.0, 6.0)).type(torch.float32).to(self.device),
        )
        return fake_dsct_distribution, fake_dsct_value

    def get_boundary_values(self):
        boundary_values = torch.zeros([self.input_dim, 4]).to(self.device)
        boundary_values[:, 0] = self.min_boundary_value
        for i in range(1, 4):
            cur_interval_length = (
                torch.relu(self.interval_lengths[:, i - 1]) + self.epsilon
            )
            boundary_values[:, i] = boundary_values[:, i - 1] + cur_interval_length
        return boundary_values

    def is_in_interval(self, rating_dsct, boundary_idx, rating_cnt, boundary_value):
        tensor_aux_0_5 = torch.tensor(0.5).to(self.device)
        tensor_aux_1 = torch.tensor(1.0).to(self.device)
        Heaviside = HeaviTanh.apply
        """
    
        :param rating_dsct: r_i_j
        :param boundary_idx: k
        :param rating_cnt: a_i_j
        :param boundary_value: b_j_k
        :return:
        """
        #
        p_1 = torch.sign(rating_dsct - boundary_idx - tensor_aux_0_5)
        #
        p_2 = rating_cnt - boundary_value
        #
        return Heaviside(p_1 * p_2, tensor_aux_1)


class DiscretGenerator_AE_1(BaseDiscretGenerator_1):
    def __init__(self, device, p_dims, q_dims=None):
        super(DiscretGenerator_AE_1, self).__init__(device, input_dim=p_dims[0])
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.p_dims + self.q_dims[1:]
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
            ]
        )
        # self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        # h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
            else:
                h = torch.nn.Tanh()(h)

        fake_tensor = (h * self.helper_tensor) + self.helper_tensor
        # project

        fake_dsct_distribution, fake_dsct_value = self.project(fake_tensor)

        sampled_filler = input > 0

        # sampled_filler = (torch.rand(fake_dsct_value.shape) < (90 / 1924)).float()

        # filler_num = np.sum(sampled_filler.detach().cpu().numpy()*(fake_dsct_value.detach().cpu().numpy()>0),1).mean()
        # if filler_num<90:
        return fake_dsct_distribution, fake_dsct_value * sampled_filler

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
