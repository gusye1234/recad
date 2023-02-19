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
        self.train_data_array = config['dataset'].info_describe()['train_mat']
        self.build_network()

    def build_network(self):
        self.netG = AushGenerator(input_dim=self.n_items).to(self.device)
        self.G_optimizer = pick_optim(self.config['optim_g'])(
            self.netG.parameters(), lr=self.config['lr_g']
        )

        self.netD = AushDiscriminator(input_dim=self.n_items).to(self.device)
        self.D_optimizer = pick_optim(self.config['optim_d'])(
            self.netD.parameters(), lr=self.config['lr_d']
        )

    @classmethod
    def from_config(cls, **kwargs):
        args = list(MODEL['attacker']['aush'])
        user_args = "dataset"
        return super().from_config("aush", args, user_args, kwargs)

    def info_describe(self):
        return super().info_describe()

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

    def sample_fillers(self, real_profiles, target_id_list):
        fillers = np.zeros_like(real_profiles)
        filler_pool = (
            set(range(self.n_items)) - set(self.selected_ids) - set(target_id_list)
        )

        # print(filler_pool)
        # print(np.argwhere(real_profiles[0] > 0).flatten())
        # print(list(set(np.argwhere(real_profiles[0] > 0).flatten()) & filler_pool))
        filler_sampler = lambda x: np.random.choice(
            size=self.filler_num,
            replace=True,
            a=list(set(np.argwhere(x > 0).flatten()) & filler_pool),
        )

        sampled_cols = np.array(
            [filler_sampler(x) for x in real_profiles], dtype="int64"
        )

        sampled_rows = np.repeat(np.arange(real_profiles.shape[0]), self.filler_num)
        fillers[sampled_rows, sampled_cols.flatten()] = 1
        return fillers

    def train_step(self, **config):
        target_id_list = config['target_id_list']
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        g_loss_rec_l = []
        g_loss_shilling_l = []
        g_loss_gan_l = []
        d_losses = []
        for idx, dp in enumerate(
            self.dataset.generate_batch(
                filler_num=self.filler_num, selected_ids=self.selected_ids, **config
            )
        ):
            batch_set_idx = dp['users']
            real_profiles = dp['users_mat']

            valid_labels = (
                torch.ones_like(batch_set_idx)
                .type(torch.float)
                .to(self.device)
                .reshape(len(batch_set_idx), 1)
            )
            fake_labels = (
                torch.zeros_like(batch_set_idx)
                .type(torch.float)
                .to(self.device)
                .reshape(len(batch_set_idx), 1)
            )
            fillers_mask = self.sample_fillers(
                real_profiles.cpu().numpy(), target_id_list
            )

            # selected
            selects_mask = np.zeros_like(fillers_mask)
            selects_mask[:, self.selected_ids] = 1.0
            # target
            target_patch = np.zeros_like(fillers_mask)
            target_patch[:, self.selected_ids] = 5.0
            # ZR_mask
            ZR_mask = (real_profiles.cpu().numpy() == 0) * selects_mask
            pools = np.argwhere(ZR_mask)
            np.random.shuffle(pools)
            pools = pools[: math.floor(len(pools) * (1 - self.ZR_ratio))]
            ZR_mask[pools[:, 0], pools[:, 1]] = 0

            fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
            selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
            target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
            ZR_mask = torch.tensor(ZR_mask).type(torch.float).to(self.device)

            input_template = torch.mul(real_profiles, fillers_mask)
            # ----------generate----------
            self.netG.eval()
            gen_output = self.netG(input_template)
            gen_output = gen_output.detach()
            # ---------mask--------
            selected_patch = torch.mul(gen_output, selects_mask)
            middle = torch.add(input_template, selected_patch)
            fake_profiles = torch.add(middle, target_patch)
            # --------Discriminator------
            # forward
            self.D_optimizer.zero_grad()
            self.netD.train()
            d_valid_labels = self.netD(real_profiles * (fillers_mask + selects_mask))
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            # loss
            D_real_loss = bce_loss(d_valid_labels, valid_labels)
            D_fake_loss = bce_loss(d_fake_labels, fake_labels)
            d_loss = 0.5 * (D_real_loss + D_fake_loss)
            d_loss.backward()
            self.D_optimizer.step()
            self.netD.eval()
            d_losses.append(d_loss.item())

            # ---------train G-------
            self.netG.train()
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            g_loss_gan = bce_loss(d_fake_labels, valid_labels)
            g_loss_shilling = mse_loss(fake_profiles * selects_mask, selects_mask * 5.0)
            g_loss_rec = mse_loss(
                fake_profiles * selects_mask * ZR_mask,
                selects_mask * input_template * ZR_mask,
            )
            g_loss = g_loss_gan + g_loss_rec + g_loss_shilling

            g_loss_rec_l.append(g_loss_rec.item())
            g_loss_shilling_l.append(g_loss_shilling.item())
            g_loss_gan_l.append(g_loss_gan.item())
            self.G_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()
        return (
            np.mean(d_losses),
            np.mean(g_loss_rec_l),
            np.mean(g_loss_shilling_l),
            np.mean(g_loss_gan_l),
        )

    def generate_fake(self, **kwargs):
        target_id_list = kwargs['target_id_list']
        mask_array = (self.train_data_array > 0).astype(np.float)
        mask_array[:, self.selected_ids + target_id_list] = 0
        available_idx = np.where(np.sum(mask_array, 1) >= self.filler_num)[0]
        available_idx = np.random.permutation(available_idx)
        idx = available_idx[np.random.randint(0, len(available_idx), self.attack_num)]
        idx = list(idx)

        real_profiles = self.train_data_array[idx, :]
        # sample fillers
        fillers_mask = self.sample_fillers(real_profiles, target_id_list)
        # selected
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.0
        # target
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, target_id_list] = 5.0

        # Generate
        real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
        fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
        selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
        target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
        input_template = torch.mul(real_profiles, fillers_mask)
        self.netG.eval()
        gen_output = self.netG(input_template)
        selected_patch = torch.mul(gen_output, selects_mask)
        middle = torch.add(input_template, selected_patch)
        fake_profiles = torch.add(middle, target_patch)
        fake_profiles = fake_profiles.detach().cpu().numpy()
        # fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        # selected patches
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 5] = 5
        selected_patches[selected_patches < 1] = 1
        fake_profiles[:, self.selected_ids] = selected_patches

        return fake_profiles

    def forward(self):
        pass


class AushGenerator(nn.Module):
    def __init__(self, input_dim):
        super(AushGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input) * 5


class AushDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(AushDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.Sigmoid(),
            nn.Linear(150, 150),
            nn.Sigmoid(),
            nn.Linear(150, 150),
            nn.Sigmoid(),
            nn.Linear(150, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
