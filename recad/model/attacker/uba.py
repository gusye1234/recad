from .base import BaseAttacker
import math
import torch
from torch import nn
import numpy as np
from functools import partial
from ...default import MODEL
from ...utils import pick_optim, VarDim, filler_filter_mat
import random

class UBA(BaseAttacker):
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
        self.target_user_id = config['target_user_ids']
        self.budget = config['budget']
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
        args = list(MODEL['attacker']['uba'])
        user_args = "dataset"
        return super().from_config("uba", args, user_args, kwargs)

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


    def prob_matrix_compute(self, add_num):
        train_data_array = self.train_data_array
        row_max_values = np.max(train_data_array, axis=1)
        # print(row_max_values)
        target_user_id = self.target_user_id
        for user_id in target_user_id:
            row_to_duplicate = train_data_array[user_id, :]
            non_zero_positions = np.where(row_to_duplicate != 0)[0]
            for pos in non_zero_positions:
                row_to_duplicate[pos] = random.randint(1, 5)
            row_to_duplicate[self.selected_ids[0]] = 5
            for _ in range(add_num):
                train_data_array = np.vstack([train_data_array, row_to_duplicate])
        # print(train_data_array.shape)
        M, N = train_data_array.shape
        expanded_matrix = np.zeros((M + N, M + N))
        expanded_matrix[:M, M : M + N] = train_data_array
        expanded_matrix[M : M + N, :M] = train_data_array.T
        A3_matrix = expanded_matrix * expanded_matrix * expanded_matrix
        A3_matrix = A3_matrix[:M, M : M + N]
        target_user_mat = A3_matrix[target_user_id,:]
        N = 10  
        sorted_column_indices = np.argsort(-target_user_mat, axis=1)
        top_n_column_indices = sorted_column_indices[:, :N]
        # print(top_n_column_indices)
        position_list = []
        for row in top_n_column_indices:
            if self.selected_ids[0] in row:
                position = np.where(row == self.selected_ids[0])[0][0] + 1
                position_list.append(position)
            else:
                position = 0
                position_list.append(position)
        
        # print(position_list)
        return position_list


    def budget_matrix(self):
        prob_mat = np.zeros((len(self.target_user_id), self.budget))
        lists_to_insert = []
        for i in range(self.budget):
            add_numm = i+1
            temp_consume = []
            for i in range(len(self.target_user_id)):
                temp_consume.append(0)
            for i in range(10):
                prob_list = self.prob_matrix_compute(add_numm)
                for idx, i in enumerate(prob_list):
                    if i != 0:
                        temp_consume[idx] += 1
                    else:
                        temp_consume[idx] += 0
            lists_to_insert.append(temp_consume)
        for i, lst in enumerate(lists_to_insert):
            prob_mat[:, i] = lst
        
        prob_mat = prob_mat / 10
        # print(prob_mat)
        return prob_mat

    def DSP_part(self, w, v, n, c):
        rec = []
        for i in range(len(n)):
            rec.append([])
        sum = 0
        for i in n:
            for j in range(i):
                rec[sum].append(0)
            sum += 1
        mydict = {}
        dict_list = []
        dp = [0 for _ in range(c + 1)]
        for i in range(1, len(w) + 1):
            for j in reversed(range(1, c + 1)):
                for k in range(n[i - 1]):
                    if j - w[i - 1][k] >= 0:
                        dp[j] = max(dp[j], dp[j - w[i - 1][k]] + v[i - 1][k])
                        if dp[j - w[i - 1][k]] + v[i - 1][k] >= dp[j]:
                            if (j,round(dp[j],2)) in mydict.keys():
                                mydict[(j,round(dp[j],2))].append((i,k))
                            else:
                                mydict[(j,round(dp[j],2))] = []
                                mydict[(j,round(dp[j],2))].append((i,k))
                            dict_list.append((i,k,j,round(dp[j],2)))
        return mydict,dict_list,dp[c].max()


    def DSP(self):
        print('Finish compute A3 matrix...')
        # to be continued
        prob_mat = self.budget_matrix()
        data_index = self.target_user_id
        x = prob_mat
        index = 0
        weight = []
        user_id = []
        v = []
        for user in data_index:
            temp_w = []
            temp_v = []
            for i in range(6):
                if x[index][i] != 0:
                    temp_w.append(i)
                    temp_v.append(x[index][i])
            if len(temp_w) != 0:
                weight.append(temp_w)
                v.append(temp_v)
                user_id.append(user)
            index += 1
        lenth = []
        for i in weight:
            lenth.append(len(i))
        c = 100
        w = weight
        v = v
        n = lenth
        mydict,dict_list,max_v = self.DSP_part(w, v, n, c)
        max_v = round(max_v,1)
        for i in dict_list:
            if i[2]==100 and i[3] == max_v:
                temp_list = i
        value = max_v
        c = 100
        team = self.attack_num
        order = temp_list[1]
        dict_final = {}
        sum_list = []
        while (team,order,c,value) in dict_list:
            # print(((team,order,c,value),w[team-1][order],v[team-1][order],user_id[team-1]))
            dict_final[user_id[team-1]] = w[team-1][order]
            sum_list.append(v[team-1][order])
            c_d = w[team-1][order]
            value_d = v[team-1][order]
            c = c - c_d
            value=value- value_d
            c = round(c,2)
            value = round(value,2)
            temp = []
            for i in dict_list:
                if i[2]==c and i[3] == value:
                    temp.append(i)
            sign = 0
            for j in range(len(temp)):
                if sign == 0:
                    if temp[j][0] != team:
                        team = temp[j][0]
                        order = temp[j][1]
                        sign = 1
        user_list = []
        for key in dict_final.keys():
            for j in range(dict_final[key]):
                user_list.append(key)
        return user_list

    def train_step(self, **config):
        target_id_list = config['target_id_list']
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        g_loss_rec_l = []
        g_loss_shilling_l = []
        g_loss_gan_l = []
        d_losses = []

        index_filter = partial(
            filler_filter_mat,
            selected_ids=self.selected_ids,
            filler_num=self.filler_num,
            target_id_list=target_id_list,
        )
        for idx, dp in enumerate(
            self.dataset.generate_batch(user_filter=index_filter, **config)
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
        mask_array = (self.train_data_array > 0).astype('float')
        mask_array[:, self.selected_ids + target_id_list] = 0
        available_idx = np.where(np.sum(mask_array, 1) >= self.filler_num)[0]
        available_idx = np.random.permutation(available_idx)
        idx = available_idx[np.random.randint(0, len(available_idx), self.attack_num)]
        idx = list(idx)
        idx = self.DSP()
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
