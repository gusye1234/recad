from .base import BaseAttacker
import torch
from torch import nn
import numpy as np
from ...default import MODEL
from .aia import WRMFTrainer, WMFTrainer
from ...utils import VarDim, pick_optim


class DADA(BaseAttacker):
    def __init__(self, **config) -> None:
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.n_users = config['dataset'].info_describe()['n_users']
        self.n_items = config['dataset'].info_describe()['n_items']
        self.attack_num = round(config['attack_num_ratio'] * self.n_users)
        self.filler_num = round(config['filler_num_ratio'] * self.n_items)
        self.device = config['device']
        self.surrogate = config["surrogate_model"]
        
        train_dict = config['dataset'].info_describe()['train_dict']
        self.real_profile = self.dict2tensor(train_dict, self.n_users, self.n_items).requires_grad_(True).to(self.device)
        self.build_network()
        
    def dict2tensor(self, train_dict, n_users, n_items):
        train_tensor = torch.zeros((n_users, n_items)).to(self.device)
        for user in train_dict:
            train_tensor[user, train_dict[user]] = 1.0
        return train_tensor
    
    def build_network(self):
        self.fake_tensor = torch.zeros((self.attack_num, self.n_items)).type(torch.float).uniform_().to(self.device)
        self.template = torch.ones((self.attack_num, self.n_items)).type(torch.float).to(self.device)  
        self.net = FakeProfile(self.fake_tensor, self.device, self.filler_num, self.config['threshold']).to(self.device)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr_g'])
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config['lr_g'])
        # self.optimizer = pick_optim(self.config['optim_g'])(
        #     self.parameters(), lr=self.config['lr_g']
        # )
    
    @classmethod
    def from_config(cls, **kwargs):
        print(DADA.__bases__)
        args = list(MODEL['attacker']['dada'])
        user_args = "dataset"
        return super().from_config("dada", args, user_args, kwargs)

    def input_describe(self):
        return {"train_step": {"target_id_list": (list, VarDim())}}

    def output_describe(self):
        return {
            "train_step": {
                "losses": (float, []),
            }
        }
        
    def train_step(self, **config):
        target_id_list = config['target_id_list']
        
        self.net.train()
        fake_profile = self.net(self.template)
        # profile = torch.cat([self.real_profile, fake_profile], dim=0)
        loss = torch.tensor(0.0).to(self.device)
        
        non_attack_sur_predictions = self.get_sur_predictions(None)
        attack_sur_predictions = self.get_sur_predictions(fake_profile)
        DADA_loss = DADALoss(self.device, self.config['topk'], self.config['b1'], self.config['b2'], self.config['lambda'], target_id_list)
        loss = DADA_loss(attack_sur_predictions, non_attack_sur_predictions)

        self.optimizer.zero_grad()
        grad_groups = torch.autograd.grad(
            loss, self.optimizer.param_groups[0]['params'], allow_unused=True
        )
        for para_, grad_ in zip(self.optimizer.param_groups[0]['params'], grad_groups):
            if para_.grad is None:
                print(grad_.sum()*1000000)
                para_.grad = grad_.clone()
                # para_.grad = grad_
            else:
                print(grad_.sum()*1000000)
                para_.grad.data = grad_
                
        self.optimizer.step()

        self.net.eval()
        # print(self.net.fake_parameter)
        print(f'dada loss: {loss}')
        return (loss,)  
    
    def generate_fake(self, **config):
        target_id_list = config['target_id_list']
        with torch.no_grad():
            fake_tensor = self.net(self.template)
            rate = int(self.attack_num / len(target_id_list))
            for i in range(len(target_id_list)):
                fake_tensor[i * rate : (i + 1) * rate, target_id_list[i]] = 1
        return fake_tensor.detach().cpu().numpy()
    
    def get_sur_predictions(self, fake_tensor=None):
        if fake_tensor is None:
            n_users = self.n_users
            data_tensor = self.real_profile
        else:
            n_users = self.n_users + self.attack_num
            data_tensor = torch.cat([self.real_profile, fake_tensor], dim=0)

        surrogate = self.surrogate
        if surrogate == 'WRMF':
            sur_trainer_ = WRMFTrainer(
                n_users=n_users,
                n_items=self.n_items,
                device=self.device,
                hidden_dim=self.config['hidden_dim_s'],
                regu_coef=self.config['regu_coef_s'],
                lr=self.config['lr_s'],
                batch_size=self.config['batch_size_s'],
                weight_pos=self.config['weight_pos_s'],
                weight_neg=self.config['weight_neg_s'],
                verbose=False,
            )
            epoch_num_ = self.config['epoch_s']
            unroll_steps_ = 1
        elif surrogate == 'WMF':
            sur_trainer_ = WMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dim=self.config['hidden_dim_s'],
                device=self.device,
                lr=self.config['lr_s'],
                weight_decay=0.0,
                batch_size=self.config['batch_size_s'],
                weight_pos=self.config['weight_pos_s'],
                weight_neg=self.config['weight_neg_s'],
                verbose=False,
            )
            epoch_num_ = self.config['epoch_s']
            unroll_steps_ = 1
        else:
            raise ValueError(
                f'surrogate model error : {surrogate}',
            )
            
        sur_predictions = sur_trainer_.fit_adv(
            data_tensor=data_tensor, epoch_num=epoch_num_, unroll_steps=unroll_steps_
        )
        return sur_predictions[0:self.n_users]
        
class FakeProfile(nn.Module):
    def __init__(self, init_tensor, device, filler_num, threshold):
        super(FakeProfile, self).__init__()

        self.filler_num = filler_num
        self.threshold = threshold
        self.device = device
        
        fake_tensor = init_tensor.clone().detach().requires_grad_(True)
        self.fake_parameter = torch.nn.Parameter(fake_tensor, requires_grad=True)
        self.register_parameter("fake_tensor", self.fake_parameter)
    
    def forward(self, input=None):
        return self.project(self.fake_parameter * (input > 0))
    
    def project(self, fake_tensor):
        fake_tensor.data = torch.where(
            fake_tensor > self.threshold, torch.ones_like(fake_tensor).type(torch.float).to(self.device), torch.zeros_like(fake_tensor).to(self.device)
        )
        topk_tensor = torch.topk(fake_tensor, self.filler_num, dim=1)
        mask = torch.zeros_like(fake_tensor).to(self.device)
        mask.scatter_(dim=1, index=topk_tensor.indices, value=1)
        fake_tensor.data *= mask
        return fake_tensor
    
class DADALoss(nn.Module):
    def __init__(self, device, topk, b1, b2, lambda_, target_list):
        super(DADALoss, self).__init__()
        
        self.device = device
        self.topk = topk
        self.b1 = b1
        self.b2 = b2
        self.lambda_ = lambda_
        self.target_list = target_list
        
    def forward(self, attack_tensor, non_attack_tensor):
        return self.DictLoss(attack_tensor) + self.lambda_ * self.DivLoss(attack_tensor, non_attack_tensor)
    
    def DictLoss(self, attacked_tensor):
        loss = torch.tensor(0.0).to(self.device)
        
        topk_tensor = torch.topk(attacked_tensor, self.topk, dim=1)
        mask = torch.ones_like(attacked_tensor).to(self.device)
        mask.scatter_(dim=1, index=topk_tensor.indices, value=0)
        target_tensor = (attacked_tensor * mask)[:, self.target_list]  
        scale_tensor = target_tensor / ((target_tensor.max() + 0.1) * self.b1)
        loss = -(1 - self.b1*scale_tensor).log().sum()
        return loss
    
    def DivLoss(self, attacked_tensor, non_attacked_tensor):
        loss = torch.tensor(0.0).to(self.device)
        
        topk_tensor = torch.topk(attacked_tensor, self.topk, dim=1)
        mask = torch.ones_like(attacked_tensor).to(self.device)
        mask.scatter_(dim=1, index=topk_tensor.indices, value=0)
        target_atk_tensor = (attacked_tensor * mask)[:, self.target_list]
        target_non_atk_tensor = (non_attacked_tensor * mask)[:, self.target_list]
        c = torch.min(attacked_tensor[:, self.target_list] - non_attacked_tensor[:, self.target_list], dim=0).values
        loss = (self.b2 * (target_atk_tensor - target_non_atk_tensor - c)).sigmoid().sum()
        return loss