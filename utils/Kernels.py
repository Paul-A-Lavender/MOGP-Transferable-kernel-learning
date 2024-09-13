import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import  DataLoader
import numpy as np
from gpytorch.kernels import Kernel
import pickle

from gpytorch.kernels import Kernel, RBFKernel

import torch.nn as nn
import logging
from itertools import product
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset,TensorDataset

from utils.HelperFunctions import *
from utils.Models import *

class Kernel_Exponential_Squared(Kernel):
    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        self.height_scales=torch.nn.Parameter(torch.ones((config.NUMS_DOMAIN)))
        self.length_scales=torch.nn.Parameter(torch.ones((config.NUMS_DOMAIN))) 
        self.register_parameter(name='height_scales', parameter=self.height_scales)
        self.register_parameter(name='length_scales', parameter=self.length_scales)
        self.config=config
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # x1_domain   =   x1[:, :num_domain_feat]
        # x1_genetics =   x1[:, num_domain_feat:]
        # x2_domain   =   x2[:, :num_domain_feat]
        # x2_genetics =   x2[:, num_domain_feat:]

        x1_domain   =   x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x1_genetics =   x1
        x2_domain   =   x2[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_genetics =   x2

        dist_mat=self.covar_dist(x1_genetics, x2_genetics, square_dist=True, diag=diag, **params)
        height_scales_parsed_1 = self.height_scales[x1_domain.long()].flatten()
        height_scales_parsed_2 = self.height_scales[x2_domain.long()].flatten()
        length_scales_parsed_1 = self.length_scales[x1_domain.long()].flatten()
        length_scales_parsed_2 = self.length_scales[x2_domain.long()].flatten()

        part_L1L2T=torch.sqrt(torch.outer(length_scales_parsed_1*length_scales_parsed_1,length_scales_parsed_2.T*length_scales_parsed_2.T))
        part_L1L1T=length_scales_parsed_1*length_scales_parsed_1
        part_L2L2T=length_scales_parsed_2*length_scales_parsed_2
        part_L1sqrL2sqr=torch.outer(part_L1L1T,torch.ones_like(part_L2L2T).T)+torch.outer(torch.ones_like(part_L1L1T),part_L2L2T)

        part_1=torch.outer(height_scales_parsed_1,height_scales_parsed_2.T)
        part_2=torch.sqrt(2*part_L1L2T/part_L1sqrL2sqr)
        part_3=torch.exp(-dist_mat/part_L1sqrL2sqr)

        result=part_1*part_2*part_3
        if diag:
            return result.diag()
        return result
    
  
class K_MS(Kernel):
    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        self.domain_coefficient=torch.nn.Parameter(torch.rand((config.NUMS_DOMAIN))*2-1)
        self.K_ES=Kernel_Exponential_Squared(config)
        self.register_parameter(name='domain_coefficient', parameter=self.domain_coefficient)
        self.config=config
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_domain   =   x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_domain   =   x2[:, :self.config.NUMS_DOMAIN_FEATURE]

        domain_coefficient_parsed_1=self.domain_coefficient[x1_domain.long()].flatten()
        domain_coefficient_parsed_2=self.domain_coefficient[x2_domain.long()].flatten()
        domain_scaler=torch.outer(torch.tanh(domain_coefficient_parsed_1),torch.tanh(domain_coefficient_parsed_2.T))

        domain_mat=torch.outer(x1_domain.flatten()+1,1/(x2_domain.flatten().T+1))
        mask = (domain_mat == 1.0)

        # 等于1的位置置0
        domain_mat[mask] = 0.0

        # 不等于1的位置置1
        domain_mat[~mask] = 1.0
        # print(domain_scaler)

        
        base_cov=self.K_ES(x1,x2,diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

        final_scaler=(domain_scaler-1.0)*domain_mat+1.0
        return base_cov*final_scaler

class K_Alpha_Beta(Kernel):
    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        self.kernels=[]
        for _ in range(config.NUMS_DOMAIN):
            self.kernels.append(RBFKernel())
        self.config=config
        self.alpha= torch.nn.Parameter(torch.rand(config.NUMS_DOMAIN))
        self.beta=torch.nn.Parameter(torch.rand(config.NUMS_DOMAIN))
        self.register_parameter(name='alpha', parameter=self.alpha)
        self.register_parameter(name='beta', parameter=self.beta)
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):#
        x1_domain   =   x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_domain   =   x2[:, :self.config.NUMS_DOMAIN_FEATURE]
        x1_num_data=x1_domain.shape[0]
        x2_num_data=x2_domain.shape[0]
        cov_mats_alpha=torch.zeros([x1_num_data,x2_num_data])
        cov_mats_beta=torch.zeros([x1_num_data,x2_num_data])

        alpha_reparameterized=torch.sqrt(self.alpha*self.alpha)
        beta_reparameterized=torch.sqrt(self.beta*self.beta)
        for i in range(self.config.NUMS_DOMAIN):
            cov_mats_alpha=cov_mats_alpha.add(self.kernels[i](x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)*alpha_reparameterized[i])
            cov_mats_beta=cov_mats_beta.add(self.kernels[i](x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)*beta_reparameterized[i])

        domain_mat_alpha=torch.outer(x1_domain.flatten()+1,1/(x2_domain.flatten().T+1))
        domain_mat_beta=torch.outer(x1_domain.flatten()+1,1/(x2_domain.flatten().T+1))
        mask = (domain_mat_alpha == 1.0)
        domain_mat_alpha[mask] = 0.0
        domain_mat_alpha[~mask] = 1.0
        domain_mat_beta[mask] = 1.0
        domain_mat_beta[~mask] = 0.0

        
        result=cov_mats_alpha*domain_mat_alpha+cov_mats_beta*domain_mat_beta
        if diag:
            return result.diag()
        return result
    
  
class K_MS_with_Feat_Scaling(Kernel):
    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        
        self.domain_coefficient=torch.nn.Parameter(torch.rand(config.NUMS_DOMAIN)*2-1)
        self.K_ES=Kernel_Exponential_Squared(config)
        self.register_parameter(name='domain_coefficient', parameter=self.domain_coefficient)
        
        
        feature_relateness_init=torch.ones(config.NUM_FEAT)
        self.feature_relateness=torch.nn.Parameter(feature_relateness_init)
        self.register_parameter(name='feature_relateness', parameter=self.feature_relateness)
        self.config=config
        
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_domain   =   x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_domain   =   x2[:, :self.config.NUMS_DOMAIN_FEATURE]
        x1_genetic   =   x1[:, self.config.NUMS_DOMAIN_FEATURE:]
        x2_genetic   =   x2[:, self.config.NUMS_DOMAIN_FEATURE:]        
        
        feature_scaler_rep=(torch.tanh(self.feature_relateness)+1.0)
        x1_scaled=x1_genetic*torch.outer(torch.ones(x1.shape[0]),feature_scaler_rep)
        x2_scaled=x2_genetic*torch.outer(torch.ones(x2.shape[0]),feature_scaler_rep)
        domain_coefficient_parsed_1=self.domain_coefficient[x1_domain.long()].flatten()
        domain_coefficient_parsed_2=self.domain_coefficient[x2_domain.long()].flatten()
        domain_scaler=torch.outer(torch.tanh(domain_coefficient_parsed_1),torch.tanh(domain_coefficient_parsed_2.T))
        domain_mat=torch.outer(x1_domain.flatten()+1,1/(x2_domain.flatten().T+1))
        mask = (domain_mat == 1.0)

        # 等于1的位置置0
        domain_mat[mask] = 0.0

        # 不等于1的位置置1
        domain_mat[~mask] = 1.0
        # print(domain_scaler)
        x1_cat=torch.cat((x1_domain, x1_scaled), dim=1)
        x2_cat=torch.cat((x2_domain, x2_scaled), dim=1)
        base_cov=self.K_ES(x1_cat,x2_cat,diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

        final_scaler=(domain_scaler-1.0)*domain_mat+1.0
        return base_cov*final_scaler
