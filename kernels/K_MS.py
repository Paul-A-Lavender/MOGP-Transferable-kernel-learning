import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import train_test_split
from gpytorch.kernels import Kernel
import pickle
from sklearn.preprocessing import StandardScaler
from gpytorch.means import Mean
from sklearn.metrics import mean_squared_error
from gpytorch.kernels import Kernel, RBFKernel,ScaleKernel
from gpytorch.constraints import Interval
from gpytorch.settings import cholesky_jitter
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
import torch.nn as nn

class K_MS(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_coefficient=torch.nn.Parameter(torch.rand((nums_domain.long()))*2-1)
        self.K_ES=Kernel_Exponential_Squared()
        self.register_parameter(name='domain_coefficient', parameter=self.domain_coefficient)
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_domain   =   x1[:, :num_domain_feat]
        x2_domain   =   x2[:, :num_domain_feat]

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