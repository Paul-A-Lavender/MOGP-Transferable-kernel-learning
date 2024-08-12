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

class Kernel_Squared_Exponential(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height_scales=torch.nn.Parameter(torch.ones((nums_domain.long())))
        self.length_scales=torch.nn.Parameter(torch.ones((nums_domain.long()))) 
        self.register_parameter(name='height_scales', parameter=self.height_scales)
        self.register_parameter(name='length_scales', parameter=self.length_scales)
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # x1_domain   =   x1[:, :num_domain_feat]
        # x1_genetics =   x1[:, num_domain_feat:]
        # x2_domain   =   x2[:, :num_domain_feat]
        # x2_genetics =   x2[:, num_domain_feat:]

        x1_domain   =   x1[:, :num_domain_feat]
        x1_genetics =   x1
        x2_domain   =   x2[:, :num_domain_feat]
        x2_genetics =   x2

        dist_mat=self.covar_dist(x1_genetics, x2_genetics, square_dist=True, diag=diag, **params)

        height_scales_parsed_1 = self.height_scales[x1_domain.long()].flatten()
        height_scales_parsed_2 = self.height_scales[x2_domain.long()].flatten()
        length_scales_parsed_1 = self.length_scales[x1_domain.long()].flatten()
        length_scales_parsed_2 = self.length_scales[x2_domain.long()].flatten()


        # print(f"height_scales_parsed_1: {height_scales_parsed_1.shape}")
        # print(f"height_scales_parsed_2: {height_scales_parsed_2.shape}")
        # print(f"length_scales_parsed_1: {length_scales_parsed_1.shape}")
        # print(f"length_scales_parsed_2: {length_scales_parsed_2.shape}")
        part_L1L2T=torch.sqrt(torch.outer(length_scales_parsed_1*length_scales_parsed_1,length_scales_parsed_2.T*length_scales_parsed_2.T))
        part_L1L1T=length_scales_parsed_1*length_scales_parsed_1
        part_L2L2T=length_scales_parsed_2*length_scales_parsed_2
        part_L1sqrL2sqr=torch.outer(part_L1L1T,torch.ones_like(part_L2L2T).T)+torch.outer(torch.ones_like(part_L1L1T),part_L2L2T)

        # print(part_L1L2T.shape)
        # print(part_L1L1T.shape)
        # print(part_L2L2T.shape)
        # print(part_L1sqrL2sqr.shape)

        part_1=torch.outer(height_scales_parsed_1,height_scales_parsed_2.T)
        part_2=torch.sqrt(2*part_L1L2T/part_L1sqrL2sqr)
        part_3=torch.exp(-dist_mat/part_L1sqrL2sqr)

        result=part_1*part_2*part_3
        if diag:
            return result.diag()
        return result