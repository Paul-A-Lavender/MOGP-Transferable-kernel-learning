import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import  DataLoader
import numpy as np
from gpytorch.kernels import Kernel
import pickle

from gpytorch.kernels import Kernel, RBFKernel

from itertools import product
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset,TensorDataset




class Linear_Model_Of_Corregionalization(gpytorch.models.ExactGP):
    #TODO add input for domain information
    def __init__(self, train_x, train_y,likelihood,kernel,config):
        super(Linear_Model_Of_Corregionalization, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=config.NUM_CONC
        )
        kernels=[]
        if kernel==RBFKernel:
            for _ in range(config.NUM_CONC):
                kernels.append(kernel())
        else:
            for _ in range(config.NUM_CONC):
                kernels.append(kernel(config))
        self.covar_module =gpytorch.kernels.LCMKernel(
            kernels, num_tasks=config.NUM_CONC, rank=config.NUM_CONC
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MultitaskGP(gpytorch.models.ExactGP):
    #TODO add input for domain information
    def __init__(self, train_x, train_y,likelihood,kernel,config):
        super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=config.NUM_CONC
        )
        kernels=[]
        
        if kernel==RBFKernel:
            kern=kernel()
        else:
            kern=kernel(config)
                
        self.covar_module =gpytorch.kernels.MultitaskKernel(
            kern, num_tasks=config.NUM_CONC, rank=config.NUM_CONC
        )
        #     self.covar_module = gpytorch.kernels.MultitaskKernel(
        #         K_MS(),  # 2 latent GP's

        #     num_tasks=num_conc, rank=num_conc-1
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)