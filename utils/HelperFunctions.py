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
import torch.nn as nn
import logging
from itertools import product
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset,TensorDataset

def load_dataset(X_path,y_path,X_domain_path=None,do_standardisation=False,test_size=0.1,random_state=42):
    X_df=None
    X_domain_info=None
    y_df=None
    with open(X_path, 'rb') as f:
        X_df = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_df = pickle.load(f)
    print(X_df.shape)
    print(y_df.shape)
    if X_domain_path!=None:
        with open(X_domain_path, 'rb') as f:
            X_domain_info = pickle.load(f)
        print(X_domain_info.shape)
        X_train, X_test, y_train, y_test,X_D_train,X_D_test = train_test_split(X_df, y_df,X_domain_info, test_size=test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test= train_test_split(X_df, y_df, test_size=test_size, random_state=random_state)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    if X_domain_path!=None:
        X_D_train = np.array(X_D_train, dtype=np.float32)
        X_D_test = np.array(X_D_test, dtype=np.float32)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    if do_standardisation:
        print("Performing standardisation")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    if X_domain_path!=None:
        X_D_train_tensor = torch.tensor(X_D_train)
        X_D_test_tensor = torch.tensor(X_D_test)

    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    if X_domain_path!=None: 
        return (X_train_tensor,X_D_train_tensor,y_train_tensor),(X_test_tensor,X_D_test_tensor,y_test_tensor)
    else:
        return (X_train_tensor,y_train_tensor),(X_test_tensor,y_test_tensor)

def sigmoid_4_param(x, x0, L, k, d):
    """ Comparing with Dennis Wang's sigmoid:
    x0 -  p - position, correlation with IC50 or EC50
        bounds [0, 1]
    L = 1 in Dennis Wang's sigmoid, protect from devision by zero if x is too small 
        L<1 inverted sigmoid, l=100 - lower upper and lower boundso sigmpoid on y axis (y= [0.1, 0.11])
        bounds [0.8, 10]
    k = -1/s (s -shape parameter)  default = -10 k=0 straight line, k<0 sigmoid around k=-10
        bounds [1, -100]
    d - determines the vertical position of the sigmoid - shift on y axis - better fitting then Dennis Wang's sigmoid
         bounds [0, 0.9]
    parameters_bound ((0, 0.8, -100, 0), (1, 10, 1, 0.9))
    """
    return ( 1/ (L + np.exp(-k*(x-x0))) + d)

def overwrite_to_test():
    VAR=0.09
    AMP=1.0
    train_size=100

    X_sin=torch.linspace(0, 1, train_size)
    X_cos=torch.linspace(0, 1, train_size)
    X_sig=torch.linspace(0, 1, train_size)
    y_sin = AMP*torch.sin(X_sin * (2 * math.pi)) + torch.randn(X_sin.size()) * math.sqrt(VAR)
    y_cos = AMP*torch.cos(X_cos * (2 * math.pi)) + torch.randn(X_cos.size()) * math.sqrt(VAR)
    y_sig = AMP*(torch.sigmoid(15*(X_sig-0.5))-0.5)*2+torch.randn(X_cos.size()) * math.sqrt(VAR)
    # Training data is 100 points in [0,1] inclusive regularly spaced
    X_train_tensor = torch.linspace(0, 1, train_size)
    # True function is sin(2*pi*x) with Gaussian noise

    num_conc=1
    num_feat=1
    nums_domain=torch.Tensor([3])
    num_data=train_size

    X_sin = X_sin.unsqueeze(1)
    X_cos = X_cos.unsqueeze(1)
    X_sig = X_sig.unsqueeze(1)
    y_sin = y_sin.unsqueeze(1)
    y_cos = y_cos.unsqueeze(1)
    y_sig = y_sig.unsqueeze(1)

    X_sin_domain=torch.zeros(train_size,1)
    X_cos_domain=torch.ones(train_size,1)
    X_sig_domain=torch.ones(train_size,1)*2
    X_sin_cat = torch.cat((X_sin_domain, X_sin), dim=1)
    X_cos_cat = torch.cat((X_cos_domain, X_cos), dim=1)
    X_sig_cat = torch.cat((X_sig_domain, X_sig), dim=1)
    X_train_tensor = torch.cat((X_sin_cat, X_cos_cat,X_sig_cat), dim=0)
    y_train_tensor = torch.cat((y_sin, y_cos,y_sig), dim=0)

    f, ax = plt.subplots(1, 3, figsize=(12, 3))
    for i in range(3):
        ax[i].plot(X_train_tensor[i*train_size:(i+1)*train_size,1:].squeeze().numpy(),y_train_tensor[i*train_size:(i+1)*train_size], 'k*')
        axis=X_train_tensor[i*train_size:(i+1)*train_size,1:].flatten().numpy()
        ax[i].set_ylim([-3, 3])
        ax[i].legend(['Observed Data', 'Mean', 'Confidence'])

    # indices = torch.randperm(X_train_tensor.size(0))
    # X_train_tensor = X_train_tensor[indices]
    # y_train_tensor = y_train_tensor[indices]

    test_size=50
    X_sin=torch.linspace(0, 1, test_size)
    X_cos=torch.linspace(0, 1, test_size)
    X_sig=torch.linspace(0, 1, test_size)

    X_sin_domain=torch.zeros(test_size,1)
    X_cos_domain=torch.ones(test_size,1)
    X_sig_domain=torch.ones(test_size,1)*2

    X_sin = X_sin.unsqueeze(1)
    X_cos = X_cos.unsqueeze(1)
    X_sig = X_sig.unsqueeze(1)

    y_sin =AMP* torch.sin(X_sin * (2 * math.pi)) + torch.randn(X_sin.size()) * math.sqrt(VAR)
    y_cos =AMP* torch.cos(X_cos * (2 * math.pi)) + torch.randn(X_cos.size()) * math.sqrt(VAR)
    y_sig =AMP* (torch.sigmoid(15*(X_sig-0.5))-0.5)*2+torch.randn(X_cos.size()) * math.sqrt(VAR)

    X_sin_cat = torch.cat((X_sin_domain, X_sin), dim=1)
    X_cos_cat = torch.cat((X_cos_domain, X_cos), dim=1)
    X_sig_cat = torch.cat((X_sig_domain, X_sig), dim=1)

    # X_test_tensor =X_sin_cat
    # y_test_tensor =y_sin
    
    X_test_tensor = torch.cat((X_sin_cat, X_cos_cat,X_sig_cat), dim=0)
    y_test_tensor = torch.cat((y_sin, y_cos,y_sig), dim=0)

    # X_test_tensor = torch.cat((X_sin_cat, X_cos_cat), dim=0)
    # y_test_tensor = torch.cat((y_sin, y_cos), dim=0)  

def dataloader2tensor(data_loader):
    all_inputs = []
    all_labels = []

    # 遍历 DataLoader 并将所有批次数据合并成一个大张量
    for inputs, labels in data_loader:
        all_inputs.append(inputs)
        all_labels.append(labels)

    # 使用 torch.cat 将所有小批量张量合并成一个大张量
    X_tensor = torch.cat(all_inputs, dim=0)
    y_tensor = torch.cat(all_labels, dim=0)

    # 现在，X_train_tensor 和 y_train_tensor 是重新合并后的大张量
    print(f"X_train_tensor shape: {X_tensor.shape}")
    print(f"y_train_tensor shape: {y_tensor.shape}")
    return X_tensor,y_tensor

class config():
    NUMS_DOMAIN=None
    NUMS_DOMAIN_FEATURE=None
    NUMS_DOMAIN_AS_INT=None
    NUM_CONC=None
    STEP_SIZE=None
    lr=None
    gamma=None
    def __init__(self, NUMS_DOMAIN=None,
                 NUMS_DOMAIN_FEATURE=None,
                 NUMS_DOMAIN_AS_INT=None,
                 NUM_CONC=None,
                 STEP_SIZE=None,
                 lr=None,
                 gamma=None,
                 NUM_FEAT=None,):
        self.NUMS_DOMAIN=NUMS_DOMAIN
        self.NUMS_DOMAIN_FEATURE=NUMS_DOMAIN_FEATURE
        self.NUMS_DOMAIN_AS_INT=NUMS_DOMAIN_AS_INT
        self.NUM_CONC=NUM_CONC
        self.STEP_SIZE=STEP_SIZE
        self.lr=lr
        self.gamma=gamma
        self.NUM_FEAT=NUM_FEAT

def shadowLogger(logger,level,msg):
    if logger==None:
        return
    if level=="INFO":
        logger.info(msg)
    elif level=="DEBUG":
        logger.debug(msg)


def run_test(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,model,kernel,config,logger=None):
    shadowLogger(logger,"INFO",f'training starts for model {str(model)} with kernel {str(kernel)}; lr: {config.lr}; step_size:{config.STEP_SIZE}; gamma:{config.gamma}')
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=config.NUM_CONC)

    m = model(X_train_tensor, y_train_tensor, likelihood,kernel,config)

    training_iterations = 500

    # Find optimal model hyperparameters
    m.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(m.parameters(), lr=config.lr)
    STEP_SIZE=config.STEP_SIZE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=config.gamma)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, m)
    last_loss=1
    avg_loss=0
    this_loss=0
    for i in range(training_iterations):
        # try:
            optimizer.zero_grad()
            output = m(X_train_tensor)
            loss = -mll(output, y_train_tensor)
            loss.backward()
                
            this_loss=loss.item()
            avg_loss+=this_loss
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
            
            if i%STEP_SIZE==STEP_SIZE-1:
                shadowLogger(logger,"DEBUG"'Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                avg_loss=avg_loss/STEP_SIZE
                if abs((this_loss-avg_loss)/avg_loss)<0.01 or ((this_loss-avg_loss)/avg_loss>0.30 and i>250):
                    shadowLogger(logger,"INFO",f'Early cut off at epoch {i} with loss of {this_loss }')
                    break
                        
                avg_loss=0
                    
        # except Exception as e:
        #     logger.error(f"发生了一个异常: {e}")
        #     continue
        
    m.eval()
    likelihood.eval()
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        distribution = likelihood(m(X_test_tensor))
        mean = distribution.mean
        lower, upper = distribution.confidence_region()
    nll = -torch.distributions.Normal(mean, distribution.variance.sqrt()).log_prob(y_test_tensor).mean().item()

    rmse=torch.sqrt(torch.tensor(mean_squared_error(y_test_tensor.numpy(), mean.numpy()))).item() 
    nmse = rmse / torch.var(y_test_tensor).item()

    shadowLogger(logger,"INFO",f'NLL: {nll:.4f}; RMSE: {rmse:.4f}; NMSE: {nmse:.4f}')
    return m,nll

def get_dataset_path(name):
    paths={
        "TOY":{'X':"data/X_df_toy.pkl",'Y':"data/y_df_toy.pkl",'D':None},
        "FULL_SHOTS":{'X':"data/full_shots_Shikonin/X_df.pkl",'Y':"data/full_shots_Shikonin/y_df.pkl",'D':"data/full_shots_Shikonin/X_domain_info.pkl"},
        "2_SHOTS":{'X':"data/2_shots/X_df.pkl",'Y':"data/2_shots/y_df.pkl",'D':"data/2_shots/X_domain_info.pkl"},
        "8_SHOTS":{'X':"data/8_shots/X_df.pkl",'Y':"data/8_shots/y_df.pkl",'D':"data/8_shots/X_domain_info.pkl"}   
    }
    
    try:
        return paths[name]['X'],paths[name]['Y'],paths[name]['D']
    except:
        print(f"Invalid Dataset Name!({name})")
        return

def grid_search(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,param_grid,Global): 
    param_combinations = list(product(*param_grid.values()))
    for each_param in param_combinations:
        Global.lr=each_param[2]
        Global.gamma=each_param[3]
        Global.STEP_SIZE=each_param[4]
        run_test(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,model=each_param[0],kernel=each_param[1],config=Global)