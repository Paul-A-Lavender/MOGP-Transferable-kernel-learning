
import gpytorch
from gpytorch.kernels import  RBFKernel
class Linear_Model_Of_Corregionalization(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y,likelihood,kernel,config):
        super(Linear_Model_Of_Corregionalization, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=config.NUM_CONC
        )
        kernels=[]
        # RBFKernel do not need config to pass on global configurations
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
    def __init__(self, train_x, train_y,likelihood,kernel,config):
        super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=config.NUM_CONC
        )
        # RBFKernel do not need config to pass on global configurations
        if kernel==RBFKernel:
            kern=kernel()
        else:
            kern=kernel(config)
                
        self.covar_module =gpytorch.kernels.MultitaskKernel(
            kern, num_tasks=config.NUM_CONC, rank=config.NUM_CONC
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)