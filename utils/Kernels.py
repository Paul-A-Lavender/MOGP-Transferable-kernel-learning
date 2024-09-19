import torch
from gpytorch.kernels import Kernel, RBFKernel
from utils.HelperFunctions import *
from utils.Models import *

class K_SE(Kernel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Initialize height and length scale parameters for each domain
        self.height_scales = torch.nn.Parameter(torch.ones((config.NUMS_DOMAIN)))
        self.length_scales = torch.nn.Parameter(torch.ones((config.NUMS_DOMAIN)))
        self.register_parameter(name='height_scales', parameter=self.height_scales)
        self.register_parameter(name='length_scales', parameter=self.length_scales)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Separate domain features and genetic features
        x1_domain = x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x1_genetics = x1
        x2_domain = x2[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_genetics = x2

        # Compute distance matrix based on genetic features
        dist_mat = self.covar_dist(x1_genetics, x2_genetics, square_dist=True, diag=diag, **params)

        # Extract and parse domain-specific height and length scales
        height_scales_parsed_1 = self.height_scales[x1_domain.long()].flatten()
        height_scales_parsed_2 = self.height_scales[x2_domain.long()].flatten()
        length_scales_parsed_1 = self.length_scales[x1_domain.long()].flatten()
        length_scales_parsed_2 = self.length_scales[x2_domain.long()].flatten()

        # Compute the major components of the kernel function
        part_L1L2T = torch.sqrt(torch.outer(length_scales_parsed_1 * length_scales_parsed_1, length_scales_parsed_2.T * length_scales_parsed_2.T))
        part_L1L1T = length_scales_parsed_1 * length_scales_parsed_1
        part_L2L2T = length_scales_parsed_2 * length_scales_parsed_2
        part_L1sqrL2sqr = torch.outer(part_L1L1T, torch.ones_like(part_L2L2T).T) + torch.outer(torch.ones_like(part_L1L1T), part_L2L2T)

        part_1 = torch.outer(height_scales_parsed_1, height_scales_parsed_2.T)
        part_2 = torch.sqrt(2 * part_L1L2T / part_L1sqrL2sqr)
        part_3 = torch.exp(-dist_mat / part_L1sqrL2sqr)

        # Combine components element-wise
        result = part_1 * part_2 * part_3

        if diag:
            return result.diag()
        return result


class K_MS(Kernel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Initialize domain coefficients
        self.domain_coefficient = torch.nn.Parameter(torch.rand((config.NUMS_DOMAIN)) * 2 - 1)
        self.K_ES = K_SE(config)
        self.register_parameter(name='domain_coefficient', parameter=self.domain_coefficient)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Extract domain features
        x1_domain = x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_domain = x2[:, :self.config.NUMS_DOMAIN_FEATURE]

        # Parse domain-specific coefficients and compute domain scaler
        domain_coefficient_parsed_1 = self.domain_coefficient[x1_domain.long()].flatten()
        domain_coefficient_parsed_2 = self.domain_coefficient[x2_domain.long()].flatten()
        domain_scaler = torch.outer(torch.tanh(domain_coefficient_parsed_1), torch.tanh(domain_coefficient_parsed_2.T))

        # Create a domain matrix to mask identical domains
        domain_mat = torch.outer(x1_domain.flatten() + 1, 1 / (x2_domain.flatten().T + 1))
        mask = (domain_mat == 1.0)

        # Set values equal to 1 to 0 in the domain matrix
        domain_mat[mask] = 0.0
        # Set other values to 1
        domain_mat[~mask] = 1.0

        # Calculate base covariance and adjust by domain scaler
        base_cov = self.K_ES(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        final_scaler = (domain_scaler - 1.0) * domain_mat + 1.0
        return base_cov * final_scaler


class K_Alpha_Beta(Kernel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Initialize a list of RBF kernels, one for each domain
        self.kernels = [RBFKernel() for _ in range(config.NUMS_DOMAIN)]
        self.config = config

        # Initialize alpha and beta parameters for each domain
        self.alpha = torch.nn.Parameter(torch.rand(config.NUMS_DOMAIN))
        self.beta = torch.nn.Parameter(torch.rand(config.NUMS_DOMAIN))
        self.register_parameter(name='alpha', parameter=self.alpha)
        self.register_parameter(name='beta', parameter=self.beta)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Extract domain features
        x1_domain = x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_domain = x2[:, :self.config.NUMS_DOMAIN_FEATURE]
        x1_num_data = x1_domain.shape[0]
        x2_num_data = x2_domain.shape[0]

        # Initialize covariance matrices for alpha and beta terms
        cov_mats_alpha = torch.zeros([x1_num_data, x2_num_data])
        cov_mats_beta = torch.zeros([x1_num_data, x2_num_data])

        # Reparameterize alpha and beta
        alpha_reparameterized = torch.sqrt(self.alpha * self.alpha)
        beta_reparameterized = torch.sqrt(self.beta * self.beta)

        # Compute covariance matrices for each domain using RBF kernel
        for i in range(self.config.NUMS_DOMAIN):
            cov_mats_alpha = cov_mats_alpha.add(self.kernels[i](x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params) * alpha_reparameterized[i])
            cov_mats_beta = cov_mats_beta.add(self.kernels[i](x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params) * beta_reparameterized[i])

        # Create domain matrices for alpha and beta
        domain_mat_alpha = torch.outer(x1_domain.flatten() + 1, 1 / (x2_domain.flatten().T + 1))
        domain_mat_beta = torch.outer(x1_domain.flatten() + 1, 1 / (x2_domain.flatten().T + 1))
        mask = (domain_mat_alpha == 1.0)

        # Mask identical domains and adjust domain matrices
        domain_mat_alpha[mask] = 0.0
        domain_mat_alpha[~mask] = 1.0
        domain_mat_beta[mask] = 1.0
        domain_mat_beta[~mask] = 0.0

        # Combine alpha and beta covariance matrices with domain masks
        result = cov_mats_alpha * domain_mat_alpha + cov_mats_beta * domain_mat_beta

        if diag:
            return result.diag()
        return result


class K_MS_with_Feat_Scaling(Kernel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Initialize domain coefficient
        self.domain_coefficient = torch.nn.Parameter(torch.rand(config.NUMS_DOMAIN) * 2 - 1)
        self.K_ES = K_SE(config)
        self.register_parameter(name='domain_coefficient', parameter=self.domain_coefficient)

        # Initialize feature relatedness parameter for scaling
        feature_relateness_init = torch.ones(config.NUM_FEAT)
        self.feature_relateness = torch.nn.Parameter(feature_relateness_init)
        self.register_parameter(name='feature_relateness', parameter=self.feature_relateness)
        self.config = config

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Extract domain and genetic features
        x1_domain = x1[:, :self.config.NUMS_DOMAIN_FEATURE]
        x2_domain = x2[:, :self.config.NUMS_DOMAIN_FEATURE]
        x1_genetic = x1[:, self.config.NUMS_DOMAIN_FEATURE:]
        x2_genetic = x2[:, self.config.NUMS_DOMAIN_FEATURE:]

        # Scale genetic features using feature relatedness
        feature_scaler_rep = (torch.tanh(self.feature_relateness) + 1.0)
        x1_scaled = x1_genetic * torch.outer(torch.ones(x1.shape[0]), feature_scaler_rep)
        x2_scaled = x2_genetic * torch.outer(torch.ones(x2.shape[0]), feature_scaler_rep)

        # Parse domain-specific coefficients and compute domain scaler
        domain_coefficient_parsed_1 = self.domain_coefficient[x1_domain.long()].flatten()
        domain_coefficient_parsed_2 = self.domain_coefficient[x2_domain.long()].flatten()
        domain_scaler = torch.outer(torch.tanh(domain_coefficient_parsed_1), torch.tanh(domain_coefficient_parsed_2.T))

        # Create a domain matrix to mask identical domains
        domain_mat = torch.outer(x1_domain.flatten() + 1, 1 / (x2_domain.flatten().T + 1))
        mask = (domain_mat == 1.0)

        # Set values equal to 1 to 0 in the domain matrix
        domain_mat[mask] = 0.0
        # Set other values to 1
        domain_mat[~mask] = 1.0

        # Concatenate domain and scaled genetic features
        x1_cat = torch.cat((x1_domain, x1_scaled), dim=1)
        x2_cat = torch.cat((x2_domain, x2_scaled), dim=1)

        # Calculate base covariance and adjust by domain scaler
        base_cov = self.K_ES(x1_cat, x2_cat, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        final_scaler = (domain_scaler - 1.0) * domain_mat + 1.0

        return base_cov * final_scaler
