import numpy as np
import torch

from torch import nn, distributions
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment



def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld



def losses( mu_z, logvar_z):
    std_z = torch.exp(logvar_z / 2)    

    KLDs = distributions.kl_divergence(distributions.Normal(mu_z,std_z),
                                    distributions.Normal(torch.zeros_like(mu_z),torch.ones_like(std_z)))
    KLD_total_mean = KLDs.sum(-1).mean()

    return KLD_total_mean
EPS  =1e-12
def _kl_discrete_loss( alpha):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])

    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss

log_var = torch.rand((4,6))
mu = torch.rand((4,6))

kld1 = losses(mu,log_var)
kld2 = kl_divergence(mu,log_var)

#print(kld1)
#print(kld2)

alpha = torch.rand(3,4)
print(alpha)
#print(alpha.sum(-1,keepdim=True))
#alpha /= alpha.sum(-1,keepdim=True)
#alpha = F.softmax(alpha)
unifom_params = torch.ones_like(alpha)/alpha.shape[1]

kld = distributions.kl_divergence(distributions.Categorical(alpha),distributions.Categorical(unifom_params))
print(alpha)
print(kld.mean())
kld  = _kl_discrete_loss(alpha)
print(kld)
print(F.cross_entropy(alpha,unifom_params))

#print(unifom_params * alpha)