import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.functions import STHeaviside

eps = 1e-8


class Normal(nn.Module):
    """Samples from a Normal distribution using the reparameterization trick.
    """

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        self.normalization = torch.Tensor([np.log(2 * np.pi)]).requires_grad_()

        self.mu = torch.Tensor([mu]).requires_grad_()
        self.logsigma = torch.Tensor([math.log(sigma)]).requires_grad_()

    def _check_inputs(self, size, mu_logsigma):
        if size is None and mu_logsigma is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0).expand(size)
            logsigma = mu_logsigma.select(-1, 1).expand(size)
            return mu, logsigma
        elif size is not None:
            mu = self.mu.expand(size)
            logsigma = self.logsigma.expand(size)
            return mu, logsigma
        elif mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0)
            logsigma = mu_logsigma.select(-1, 1)
            return mu, logsigma
        else:
            raise ValueError(
                'Given invalid inputs: size={}, mu_logsigma={})'.format(
                    size, mu_logsigma))

    def sample(self, size=None, params=None):
        mu, logsigma = self._check_inputs(size, params)
        std_z = torch.randn_like(mu)
        sample = std_z * torch.exp(logsigma) + mu
        return sample

    def log_density(self, sample, params=None):
        if params is not None:
            mu, logsigma = self._check_inputs(None, params)
        else:
            mu, logsigma = self._check_inputs(sample.size(), None)
            mu = mu.type_as(sample)
            logsigma = logsigma.type_as(sample)

        c = self.normalization.type_as(sample.data)
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logsigma + c)

    def NLL(self, params, sample_params=None):
        """Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.
        """
        mu, logsigma = self._check_inputs(None, params)
        if sample_params is not None:
            sample_mu, sample_logsigma = self._check_inputs(None, sample_params)
        else:
            sample_mu, sample_logsigma = mu, logsigma

        c = self.normalization.type_as(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
            + torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)

    def kld(self, params):
        """Computes KL(q||p) where q is the given distribution and p
        is the standard Normal distribution.
        """
        mu, logsigma = self._check_inputs(None, params)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
        kld.mul_(-0.5)
        return kld

    def get_params(self):
        return torch.cat([self.mu, self.logsigma])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(
            self.mu.data[0], self.logsigma.exp().data[0])
        return tmpstr


class Laplace(nn.Module):
    """Samples from a Laplace distribution using the reparameterization trick.
    """

    def __init__(self, mu=0, scale=1):
        super(Laplace, self).__init__()
        self.normalization = torch.Tensor([-math.log(2)]).requires_grad_()

        self.mu = torch.Tensor([mu]).requires_grad_()
        self.logscale = torch.Tensor([math.log(scale)]).requires_grad_()

    def _check_inputs(self, size, mu_logscale):
        if size is None and mu_logscale is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and mu_logscale is not None:
            mu = mu_logscale.select(-1, 0).expand(size)
            logscale = mu_logscale.select(-1, 1).expand(size)
            return mu, logscale
        elif size is not None:
            mu = self.mu.expand(size)
            logscale = self.logscale.expand(size)
            return mu, logscale
        elif mu_logscale is not None:
            mu = mu_logscale.select(-1, 0)
            logscale = mu_logscale.select(-1, 1)
            return mu, logscale
        else:
            raise ValueError(
                'Given invalid inputs: size={}, mu_logscale={})'.format(
                    size, mu_logscale))

    def sample(self, size=None, params=None):
        mu, logscale = self._check_inputs(size, params)
        scale = torch.exp(logscale)
        # Unif(-0.5, 0.5)
        u = torch.rand_like(mu) - 0.5
        sample = mu - scale * torch.sign(u) * torch.log(1 - 2 * torch.abs(u) + eps)
        return sample

    def log_density(self, sample, params=None):
        if params is not None:
            mu, logscale = self._check_inputs(None, params)
        else:
            mu, logscale = self._check_inputs(sample.size(), None)
            mu = mu.type_as(sample)
            logscale = logscale.type_as(sample)

        c = self.normalization.type_as(sample.data)
        inv_scale = torch.exp(-logscale)
        ins_exp = - torch.abs(sample - mu) * inv_scale
        return ins_exp + c - logscale

    def get_params(self):
        return torch.cat([self.mu, self.logscale])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(
            self.mu.data[0], self.logscale.exp().data[0])
        return tmpstr

class Gumbel_Softmax(nn.Module):
    """
    CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX
    Jang et al, 2017
    """
    @staticmethod
    def sample( alpha, temperature, training =True):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        EPS = 1e-12
        if training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size()).cuda()
           

            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
          
            one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples
    @staticmethod
    def log_density(pis, samples, temperature):
        """
        Parameters:
            alphas(pis): list of scalars of type float torch.tensors
                the list should sum up to one. In the original paper (Jang et al, 2017)
                they are refered to as pi.
        """
        k = pis.shape[-1]
        consts = math.log(math.gamma(k)* temperature**(k-1))
        logsum_probs_sampels = -k * torch.log(pis.div(samples.pow(temperature)).sum(-1))
        logmulti_probs_sampels = pis.div(samples.pow(temperature + 1)).log().sum(-1)
        return consts + logsum_probs_sampels + logmulti_probs_sampels
        
class Bernoulli(nn.Module):
    """Samples from a Bernoulli distribution where the probability is given
    by the sigmoid of the given parameter.
    """

    def __init__(self, p=0.5, stgradient=False):
        super(Bernoulli, self).__init__()
        p = torch.Tensor([p])
        self.p = torch.log(p / (1 - p) + eps).requires_grad_()
        self.stgradient = stgradient

    def _check_inputs(self, size, ps):
        if size is None and ps is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and ps is not None:
            if ps.ndimension() > len(size):
                return ps.squeeze(-1).expand(size)
            else:
                return ps.expand(size)
        elif size is not None:
            return self.p.expand(size)
        elif ps is not None:
            return ps
        else:
            raise ValueError(
                'Given invalid inputs: size={}, ps={})'.format(size, ps))

    def _sample_logistic(self, size):
        u = torch.rand(size).requires_grad_()
        l = torch.log(u + eps) - torch.log(1 - u + eps)
        return l

    def sample(self, size=None, params=None):
        presigm_ps = self._check_inputs(size, params)
        logp = F.logsigmoid(presigm_ps)
        logq = F.logsigmoid(-presigm_ps)
        l = self._sample_logistic(logp.size()).type_as(presigm_ps)
        z = logp - logq + l
        b = STHeaviside.apply(z)
        return b if self.stgradient else b.detach()

    def log_density(self, sample, params=None):
        presigm_ps = self._check_inputs(sample.size(), params).type_as(sample)
        p = (F.sigmoid(presigm_ps) + eps) * (1 - 2 * eps)
        logp = sample * torch.log(p + eps) + (1 - sample) * torch.log(1 - p + eps)
        return logp

    def get_params(self):
        return self.p

    @property
    def nparams(self):
        return 1

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return self.stgradient

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f})'.format(
            torch.sigmoid(self.p.data)[0])
        return tmpstr