import numpy as np
import torch
from math import exp, pi, log
from torch import nn, distributions
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable
import torch.nn.init as init
from models.decoders import Discriminator
from models.layers import Lambda

import lib.dist as dist
import lib.ops as ops

class Beta_VAE(nn.Module):
    def __init__ (self, z_dim, nc = 1, hyper_params =[1,1,1], capacity = 0, output_type = 'binary',
                    with_clusters = True, n_cls = 0, weight_init = '', computes_std = True, x_num_param = 1):
        super(Beta_VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.computes_std = computes_std
        # self.prior_dist, self.q_dist = dist.Normal(), dist.Normal()
        # self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))
        self.x_num_param = 1 
        self.init_architecture()
        self.reparam = Lambda(self.reparameterize)
        self.alpha = hyper_params[0]
        self.beta = hyper_params[1]
        self.gamma = hyper_params[2]
        self.capacity = capacity
        self.output_type = output_type
        self.with_clusters = with_clusters
        if with_clusters:
            self.pi = nn.Parameter(torch.FloatTensor(n_cls, ).fill_(1) / n_cls, requires_grad=True)
            self.mu_c = nn.Parameter(torch.FloatTensor(n_cls, z_dim).fill_(0), requires_grad=True)
            self.logvar_c = nn.Parameter(torch.FloatTensor(n_cls, z_dim).fill_(0), requires_grad=True)
            self.n_cls = n_cls

        if weight_init =='xavier':
            self.apply(xavier_init)
        else:
            self.apply(kaiming_init)

    def init_architecture(self): 
        pass

    
    def forward(self, x):
        params = self.encoder(x).view(x.size(0),self.z_dim,2)
        mu, logstd_var = params.select(-1,0), params.select(-1,1)
        z = self.reparam(mu, logstd_var)
        
        x_hat = self.decoder(z)
        return x_hat, mu, logstd_var, z, params

    def reparameterize(self, arguments):
        mu, logstd_var= arguments
        if self.computes_std:
            std_var = torch.exp( logstd_var)
        else:
            std_var = torch.exp( 0.5 * logstd_var)

        #std = torch.exp(logvar)
        #eps = Variable(torch.randn(mu.size()).type_as(mu.data))
        eps = torch.randn_like(mu)
        #z = eps * std + mu
        return eps.mul(std_var).add_(mu)

    def generate(self):
        "Generates a random sample"
        z = torch.randn((self.z_dim))
        x = self.decoder(z.cuda())
        if self.output_type == 'binary':
            x = torch.sigmoid(x)
        return x


  
    def vae_loss(self,x, x_logit, mean, logstd_var, z):

        """VAE loss where it is calculated with a monte-carlo estimation of negative ELBO
        """

        if not self.computes_std:
            std_z = torch.exp(logstd_var / 2)
        else:
            std_z = torch.exp(logstd_var)   
        # logpx_z = -F.binary_cross_entropy_with_logits(x_logit, x, reduction='mean') * self.width*self.height
        # logpz = self.log_normal_pdf(z, torch.tensor([0.]).cuda(), torch.tensor([0.]).cuda())
        # logqz_x = self.log_normal_pdf(z, mean, logvar)
        # return -torch.mean(logpx_z + logpz - logqz_x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = self.prior_params.expand(expanded_size)
        prior_params.requires_grad_()
        if self.output_type == 'binary':
            logpx_z = -F.binary_cross_entropy_with_logits(x_hat, x, reduction='mean') * self.width*self.height
        else:
            logpx_z = -F.mse_loss(x_hat,x, reduction='mean') * self.width*self.height

        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        return -torch.mean(logpx_z + logpz - logqz_x)

    def losses(self, x, x_hat, mu_z, logstd_var, z, objective='B',TC = 0.0, equality= False):
        """
            Inputs:
                x: float Tensor of shape (self.height, self.width)
                    input to the network
                x_hat: float Tensor of shape (self.height, self.width)
                    output of the network
                mu_z: float Tensor of shape (self.z_dim)
                    output of encoder, mean of distribtion q
                logstd_var: flaot Tensor of shape (self.z_dim)
                    output of encoder, log standard deviation/varaince of distribtion q
                z: float Tensor of shape (self.z_dim)
                    output of reparameterization, sample from q(z|x)
                objective: char
                    B: stands for objective Burgess et al. (2017) Understand disentanglement in B-VAEs
                    H: is beta-VAE objective Higgins et al. (2016);
        """
        if not self.computes_std:
            std_z = torch.exp(logstd_var / 2)
        else:
            std_z = torch.exp(logstd_var)   
 
        #compute the reconstruction error
        recons = 0
        if self.output_type == 'binary':
            recons = F.binary_cross_entropy_with_logits(x_hat, x, reduction='mean') * self.width*self.height
        else:
            recons = F.mse_loss(x_hat * 255, x *255, reduction='sum') / 255
        KLDs = distributions.kl_divergence(distributions.Normal(mu_z,std_z),
                                        distributions.Normal(torch.zeros_like(mu_z),torch.ones_like(std_z)))
        KLD_total_mean = KLDs.sum(-1).mean()
        if equality:
            kl_divs = ops.equal_kl_loss(KLDs.mean(0))
        else:
            kl_divs = 0

        if objective== 'B':
            loss = recons + self.gamma * torch.abs(KLD_total_mean - self.capacity) + self.alpha *TC + self.beta * kl_divs
        else:
            loss = recons + self.beta * KLD_total_mean + self.gamma*TC + self.alpha * kl_divs

        return loss, recons, KLD_total_mean, KLDs

    def beta_tc_loss(self, x, x_hat, z_params, z, dataset_size):
        """
            Inputs:
                x: float Tensor of shape (self.height, self.width)
                    input to the network
                x_hat: float Tensor of shape (self.height, self.width)
                    output of the network
                mu_z: float Tensor of shape (self.z_dim)
                    output of encoder, mean of distribtion q
                logvar_z: flaot Tensor of shape (self.z_dim)
                    output of encoder, log varaince of distribtion q
                z: float Tensor of shape (self.z_dim)
                    output of reparameterization, sample from q(z|x)
            Output:
                float Tensors of scalars
                monte carlo estimate of EBLO decoposition 
                according to, Isolating Sources 
                of Disentanglement in VAEs (Chen et all, 2018)
        """
        if not self.computes_std:
            z_params[:,:,1] = z_params[:,:,1] / 2
    
        prior_dist, q_dist = dist.Normal(), dist.Normal()
        prior_params= torch.zeros(self.z_dim, 2)
        batch_size = x.size(0)
        x = x.view(batch_size, self.nc, self.height, self.width)
        expanded_size = (batch_size,) + prior_params.size()
        prior_params = prior_params.expand(expanded_size).cuda().requires_grad_()

        if self.output_type == 'binary':
            recons = F.binary_cross_entropy_with_logits(x_hat, x, reduction='mean') * self.width*self.height
        else:
            recons = F.mse_loss(x_hat * 255, x *255, reduction='sum') / 255
        logpz = prior_dist.log_density(z, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = q_dist.log_density(z, params=z_params).view(batch_size, -1).sum(1)

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = q_dist.log_density(
            z.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, 2)
        )

        logqz_prodmarginals = (torch.logsumexp(_logqz, dim=1, keepdim=False) - log(batch_size * dataset_size)).sum(1)
        logqz = (torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - log(batch_size * dataset_size))

        #monte carlo estiamtion

        #mutual information
        mi = (logqz_condx - logqz).mean()
        #total coorelation
        tc = (logqz - logqz_prodmarginals).mean()
        #dimension-wise KL. Here name regularization
        reg = (logqz_prodmarginals - logpz).mean()
        modified_elbo = recons + \
                        self.alpha *(mi) + \
                        self.beta * tc + \
                        self.gamma *reg

        return modified_elbo, recons, mi, tc, torch.abs(reg)


    def kld_unit_guassians_per_sample(self,mu,logstd_var):
        if not self.computes_std:
            std = torch.exp(logstd_var / 2)
        else:
            std = torch.exp(logstd_var)
        KLD = distributions.kl_divergence(distributions.Normal(mu,std),
            distributions.Normal(torch.zeros_like(mu),torch.ones_like(std)))
        return KLD
    
    #For clustering:

    def initialize_gmm_params(self, gmm):
        pi = torch.from_numpy(gmm.weights_).cuda().float()
        self.pi.data = torch.log(pi / (1 - pi))
        self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
        self.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
    
    def pc_given_z(self, z):
        if not self.computes_std:
            std_c = torch.exp(self.logvar_c / 2)
        else:
            std_c = torch.exp(self.logvar_c)
        pi = distributions.Categorical(torch.sigmoid(self.pi)).probs
        log_pz_given_c = distributions.Normal(self.mu_c, std_c[None, :, :]).log_prob(z[:, None, :]).sum(dim=2)
        log_pz_and_c = torch.log(pi)[None, :] + log_pz_given_c
        pc_given_z = torch.exp(log_pz_and_c - torch.logsumexp(log_pz_and_c, dim=1)[:, None])
        return pc_given_z

    def predict(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        pc_given_z_np = self.pc_given_z(z).detach().cpu().numpy()
        return np.argmax(pc_given_z_np, axis=1)
    def generate_per_cluster(self, cluster):
        if cluster >=0 and cluster < self.n_cls:
            mu = self.mu_c[cluster,:]
            log_var = self.logvar_c[cluster,:]
            z = self.reparam(mu,log_var).view(1,self.z_dim)
            x = self.decoder(z)
            if self.output_type == 'binary':
                x = torch.sigmoid(x)

            return x
        else:
            print("An out of scope or invalid number has been entered")
    def losses_clustering(self, x, x_hat, mu_z, logvar_z, z):
        if not self.computes_std:
            std_z = torch.exp(logvar_z / 2)
            std_c = torch.exp(self.logvar_c / 2)
        else:
            std_z = torch.exp(logvar_z)
            std_c = torch.exp(self.logvar_c)
        pi = distributions.Categorical(torch.sigmoid(self.pi)).probs
        pc_given_z = self.pc_given_z(z)

        BCE = F.binary_cross_entropy_with_logits(x_hat, x, reduction='mean') * self.width*self.height
        KLD = torch.sum(pc_given_z * distributions.kl_divergence(
            distributions.Independent(distributions.Normal(mu_z[:, None, :], std_z[:, None, :]), reinterpreted_batch_ndims=1),
            distributions.Independent(distributions.Normal(self.mu_c[None, :, :], std_c[None, :, :]), reinterpreted_batch_ndims=1)
        ), dim=1).mean()
        KLD_c = distributions.kl_divergence(distributions.Categorical(pc_given_z), distributions.Categorical(pi[None, :])).mean()

        return BCE, KLD, KLD_c, torch.tensor(0).float(), pc_given_z
    def kld_unit_guassians_per_cluster(self):
        if not self.computes_std:
            std_c = torch.exp(self.logvar_c / 2)
        else:
            std_c = torch.exp(self.logvar_c )

        KLD = distributions.kl_divergence(distributions.Normal(self.mu_c,std_c),
            distributions.Normal(torch.zeros_like(self.mu_c),torch.ones_like(std_c)))
        return KLD



    def cluster_acc(self, Y_pred, Y):
        
        assert Y_pred.size == Y.size
    
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = [[i, j] for i, j in zip(ind[0], ind[1])]
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w



def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
