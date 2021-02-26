import numpy as np
import torch
from math import exp, pi, log
from torch import nn
from torch.distributions import kl_divergence, Normal, Categorical
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable
import torch.nn.init as init

from models.layers import Lambda

import lib.dist as dist


class Joint_VAE(nn.Module):
    def __init__ (self, z_dim = [10,[]], nc = 1, hyper_params =[1,1,1], hyper_params_disc =[1,1,1], 
                    capacity = [5,5], output_type = 'binary',  weight_init = '', computes_std = True,output_size = (64,64)):
        super(Joint_VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.computes_std = computes_std
        #number of dimensions of encoder's output
        self.num_latent_dims =  z_dim[0]
     
        self.output_size = output_size
        self.num_latent_dims_disc = sum([dim for dim in  z_dim[1]])        
        self.init_architecture()
        self.reparam = Lambda(self.reparameterize)
        self.reparam_disc = Lambda(self.reparameterize_discrete)
        self.alpha = hyper_params[0]
        self.beta = hyper_params[1]
        self.gamma = hyper_params[2]
        self.alpha_disc = hyper_params_disc[0]
        self.beta_disc = hyper_params_disc[1]
        self.gamma_disc = hyper_params_disc[2]
        #capactiy of continues bottleneck
        self.C1 = capacity[0]
        #capacity of  discrete bottleneck
        self.C2 = capacity[1]
        self.output_type = output_type
        self.temperature = .67


        if weight_init =='xavier':
            self.apply(xavier_init)
        elif weight_init == 'kaiming':
            self.apply(kaiming_init)
        

    def init_architecture(self): 
        pass

    
    def forward(self, x):
        params = self.encoder(x)
        
        mu, logstd_var = params[:, 0: self.z_dim[0]], params[:,self.z_dim[0]:2*self.z_dim[0]]
        d = 2*self.z_dim[0] #starting index of discrete vars
        z = self.reparam(mu, logstd_var)
        alphas = []
        rep_as = 0
        if len(self.z_dim[1])>0:

            for dims in self.z_dim[1]:
                alphas.append(torch.softmax(params[:,d:d+dims],dim=-1))
                d += dims
            rep_as = self.reparam_disc(alphas)
            z = torch.cat([z,torch.cat(rep_as,dim=-1)],dim=1)
        
        x_hat = self.decoder(z)
        return x_hat, mu, logstd_var, z, alphas, rep_as

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
    def reparameterize_discrete(self, arguments):
        reparm_list =[]
        for alpha in arguments:
            reparm_list.append(dist.Gumbel_Softmax.sample(alpha,self.temperature))
        
        return reparm_list

    def generate(self):
        "Generates a random sample"
        x_hat = torch.randn((1,self.z_dim[0])).cuda()
        alphas = []
     
        if len(self.z_dim[1])>0:
            for dims in self.z_dim[1]:
                alphas.append(torch.rand((1,dims)).cuda())
        rep_as = []    
        for alpha in alphas:
            rep_as.append(dist.Gumbel_Softmax.sample(alpha,self.temperature,False))
            x_hat = torch.cat([x_hat,torch.cat(rep_as,dim=-1)],dim=1)
        if self.output_type == 'binary':
            x_hat = torch.sigmoid(x_hat).view(1,-1)
        return self.decoder(x_hat)


  
    def vae_loss(self,x, x_logit, mean, logstd_var, z):
        """VAE loss where it is calculated with a monte-carlo estimation of negative ELBO
        """


        if not self.computes_std:
            std_z = torch.exp(logstd_var / 2)
        else:
            std_z = torch.exp(logstd_var)   
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
        #
        # An implementation with no use of custom lib.dist
        #
        # logpx_z = -F.binary_cross_entropy_with_logits(x_logit, x, reduction='mean') * self.width*self.height
        # logpz = self.log_normal_pdf(z, torch.tensor([0.]).cuda(), torch.tensor([0.]).cuda())
        # logqz_x = self.log_normal_pdf(z, mean, logvar)
        # return -torch.mean(logpx_z + logpz - logqz_x)

    def losses(self, x, x_hat, mu_z, logstd_var, z, alphas, rep_as, objective='B'):
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
            recons = F.mse_loss(x_hat,x, reduction='mean') * self.width*self.height
        KLDs = kl_divergence(Normal(mu_z,std_z),Normal(torch.zeros_like(mu_z),torch.ones_like(std_z)))
        KLD_total_mean = KLDs.sum(-1).mean()

        KLDs_catigorical = []
        # KLD_catigorical_total_mean = torch.zeros(x.shape[0],requires_grad=True).cuda()
        KLD_catigorical_total_mean = 0
        if self.num_latent_dims_disc >0:
            for alpha in alphas:
                unifom_params = torch.ones_like(alpha)/alpha.shape[1]
                kld = kl_divergence(Categorical(alpha),Categorical(unifom_params))
                KLDs_catigorical.append(kld.clone().detach().mean())
                KLD_catigorical_total_mean += kld.mean(0)
            # KLD_catigorical_total_mean = KLD_catigorical_total_mean.mean(0)
            
        if objective== 'B':
            loss = recons + self.gamma * torch.abs(KLD_total_mean - self.C1) \
                        + self.gamma_disc * torch.abs(KLD_catigorical_total_mean - self.C2)
        else:
            loss = recons + self.beta * KLD_total_mean \
                    + self.beta_disc * KLD_catigorical_total_mean

        return loss, recons, KLD_total_mean, KLDs, KLD_catigorical_total_mean, KLDs_catigorical

    
    def beta_tc_loss(self, x, x_hat, mu_z, logstd_var, z, alphas, rep_as, dataset_size):
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
            logstd_var = logstd_var / 2
        prior_dist, q_dist = dist.Normal(), dist.Normal()
        prior_params= torch.zeros(self.z_dim[0], 2)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        expanded_size = (batch_size,) + prior_params.size()
        prior_params = prior_params.expand(expanded_size).cuda().requires_grad_()
        z_params = torch.cat([mu_z.view(batch_size,self.num_latent_dims,1),
                                logstd_var.view(batch_size,self.num_latent_dims,1)],dim=2)

        if self.output_type == 'binary':
            recons = F.binary_cross_entropy_with_logits(x_hat, x, reduction='mean') * self.width*self.height
        else:
            recons = F.mse_loss(x_hat,x, reduction='mean') * self.width*self.height
        z_cont = z[:,:self.num_latent_dims]
        logpz = prior_dist.log_density(z_cont, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = q_dist.log_density(z_cont, params=z_params).view(batch_size, -1).sum(1)

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = q_dist.log_density(
            z_cont.view(batch_size, 1, self.z_dim[0]),
            z_params.view(1, batch_size, self.z_dim[0], 2)
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

        #For discrete 
        _logqy_s = []
        for alpha, sample_i in zip(alphas,rep_as):
            _logqy_s.append(dist.Gumbel_Softmax.log_density(alpha.view(1,batch_size,-1)
                            ,sample_i.view(batch_size,1,-1),self.temperature).view(1,batch_size,batch_size))
        
        _logqy_s = torch.cat(_logqy_s,dim=0)

        logqy_s_prodmarginals = (torch.logsumexp(_logqy_s, dim=1, keepdim=False) - log(batch_size * dataset_size)).sum(0)
        logqy = (torch.logsumexp(_logqy_s.sum(0), dim=1, keepdim=False) - log(batch_size * dataset_size))

        logqy_condx = torch.zeros_like(logqy)
        for alpha, sample_i in zip(alphas,rep_as):
            logqy_condx += dist.Gumbel_Softmax.log_density(alpha,sample_i,self.temperature)
        #mutual information
        mi_disc = (logqy_condx - logqy).mean()
        #total coorelation
        tc_disc = (logqy - logqy_s_prodmarginals).mean()
        #dimension-wise KL. Here name regularization
        reg_disc = (logqy_s_prodmarginals - logqy).mean()

        modified_elbo = recons + \
                        self.alpha *(mi) + \
                        self.beta * tc + \
                        self.gamma *reg +\
                        self.alpha_disc * mi_disc +\
                        self.beta_disc* tc_disc +\
                        self.gamma_disc * reg_disc

        return modified_elbo, recons, mi, tc, torch.abs(reg), mi_disc, tc_disc, torch.abs(reg_disc)


    def kld_unit_guassians_per_sample(self,mu,logstd_var):
        if not self.computes_std:
            std = torch.exp(logstd_var / 2)
        else:
            std = torch.exp(logstd_var)
        KLD = kl_divergence(Normal(mu,std),
            Normal(torch.zeros_like(mu),torch.ones_like(std)))
        return KLD
    


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