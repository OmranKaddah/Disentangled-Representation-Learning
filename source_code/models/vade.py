import pickle

import numpy as np
import torch
from torch import nn, distributions
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from models.layers import Lambda

########################################################
# Vade is an abstract class for all Vade architectures
########################################################
class Vade(nn.Module):
    def __init__(self, z_dim=16, h_dim=64, n_cls=10):
        super(Vade, self).__init__()

        # self.encoder = ConvEncoder(z_dim=z_dim, h_dim=h_dim)
        # self.decoder = ConvDecoder(z_dim=z_dim)
        #self.encoder = FcEncoder(z_dim=z_dim)
        #self.decoder = FcDecoder(z_dim=z_dim)
        self.z_dim = z_dim
        
        self.n_cls = n_cls

        self.pi = nn.Parameter(torch.FloatTensor(n_cls, ).fill_(1) / n_cls, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(n_cls, z_dim).fill_(0), requires_grad=True)
        self.logvar_c = nn.Parameter(torch.FloatTensor(n_cls, z_dim).fill_(0), requires_grad=True)
        #self.weight_init()
        self.init_architecture()
        self.reparam = Lambda(self.reparameterize)
   
    def init_architecture(self):
        pass
    
    # def weight_init(self, mode='normal'):
    #     if mode == 'kaiming':
    #         initializer = kaiming_init
    #     elif mode == 'normal':
    #         initializer = normal_init

    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             initializer(m)

    def initialize_gmm_params(self, gmm):
        pi = torch.from_numpy(gmm.weights_).cuda().float()
        self.pi.data = torch.log(pi / (1 - pi))
        self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
        self.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
    
    def reparameterize(self, arguments):
        mu, logvar = arguments
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

    def generate(self, cluster):
        if cluster >=0 and cluster < self.n_cls:
            mu = self.mu_c[cluster,:]
            log_var = self.logvar_c[cluster,:]
            z = self.reparam(mu,log_var)

            return self.decoder(z)
        else:
            print("An out of scope or invalid number has been entered")


    def pc_given_z(self, z):
        std_c = torch.exp(self.logvar_c / 2)
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

    def losses(self, x, x_hat, mu_z, logvar_z, z):
        std_z = torch.exp(logvar_z / 2)
        std_c = torch.exp(self.logvar_c / 2)
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
        std_c = torch.exp(self.logvar_c / 2)
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

# def kaiming_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)


# def normal_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         init.normal_(m.weight, 0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
