import torch.nn as nn
import torch.nn.init as init
from models.joint_b_vae import Joint_VAE
from models.decoders import BetaVAE_Decoder, FcDecoder, BetaVAE_Decoder_H, ConvDecoderBN, MLPDecoder, Dupont_Decoder
from models.encoders import BetaVAE_Encoder, FcEncoder, BetaVAE_Encoder_H, ConvEncoderBN, MLPEncoder, Dupont_Encoder

import numpy as np
import torch
from math import exp, pi, log
from torch import nn, distributions
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable


from models.layers import Lambda

import lib.dist as dist

class Joint_VAE_Burgess(Joint_VAE):
    """
    Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).
    """
    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = BetaVAE_Encoder(self.num_latent_dims,  self.num_latent_dims_disc,self.nc)
        self.decoder = BetaVAE_Decoder(self.num_latent_dims,  self.num_latent_dims_disc, self.nc)
    
class Joint_VAE_mnist(Joint_VAE):

    def init_architecture(self):
        self.width = 28
        self.height = 28
        self.encoder = FcEncoder(self.num_latent_dims,  self.num_latent_dims_disc)
        self.decoder = FcDecoder(self.num_latent_dims,  self.num_latent_dims_disc)


class Joint_VAE_Higgins(Joint_VAE):

    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = BetaVAE_Encoder_H(self.num_latent_dims,  self.num_latent_dims_disc,self.nc)
        self.decoder = BetaVAE_Decoder_H(self.num_latent_dims,  self.num_latent_dims_disc, self.nc)

class Joint_VAE_Ricky(Joint_VAE):

    def init_architecture(self):
        self.width = 64
        self.height = 64

        self.encoder = ConvEncoderBN(self.num_latent_dims,  self.num_latent_dims_disc,self.nc)
        self.decoder = ConvDecoderBN(self.num_latent_dims,  self.num_latent_dims_disc, self.nc)

class Joint_VAE_ConvBN(Joint_VAE):

    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = MLPEncoder(self.num_latent_dims,  self.num_latent_dims_disc)
        self.decoder = MLPDecoder(self.num_latent_dims,  self.num_latent_dims_disc)
class Joint_VAE_Dupont(Joint_VAE):
    def init_architecture(self):
        self.width = self.output_size[0]
        self.height = self.output_size[1]
        self.encoder = Dupont_Encoder(self.num_latent_dims,self.num_latent_dims_disc,self.width,nc=self.nc)
        self.decoder = Dupont_Decoder(self.num_latent_dims,self.num_latent_dims_disc,self.width,nc=self.nc)