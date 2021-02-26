import torch.nn as nn
import torch.nn.init as init
from models.b_vae import Beta_VAE
from models.decoders import BetaVAE_Decoder, FcDecoder, BetaVAE_Decoder_H, ConvDecoderBN, MLPDecoder, ConvDecoder
from models.encoders import BetaVAE_Encoder, FcEncoder, BetaVAE_Encoder_H, ConvEncoderBN, MLPEncoder, ConvEncoder

import numpy as np
import torch
from math import exp, pi, log
from torch import nn, distributions
from torch.nn import functional as F



from models.layers import Lambda

import lib.dist as dist

class Beta_VAE_Burgess(Beta_VAE):
    """
    Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018).
    """
    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = BetaVAE_Encoder(self.z_dim,nc = self.nc)
        self.decoder = BetaVAE_Decoder(self.z_dim, nc = self.nc)
    
class Beta_VAE_mnist(Beta_VAE):

    def init_architecture(self):
        self.width = 28
        self.height = 28
        self.encoder = FcEncoder(self.z_dim, self.x_num_param)
        self.decoder = FcDecoder(self.z_dim, self.x_num_param)
class Beta_VAE_mnist_conv(Beta_VAE):

    def init_architecture(self):
        self.width = 28
        self.height = 28
        self.encoder = ConvEncoder(self.z_dim, self.x_num_param)
        self.decoder = ConvDecoder(self.z_dim, self.x_num_param)


class Beta_VAE_Higgins(Beta_VAE):

    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = BetaVAE_Encoder_H(self.z_dim,nc = self.nc)
        self.decoder = BetaVAE_Decoder_H(self.z_dim, nc = self.nc)

class Beta_VAE_BN(Beta_VAE):

    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = ConvEncoderBN(self.z_dim,nc = self.nc)
        self.decoder = ConvDecoderBN(self.z_dim, nc = self.nc)

class Beta_VAE_Ricky_MLP(Beta_VAE):

    def init_architecture(self):
        self.width = 64
        self.height = 64
        self.encoder = MLPEncoder(self.z_dim)
        self.decoder = MLPDecoder(self.z_dim)
