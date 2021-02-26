from torch import nn

from .layers import Reshape

class ConvDecoder(nn.Module):
    def __init__(self, z_dim=16, x_num_param=1, z_dim_disc =0):
        super(ConvDecoder, self).__init__()
        x = x_num_param
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + z_dim_disc, 7 * 7 * int(32*x)),
            nn.ReLU(),
            Reshape((-1, int(32*x), 7, 7)),
            nn.ConvTranspose2d(32, int(64*x), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(64*x), int(32*x), kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(int(32*x), 1, kernel_size=2, stride=1, padding=0),
            #nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class FcDecoder(nn.Module):
    def __init__(self, z_dim=16, x_num_param=1,z_dim_disc =0):
        super(FcDecoder, self).__init__()
        x = x_num_param
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + z_dim_disc, int(2000 *x)),
            nn.ReLU(),
            nn.Linear(int(2000 *x), int(500 *x)),
            nn.ReLU(),
            nn.Linear(int(500 *x), int(500 *x)),
            nn.ReLU(),
            nn.Linear(int(500 *x), 784),
            #nn.Sigmoid(),
            Reshape((-1, 1, 28, 28))
        )

    def forward(self, z):
        return self.decoder(z)
class Dupont_Decoder(nn.Module):
    def __init__(self, z_dim=10, z_dim_disc=0,size=64,nc=1):
        super(Dupont_Decoder, self).__init__()
        decoder_layers = []
        decoder_layers += [
            nn.Linear(z_dim+z_dim_disc, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(),
            Reshape((-1,64,4,4))
        ]
        if size == 64:
            decoder_layers += [
                nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]
        decoder_layers += [
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, nc, (4, 4), stride=2, padding=1),

        ]
        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self,x):
        z = self.decoder(x)
        return z
class Decoder_2Dshapes(nn.Module):
    """Factor-VAE decoder for dsprites"""
    def __init__(self, z_dim, z_dim_disc =0):
        super(Decoder_2Dshapes,self).__init__()

        self.decoder = nn.Sequential(
            Reshape((-1, z_dim + z_dim_disc, 1, 1)),
            nn.Conv2d(z_dim + z_dim_disc, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
            #nn.Sigmoid()
        )
    def forward(self,z):
        return self.decoder(z)

class BetaVAE_Decoder(nn.Module):
    """Decoder proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""
    def __init__(self, z_dim, z_dim_disc =0, nc = 1):
        """
        Inputs:
            z_dim: int 
                size of latent dimension
            nc: int
                number of channels

        """
        super(BetaVAE_Decoder,self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + z_dim_disc, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            Reshape((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
    def forward(self,z):
        return self.decoder(z)

class BetaVAE_Decoder_H(nn.Module):
    """Decoder proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""
    def __init__(self, z_dim, z_dim_disc =0, nc = 1):
        """
        Inputs:
            z_dim: int 
                size of latent dimension
            nc: int
                number of channels

        """
        super(BetaVAE_Decoder_H,self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + z_dim_disc, 256),               # B, 256
            Reshape((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )
    def forward(self,z):
        return self.decoder(z)


# class ConvDecoderBN(nn.Module):
#     """Ricky's arch"""
#     def __init__(self, z_dim, z_dim_disc =0, nc = 1):
#         super(ConvDecoderBN, self).__init__()

#         self.conv1 = nn.ConvTranspose2d(z_dim + z_dim_disc, 512, 1, 1, 0)  # 1 x 1
#         self.bn1 = nn.BatchNorm2d(512)
#         self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
#         self.bn3 = nn.BatchNorm2d(64)
#         self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
#         self.bn4 = nn.BatchNorm2d(32)
#         self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
#         self.bn5 = nn.BatchNorm2d(32)
#         self.conv_final = nn.ConvTranspose2d(32, nc, 4, 2, 1)

#         # setup the non-linearity
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, z):
#         h = z.view(z.size(0), z.size(1), 1, 1)
#         h = self.act(self.bn1(self.conv1(h)))
#         h = self.act(self.bn2(self.conv2(h)))
#         h = self.act(self.bn3(self.conv3(h)))
#         h = self.act(self.bn4(self.conv4(h)))
#         h = self.act(self.bn5(self.conv5(h)))
#         mu_img = self.conv_final(h)
#         return mu_img
class ConvDecoderBN(nn.Module):
    """Ricky's arch"""
    def __init__(self, z_dim, z_dim_disc =0, nc = 1):
        super(ConvDecoderBN, self).__init__()
        self.decoder = nn.Sequential(
        Reshape((-1, z_dim + z_dim_disc, 1, 1)),
        nn.ConvTranspose2d(z_dim + z_dim_disc, 512, 1, 1, 0),  # 1 x 1
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(512, 64, 4, 1, 0),  # 4 x 4
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 8 x 8
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 x 16
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 32 x 32
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, nc, 4, 2, 1)
        )
    
    def forward(self, z):

        mu_img = self.decoder(z)
        return mu_img
class MLPDecoder(nn.Module):
    """Ricky's arch"""
    def __init__(self, z_dim, z_dim_disc =0):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + z_dim_disc, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        

        h = self.net(z)
     
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img

class Discriminator(nn.Module):
    def __init__(self, z_dim, z_dim_disc =0):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim + z_dim_disc
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
    #     self.weight_init()

    # def weight_init(self, mode=''):
    #     if mode != '':
    #     if mode == 'kaiming':
    #         initializer = kaiming_init
    #     elif mode == 'normal':
    #         initializer = normal_init

    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()
