from torch import nn
from torch.nn.modules.flatten import Flatten
from torch import cat
from models.layers import Reshape
class ConvEncoder(nn.Module):
    def __init__(self, z_dim=10, x_num_param=1, z_dim_disc =0):
        super(ConvEncoder, self).__init__()
        x = x_num_param
        self.encoder = nn.Sequential(
            nn.Conv2d(1, int(32*x), kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(int(32*x), int(64*x), kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(int(64*x)*6*6, 100),
            nn.Linear(100, 2*z_dim + z_dim_disc)
        )

    def forward(self, x):
        z = self.encoder(x)

        return z

class FcEncoder(nn.Module):
    def __init__(self, z_dim=10, x_num_param=1,z_dim_disc =0):
        super(FcEncoder, self).__init__()
        x = x_num_param
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(784, int(500 *x)),
            nn.ReLU(),
            nn.Linear(int(500 *x), int(500 *x)),
            nn.ReLU(),
            nn.Linear(int(500 *x), int(2000 *x)),
            nn.ReLU(),
            nn.Linear(int(2000 *x), 2*z_dim + z_dim_disc)
        )



    def forward(self, x):
        z = self.encoder(x)

        return z
class Dupont_Encoder(nn.Module):
    def __init__(self,z_dim=10, z_dim_disc =0,size=64,nc=1):
        super(Dupont_Encoder,self).__init__()
        encoder_layers = []
        encoder_layers = [
            nn.Conv2d(nc, 32, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]
        # Add additional layer if (64, 64) images
        if size== 64:
            encoder_layers += [
                nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]
        elif size == 32:
            # (32, 32) images are supported but do not require an extra layer
            pass
        else:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        # Add final layers
        encoder_layers += [
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]
        encoder_layers +=[
            Flatten(),
            nn.Linear(64 * 4 * 4,256),
            nn.ReLU(),
            nn.Linear(256,2*z_dim+z_dim_disc)
        ]
        self.encoder = nn.Sequential(*encoder_layers)
    def forward(self,x):
        z = self.encoder(x)
        return z

class Encoder_2Dshapes(nn.Module):
    """from facor-vae"""
    def __init__(self,z_dim=10, z_dim_disc =0):
        super(Encoder_2Dshapes,self).__init__()
   
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim + z_dim_disc, 1),
            Flatten()
        )
    def forward(self,x):
        z = self.encoder(x)

        return z  

class BetaVAE_Encoder(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""
    def __init__(self,z_dim=10, z_dim_disc =0,nc=1):
        super(BetaVAE_Encoder,self).__init__()
        self.z_dim  = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            Reshape((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 2*z_dim + z_dim_disc),             # B, z_dim*2
        )
    def forward(self,x):
        z = self.encoder(x)

        return z  


class BetaVAE_Encoder_H(nn.Module):
    """Higgens """
    def __init__(self ,z_dim, z_dim_disc= 0,nc = 1 ):
        super(BetaVAE_Encoder_H,self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            Reshape((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, 2*z_dim + z_dim_disc),             # B, z_dim*2
        )
    def forward(self,x):
        z = self.encoder(x)

        return z  

class ConvEncoderBN(nn.Module):
    def __init__(self, z_dim, z_dim_disc= 0,nc = 1):
        super(ConvEncoderBN, self).__init__()
        self.output_dim = 2*z_dim + z_dim_disc
        self.nc = nc
        self.conv1 = nn.Conv2d(nc, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, self.output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, self.nc, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.conv5(h)
        h = self.act(self.bn5(h))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z 

class MLPEncoder(nn.Module):
    """Ricky's arch"""
    def __init__(self, z_dim, z_dim_disc= 0, nc = 1):
        super(MLPEncoder, self).__init__()
        self.z_dim = 2*z_dim + z_dim_disc

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, z_dim)

       
        self.flatter = Flatten()
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.flatter(x)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0),  self.z_dim)
        return z
        #return z