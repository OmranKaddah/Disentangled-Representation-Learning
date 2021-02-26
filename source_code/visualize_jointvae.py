import os
#import visdom

import torch
import datetime
from torch import nn


import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from lib.utils import  mkdirs, print_label, LatentTraverser
from models.vade_archs import Vade_mnist, Vade2D
from tqdm import tqdm
from lib import dist

from miscellaneous.args_parsing import args_parsing
from dataset import return_data
from miscellaneous.args_parsing import args_parsing


class Visualizer(object):
    def __init__(self, args):
        self.data_loader = return_data(args)
        
        self.model_name = args.model
        self.z_dim = [args.con_latent_dims, args.disc_latent_dims]
        self.nc = 1
        dim = args.image_size
        mod = __import__('models.joint_b_vae_archs', fromlist=['Joint_VAE_{}'.format(args.arch)])
        self.model = nn.DataParallel(getattr(mod, 'Joint_VAE_{}'.format(args.arch))(z_dim=args.z_dim, n_cls=args.n_cls,
                                computes_std=args.computes_std, nc=args.number_channel, output_size=(dim,dim),
                                output_type= args.output_type).cuda(), device_ids=range(1))
        self.nc =3
        self.output_dir = os.path.join(args.output_dir) 
        self.todays_date = datetime.datetime.today()
        self.output_dir = os.path.join(self.output_dir, str(self.todays_date))
        if not os.path.exists(self.output_dir):
            mkdirs(self.output_dir)

        self.device = 'cuda'
        self.dataset = args.dataset
        self.output_save = args.output_save
        if args.dataset == "mnist":
            self.width = 32
            self.height = 32
        
        else:
            self.width = 64
            self.height = 64
        if  os.path.exists('./AVG_KLDs_VAR_means.pth'):
            AVG_KLDs_var_means = torch.load('./AVG_KLDs_VAR_means.pth')
            self.AVG_KLDs = AVG_KLDs_var_means['AVG_KLDs']
            self.VAR_means = AVG_KLDs_var_means['VAR_means']

        else:
            with torch.no_grad(): 
                self.model.module.eval()
         
                print('computing average of KLDs and variance of means of latent units.')
                N = len(self.data_loader)
                dataset_size = N * args.batch_size
                AVG_KLD = torch.zeros(self.model.module.z_dim[0]).cpu()
                #self.model.eval()
                means = []


                
                print('And computing average of KLDs and variance of means of discrete latent units.')
 
                means = []
                AVG_KLDs_disc = [torch.Tensor([0]) for i in range(len(self.z_dim[1]))]
                for i ,(x, y) in tqdm(enumerate(self.data_loader, 0),total=len(self.data_loader),smoothing=0.9):

                    x = x.cuda(async=True)

                    x_hat, mu, logstd_var, z, alphas, rep_as = self.model(x)

                    loss, recon, KLD_total_mean, KLDs, KLD_catigorical_total_mean, KLDs_catigorical = \
                                        self.model.module.losses(x, x_hat, mu, logstd_var, z, alphas, rep_as, 'H')
                                        
                    for ii in range(len(KLDs_catigorical)):
                        AVG_KLDs_disc[ii] += KLDs_catigorical[ii]

                    KLDs = self.model.module.kld_unit_guassians_per_sample(mu, logstd_var)
                    AVG_KLD += KLDs.sum(0).cpu().detach()
                    means.append(mu.cpu().detach())
                for ii in range(len(AVG_KLDs_disc)):
                    AVG_KLDs_disc[ii] = args.batch_size *AVG_KLDs_disc[ii]/ dataset_size
                self.AVG_KLDs_disc = torch.cat(AVG_KLDs_disc,dim=0)
                self.sorted_idx_KLDs_disc = self.AVG_KLDs_disc.argsort(0)

                self.AVG_KLDs = AVG_KLD / dataset_size
                self.VAR_means = torch.std(torch.cat(means),dim=0).pow(2)
                self.sorted_idx_KLDs = torch.argsort(self.AVG_KLDs,descending=True)
                self.sorted_idx_VAR_means = torch.argsort(self.AVG_KLDs,descending=True)
        mkdirs(os.path.join(self.output_dir,"images"))
    
    def visualize_recon(self):
        one_batch = next(iter(self.data_loader))
        
        x_true, _ = one_batch
        assert len(x_true)>=4, 'set the batch size bigger than 4'
        x_true = x_true[0:5]
        x_recon,_,_,_,_,_ = self.model.module.forward(x_true.cuda())
        if self.model.module.output_type == 'binary':
            x_recon = torch.sigmoid(x_recon)
        x_true = x_true.view(x_recon.shape)
        N = x_true.shape[0]
        fused_images = torch.zeros((2*N,self.nc,x_recon.shape[2],x_recon.shape[3]))
        fused_images[0:2*N-1:2] = x_true
        fused_images[1:2*N:2] = x_recon
        if self.nc ==3:
            fused_images = fused_images * 255
        output_dir = os.path.join(self.output_dir,"images/reconstruction.jpg")
        save_image(fused_images.cpu(),
                        output_dir,nrow=2, pad_value=1)


    def visualize_genertated_imgs(self):
        fused_images = torch.zeros((10,self.nc,self.height,self.width))

        for j in range(10):
            fused_images[j] = self.model.module.generate()
            
        if self.nc ==3:
            fused_images = fused_images * 255
        output_dir = os.path.join(self.output_dir,"images/generatedImgs.jpg")
        save_image(fused_images.cpu(),
                        output_dir,nrow=1, pad_value=1)
    def get_z(self):
        with torch.no_grad(): 
            random_img = self.data_loader.dataset.__getitem__(0)[0]
            random_img = random_img.to(self.device).unsqueeze(0)
            random_z= self.model.module.encoder(random_img)

            fixed_z = []
            Z = {}
            if self.dataset.lower() == 'mnist':
                
                fixed_idx = [550, 5000, 750, 700]
                for id in fixed_idx:
                    img = self.data_loader.dataset.__getitem__(id)[0]
                    img = img.to(self.device).unsqueeze(0)
                    fixed_z.append(self.model.module.encoder(img))
                for ii in range(len(fixed_z)):
                    Z['fixed_{}'.format(ii+1)] =fixed_z[ii]
                Z['random'] = random_z
            
            elif self.dataset.lower() == 'dsprites':
                fixed_idx = [87040, 332800, 578561]# square ellipse heart

                for id in fixed_idx:
                    img = self.data_loader.dataset.__getitem__(id)[0]
                    img = img.to(self.device).unsqueeze(0)
                    fixed_z.append(self.model.module.encoder(img))

                Z = {'fixed_square':fixed_z[0], 'fixed_ellipse':fixed_z[1],
                        'fixed_heart':fixed_z[2], 'random_img':random_z}

            elif self.dataset.lower() == 'celeba':

                fixed_idx = [191281, 143307, 101535, 70059]
                for id in fixed_idx:
                    img = self.data_loader.dataset.__getitem__(id)[0]
                    img = img.to(self.device).unsqueeze(0)
                    fixed_z.append(self.model.module.encoder(img))
                for ii in range(len(fixed_z)):
                    Z['fixed_{}'.format(ii+1)] =fixed_z[ii]
                Z['random'] = random_z

            elif self.dataset.lower() == '3dchairs':

                fixed_idx = [40919, 5172, 22330]
                for id in fixed_idx:
                    img = self.data_loader.dataset.__getitem__(id)[0]
                    img = img.to(self.device).unsqueeze(0)
                    fixed_z.append(self.model.module.encoder(img))
                for ii in range(len(fixed_z)):
                    Z['fixed_{}'.format(ii+1)] =fixed_z[ii]
                Z['random'] = random_z
            elif self.dataset.lower() == '3dshapes':
 
                fixed_idx = [40919, 5172, 22330,1000]
                for id in fixed_idx:
                    img = self.data_loader.dataset.__getitem__(id)[0]
                    img = img.to(self.device).unsqueeze(0)
                    fixed_z.append(self.model.module.encoder(img))
                for ii in range(len(fixed_z)):
                    Z['fixed_{}'.format(ii+1)] =fixed_z[ii]
                Z['random'] = random_z

            else:
                fixed_idx = [0]
                for id in fixed_idx:
                    img = self.data_loader.dataset.__getitem__(id)[0]
                    img = img.to(self.device).unsqueeze(0)
                    fixed_z.append(self.model.module.encoder(img))

                random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)
                random_z_img, random_z_img_var = self.model.module.encoder(random_z)
                Z = {'fixed_img':fixed_z[0], 'random_img':random_z, 
                        'random_z_img':(fixed_img_z,fixed_img_z_var)}
        return Z

    def visualize_traverse(self, limit=3.0, steps=10, ppf= True):
        """
        Parameters:

            limit: float
                range from -limit to +limit. Will be regarded as the variance of
                the guassian distribution if ppf is True
            steps: int
                number of steps in the traversel
            ppf: bool 
                Percent point function (inverse of cdf — percentiles).

        """

        Z = self.get_z()

        if ppf:
            interpolation = torch.linspace(0.05, 0.95, steps)
            guassian = torch.distributions.Normal(0,limit)
            interpolation = guassian.icdf(interpolation)
        else:
            interpolation = torch.linspace(-limit,limit+0.1, steps)

        gifs = []
        list_samples = []
        for key in Z:
            z_ori = Z[key][:,:self.z_dim[0]]
            z_disc = Z[key][:,2*self.model.module.num_latent_dims:]
            d_offest =0
            for d in self.z_dim[1]:
                z_disc[:,d_offest:d_offest+d] = torch.softmax(z_disc[:,d_offest:d_offest+d],dim=-1)
                z_disc[:,d_offest:d_offest+d] = dist.Gumbel_Softmax.sample(z_disc[:,d_offest:d_offest+d],0.67,False)
                d_offest += d
            z_ori = torch.cat([z_ori,z_disc],dim=-1)
            samples = []
            for row in self.sorted_idx_KLDs:
                row = int(row)
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = self.model.module.decoder(z).data
                    if self.model.module.output_type == 'binary':
                        sample = torch.sigmoid(sample)
                    samples.append(sample)
                    # gifs.append(sample)
                samples.append(print_label("z_{}".format(row),
                            size=(self.width,self.height),nc = self.nc).float().unsqueeze(0).cuda())
                samples.append(print_label("%0.2f"%(float(self.AVG_KLDs[row])),
                            size=(self.width,self.height),nc = self.nc).float().unsqueeze(0).cuda())
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.todays_date)

            list_samples.append(samples.clone()*255)

        if self.output_save:
            output_dir = os.path.join(self.output_dir,"images")
            # gifs = torch.cat(gifs)
            # gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, self.height, self.width).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                save_image(list_samples[i].cpu(),
                            os.path.join(output_dir, '{}.jpg'.format(key)),
                            nrow=len(interpolation)+2, pad_value=1)

                #grid2gif(str(os.path.join(self.output_dir, key+'*.jpg')),
                            #str(os.path.join(self.output_dir, key+'.gif')), delay=10)

    def visualize_traverse_disc(self):
        
        Z = self.get_z()
        offest = self.model.module.num_latent_dims
        nldd = self.model.module.num_latent_dims_disc
        max_num_units = max(self.z_dim[1])
        list_samples = []
        for key in Z:
            z_ori = Z[key][:,:offest]
            z_disc = Z[key][:,2*offest:]
            d_offest =0
            for d in self.z_dim[1]:
                z_disc[:,d_offest:d_offest+d] = torch.softmax(z_disc[:,d_offest:d_offest+d],dim=-1)
                z_disc[:,d_offest:d_offest+d] = dist.Gumbel_Softmax.sample(z_disc[:,d_offest:d_offest+d],0.67,False)
                d_offest += d
            z_ori = torch.cat([z_ori,z_disc],dim=-1)
            samples = []
            for row in self.sorted_idx_KLDs_disc:
                row = int(row)
                z = z_ori.clone()
                noffest = offest 
                for j in range(row):
                    noffest += self.z_dim[1][j]
                for k in range(self.z_dim[1][row]):
                    for k_2 in range(self.z_dim[1][row]):
                        if k_2 == k:
                            z[:, noffest + k] = 1
                        else:
                            z[:, noffest + k] = 0
                    sample = self.model.module.decoder(z).data
                    if self.model.module.output_type == 'binary':
                        sample = torch.sigmoid(sample)
                    samples.append(sample)
                    # gifs.append(sample)
                for i in range(max_num_units - self.z_dim[1][row]):
                     samples.append(print_label("-",size=(self.width,self.height)).float().unsqueeze(0).cuda())
                samples.append(print_label("z_{}".format(row),
                                size=(self.width,self.height),nc = self.nc).float().unsqueeze(0).cuda())
                samples.append(print_label("%0.2f"%(float(self.AVG_KLDs_disc[row])),
                                size=(self.width,self.height), nc = self.nc).float().unsqueeze(0).cuda())
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.todays_date)

            list_samples.append(samples.clone()*255)

        if self.output_save:
            output_dir = os.path.join(self.output_dir,"images")
            # gifs = torch.cat(gifs)
            # gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, self.height, self.width).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                save_image(list_samples[i].cpu(),
                            os.path.join(output_dir, '{}_disc.jpg'.format(key)),
                            nrow=max_num_units+2, pad_value=1)

    def visualize_traverse_wrt_var(self, steps = 10,ppf=True):
        
        Z = self.get_z()
        gifs = []
        list_samples = []
        for key in Z:
            z_ori, variances = Z[key][:,:self.z_dim[0]], self.VAR_means
            z_disc = Z[key][:,2*self.model.module.num_latent_dims:]
            d_offest =0
            for d in self.z_dim[1]:
                z_disc[:,d_offest:d_offest+d] = torch.softmax(z_disc[:,d_offest:d_offest+d],dim=-1)
                z_disc[:,d_offest:d_offest+d] = dist.Gumbel_Softmax.sample(z_disc[:,d_offest:d_offest+d],0.67,False)
                d_offest += d
            z_ori = torch.cat([z_ori,z_disc],dim=-1)
            samples = []
            for row in self.sorted_idx_KLDs:
                row = int(row)
                z = z_ori.clone()
                if ppf:
                    interpolation = torch.linspace(0.05, 0.95, steps)
                    guassian = torch.distributions.Normal(0,variances[row])
                    interpolation = guassian.icdf(interpolation)
                else: 
                    interpolation = torch.linspace(int(-variances[row]),int(variances[row]),steps=steps)
    
                        
                for val in interpolation:
                    z[:, row] = val
                    
                    sample = self.model.module.decoder(z).data
                    if self.model.module.output_type == 'binary':
                        sample = torch.sigmoid(sample)
                    samples.append(sample)
                    # gifs.append(sample)
                samples.append(print_label("z_{}".format(row),
                            size=(self.width,self.height),nc = self.nc).float().unsqueeze(0).cuda())
                samples.append(print_label("%0.2f"%(float(self.AVG_KLDs[row])),
                            size=(self.width,self.height),nc = self.nc).float().unsqueeze(0).cuda())
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.todays_date)

            list_samples.append(samples.clone()*255)

        if self.output_save:
            output_dir = os.path.join(self.output_dir,"images")
            # gifs = torch.cat(gifs)
            # gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, self.height, self.width).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                save_image(list_samples[i].cpu(),
                            os.path.join(output_dir, 'WRT_variance_{}.jpg'.format(key)),
                            nrow=len(interpolation)+2, pad_value=1)
    def traverse_generated_samples(self):
        output_dir = os.path.join(self.output_dir,"images")
        traverser = LatentTraverser({'cont':self.z_dim[0], 'disc':self.z_dim[1]})
        #traverser.sample_prior = True
        latent_samples = []
        for cont_idx in self.sorted_idx_KLDs:
            latent_samples.append(traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=10))

                    
        output =  self.model.module.decoder(torch.cat(latent_samples,dim=0).cuda())
        if self.model.module.output_type == 'binary':
            output = torch.sigmoid(output)
        save_image(output,
                    os.path.join(output_dir, 'generated_traversed_cont.jpg'),
                    nrow=10, pad_value=1)
        latent_samples.clear()
        
        for disc_idx in self.sorted_idx_KLDs_disc:
            latent_samples.append(traverser.traverse_line(cont_idx=None,
                                                                    disc_idx=disc_idx,
                                                                    size=3))
        
        output =  self.model.module.decoder(torch.cat(latent_samples,dim=0).cuda())
        if self.model.module.output_type == 'binary':
            output = torch.sigmoid(output)
        save_image(output,
                    os.path.join(output_dir, 'generated_traversed_disc.jpg'),
                    nrow=10, pad_value=1)
        latent_sample = traverser.traverse_grid(cont_idx=2,cont_axis=1,disc_idx=0, disc_axis=0, size=(3,10))
        output = self.model.module.decoder(latent_sample.cuda())
        if self.model.module.output_type == 'binary':
            output = torch.sigmoid(output)
        save_image(output,
                    os.path.join(output_dir, 'generated_traversed_cont_grid.jpg'),
                    nrow=10, pad_value=1)

if __name__ == "__main__":
    args = args_parsing()
    visualizer = Visualizer(args)

    visualizer.visualize_recon()
    visualizer.visualize_traverse(limit=1)
    visualizer.visualize_traverse_disc()
    visualizer.visualize_genertated_imgs()

    visualizer.visualize_traverse_wrt_var()
    visualizer.traverse_generated_samples()