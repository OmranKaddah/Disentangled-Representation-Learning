import os


import torch
import datetime
from torch import nn


import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from miscellaneous.utils import DataGather, mkdirs, grid2gif
from models.vade_archs import Vade_mnist, Vade2D
from miscellaneous.args_parsing import args_parsing
from dataset import return_data
from miscellaneous.args_parsing import args_parsing

class Visualizer(object):
    def __init__(self, args):
        self.data_loader = return_data(args)
        

        self.z_dim = args.z_dim

        if args.dataset == 'mnist':
            assert os.path.exists('./model_mnist.pk'), 'no model found!'
            self.model = Vade_mnist(z_dim=args.z_dim, h_dim=200, n_cls=9).cuda()
            self.model.load_state_dict(torch.load('./model_mnist.pk'))
            dim = 28
        elif args.dataset == 'dsprites':
            assert os.path.exists('./model_dsprites.pk'), 'no model found!'
            self.model = Vade2D(z_dim=args.z_dim, h_dim=200, n_cls=3).cuda()
            self.model.load_state_dict(torch.load('./model_dsprites.pk'))
            dim = 64
        else:
            assert os.path.exists('./model_dsprites.pk'), 'no model found!'
            self.model = Vade_mnist(z_dim=args.z_dim, h_dim=200, n_cls=9).cuda()
            self.model.load_state_dict(torch.load('./model_trained.pk'))
            dim = 64
        self.output_dir = os.path.join(args.output_dir) 
        self.todays_date = datetime.datetime.today()
        self.output_dir = os.path.join(self.output_dir, str(self.todays_date))
        if not os.path.exists(self.output_dir):
            
            mkdirs(self.output_dir)
        self.device = 'cuda'
        self.dataset = args.dataset
        self.output_save = args.output_save
        if args.dataset == "mnist":
            self.width = 28
            self.height = 28
        
        else:
            self.width = 64
            self.height = 64

    def visualize_recon(self):
        one_batch = next(iter(self.data_loader))
        x_true, _ = one_batch
        x_recon,_,_,_ = self.model.forward(x_true.cuda())
        x_true = x_true.view(x_recon.shape)
        N = x_true.shape[0]
        fused_images = torch.zeros((2*N,1,x_recon.shape[2],x_recon.shape[3]))
        fused_images[0:2*N-1:2] = x_true
        fused_images[1:2*N:2] = x_recon
        mkdirs(os.path.join(self.output_dir,"images"))
        output_dir = os.path.join(self.output_dir,"images/reconstruction.jpg")
        save_image(tensor=fused_images.cpu(),
                        filename=output_dir,nrow=2, pad_value=1)

    def visualize_genertated_imgs(self):
        num_cls = self.model.n_cls
        fused_images = torch.zeros((num_cls * 1,1,self.height,self.width))
        for i in range(num_cls):
            for j in range(1):
                fused_images[i*1+j] = self.model.generate(i)
            

        output_dir = os.path.join(self.output_dir,"images/generatedImgs.jpg")
        save_image(tensor=fused_images.cpu(),
                        filename=output_dir,nrow=1, pad_value=1)


    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        #self.net_mode(train=False)


        interpolation = torch.arange(-limit, limit+0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[0]
        random_img = random_img.to(self.device).unsqueeze(0)
        
        random_img_z,_= self.model.encoder(random_img)
        random_img_z = random_img_z [:, :self.z_dim]
        
       
        if self.dataset.lower() == 'mnist':
            fixed_idx1 = 550 
            fixed_idx2 = 5000 
            fixed_idx3 = 750 
            fixed_idx4 = 7000  

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = self.model.encoder(fixed_img1.cuda())[0][:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = self.model.encoder(fixed_img2.cuda())[0][:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = self.model.encoder(fixed_img3.cuda())[0][:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = self.model.encoder(fixed_img4.cuda())[0][:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                    'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                    'random':random_img_z}                

        elif self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = self.model.encoder(fixed_img1.cuda())[0][:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = self.model.encoder(fixed_img2.cuda())[0][:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = self.model.encoder(fixed_img3.cuda())[0][:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                    'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = self.model.encoder(fixed_img1)[0][:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = self.model.encoder(fixed_img2)[0][:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = self.model.encoder(fixed_img3)[0][:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = self.model.encoder(fixed_img4)[0][:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                    'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                    'random':random_img_z}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = self.model.encoder(fixed_img1)[0][:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = self.model.encoder(fixed_img2)[0][:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = self.model.encoder(fixed_img3)[0][:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                    'fixed_3':fixed_img_z3, 'random':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = self.model.encoder(fixed_img)[0][:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)
            random_z_img = self.model.encoder(random_z)[0][:, :self.z_dim]
            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z_img}

        gifs = []
        list_samples = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = self.model.decoder(z).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.todays_date)

            list_samples.append(samples.clone())

        if self.output_save:
            output_dir = os.path.join(self.output_dir,"images")
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), 1, self.height, self.width).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                save_image(tensor=list_samples[i].cpu(),
                            filename=os.path.join(output_dir, '{}.jpg'.format(key)),
                            nrow=len(interpolation), pad_value=1)

                #grid2gif(str(os.path.join(self.output_dir, key+'*.jpg')),
                            #str(os.path.join(self.output_dir, key+'.gif')), delay=10)
    def visualize_traverse_wrt_var(self, steps = 10,loc=-1):
        



        random_img = self.data_loader.dataset.__getitem__(0)[0]
        random_img = random_img.to(self.device).unsqueeze(0)
        
        random_img_z,random_img_z_var= self.model.encoder(random_img)

        
       
        if self.dataset.lower() == 'mnist':
            fixed_idx1 = 550 
            fixed_idx2 = 5000 
            fixed_idx3 = 750 
            fixed_idx4 = 7000  

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1, fixed_img_z1_var = self.model.encoder(fixed_img1.cuda())
    
            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2, fixed_img_z2_var = self.model.encoder(fixed_img2.cuda())

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3, fixed_img_z3_var = self.model.encoder(fixed_img3.cuda())

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4, fixed_img_z4_var = self.model.encoder(fixed_img3.cuda())

            Z = {'fixed_1':(fixed_img_z1,fixed_img_z1_var), 'fixed_2':(fixed_img_z2,fixed_img_z2_var),
                    'fixed_3':(fixed_img_z3,fixed_img_z3_var), 'fixed_4':(fixed_img_z4,fixed_img_z4_var),
                    'random':(random_img_z,random_img_z_var)}                

        elif self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1, fixed_img_z1_var = self.model.encoder(fixed_img1.cuda())
    
            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2, fixed_img_z2_var = self.model.encoder(fixed_img2.cuda())

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3, fixed_img_z3_var = self.model.encoder(fixed_img3.cuda())

            Z = {'fixed_square':(fixed_img_z1,fixed_img_z1_var), 'fixed_ellipse':(fixed_img_z2,fixed_img_z2_var),
                    'fixed_heart':(fixed_img_z3,fixed_img_z3_var), 'random_img':(random_img_z,random_img_z_var)}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1, fixed_img_z1_var = self.model.encoder(fixed_img1.cuda())
    
            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2, fixed_img_z2_var = self.model.encoder(fixed_img2.cuda())

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3, fixed_img_z3_var = self.model.encoder(fixed_img3.cuda())

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4, fixed_img_z4_var = self.model.encoder(fixed_img3.cuda())

            Z = {'fixed_1':(fixed_img_z1,fixed_img_z1_var), 'fixed_2':(fixed_img_z2,fixed_img_z2_var),
                    'fixed_3':(fixed_img_z3,fixed_img_z3_var), 'fixed_4':(fixed_img_z4,fixed_img_z4_var),
                    'random':(random_img_z,random_img_z_var)}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1, fixed_img_z1_var = self.model.encoder(fixed_img1.cuda())
    
            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2, fixed_img_z2_var = self.model.encoder(fixed_img2.cuda())

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3, fixed_img_z3_var = self.model.encoder(fixed_img3.cuda())

            Z = {'fixed_1':(fixed_img_z1,fixed_img_z1_var), 'fixed_2':(fixed_img_z2,fixed_img_z2_var),
                    'fixed_3':(fixed_img_z3,fixed_img_z3_var), 'random':(random_img_z,random_img_z_var)}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z, fixed_img_z_var = self.model.encoder(fixed_img)

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)
            random_z_img, random_z_img_var = self.model.encoder(random_z)
            Z = {'fixed_img':(fixed_img_z,fixed_img_z_var), 'random_img':(random_img_z,random_z_img_var), 'random_z_img':(fixed_img_z,fixed_img_z_var)}

        gifs = []
        list_samples = []
        for key in Z:
            z_ori,variances = Z[key]
            variances = torch.exp(variances)
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                interpolation = torch.linspace(int(-variances[0,row]),int(variances[0,row]),steps=steps)
                                
                for val in interpolation:
                    z[:, row] = val
                    sample = self.model.decoder(z).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.todays_date)

            list_samples.append(samples.clone())

        if self.output_save:
            output_dir = os.path.join(self.output_dir,"images")
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), 1, self.height, self.width).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                save_image(tensor=list_samples[i].cpu(),
                            filename=os.path.join(output_dir, 'WRT_variance_{}.jpg'.format(key)),
                            nrow=len(interpolation), pad_value=1)
        

if __name__ == "__main__":
    args = args_parsing()
    visualizer = Visualizer(args)
    visualizer.visualize_recon()
    visualizer.visualize_traverse()
    visualizer.visualize_genertated_imgs()
    visualizer.visualize_traverse_wrt_var()