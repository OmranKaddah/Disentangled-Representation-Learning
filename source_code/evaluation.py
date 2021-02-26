import os
import torch

from lib.metrics import  F_VAE_metric, mutual_info_metric_shapes, mutual_info_metric_faces, F_VAE_metric_disc
from dataset import return_data
from miscellaneous.args_parsing import args_parsing


class Evaluator(object):
    def __init__(self, args):
        args.shuffle = False
        self.args = args
        self.model_name = args.model
        self.dataset = args.dataset
        self.nc = 1
        hyp_params = [args.alpha, args.beta, args.gamma]
        if self.model_name != 'jointvae':
            self.z_dim = args.z_dim
            if args.dataset == 'mnist':
                assert os.path.exists('./{}_model_mnist.pk'.format(self.model_name)), 'no model found!'
                from models.beta_vae_archs import Beta_VAE_mnist
                self.model = torch.nn.DataParallel(Beta_VAE_mnist(z_dim=args.z_dim,n_cls=10).cuda(), device_ids=range(1))
                self.model.load_state_dict(torch.load('./{}_model_mnist.pk'.format(self.model_name)))
                dim = 28
            elif args.dataset == 'dsprites':
                assert os.path.exists('./{}_model_dsprites.pk'.format(self.model_name)), 'no model found!'
                if args.loss_fun != 'TC_montecarlo':
                    from models.beta_vae_archs import Beta_VAE_Burgess
                    self.model = torch.nn.DataParallel(Beta_VAE_Burgess(z_dim=args.z_dim, n_cls=3,
                                            computes_std=args.computes_std, with_clusters=args.cluster).cuda(), device_ids=range(1))
                else:
                    from models.beta_vae_archs import Beta_VAE_Ricky_MLP
                    self.model = torch.nn.DataParallel(Beta_VAE_Ricky_MLP(z_dim=args.z_dim,n_cls=3,
                                                    with_clusters=args.cluster).cuda(), device_ids=range(1))

                    dim = 64
                self.model.load_state_dict(torch.load('./{}_model_{}.pk'.format(args.model,args.dataset)))
            else:
                assert os.path.exists('./betavae_model_3Dshapes.pk'), 'no model found!'
                from models.beta_vae_archs import Beta_VAE_BN
                self.model = torch.nn.DataParallel(Beta_VAE_BN(z_dim=args.z_dim, hyper_params= hyp_params, n_cls=4,
                            computes_std=args.computes_std,  nc= args.number_channel, output_type = args.output_type).cuda(),device_ids=range(1))
                self.model.load_state_dict(torch.load('./betavae_model_3Dshapes.pk', map_location="cuda:0"))
                dim = 64
                self.nc =3
        else:
            objective = 'H'
            hyp_params = [args.alpha, args.beta, args.gamma]
            hyper_params_disc = [args.alpha, args.beta, args.gamma]
            z_dim = [args.con_latent_dims, args.disc_latent_dims]
            if args.dataset == 'mnist':
                from models.joint_b_vae_archs import Joint_VAE_mnist, Joint_VAE_Dupont, Joint_VAE_Burgess
                # model = nn.DataParallel(Joint_VAE_mnist(z_dim=z_dim, hyper_params= hyp_params, hyper_params_disc=hyper_params_disc,
                #                         computes_std=args.computes_std).cuda(), device_ids=range(1))
                self.model = torch.nn.DataParallel(Joint_VAE_Burgess(z_dim=z_dim, hyper_params= hyp_params, hyper_params_disc=hyper_params_disc,
                                        computes_std=args.computes_std,output_size=(32,32)).cuda(), device_ids=range(1))
                objective ='B'
                dim = 32
            elif args.dataset == 'dsprites':
                if args.loss_fun != 'TC_montecarlo':
                    from models.joint_b_vae_archs import Joint_VAE_Burgess, Joint_VAE_Dupont
                    self.model = torch.nn.DataParallel(Joint_VAE_Burgess(z_dim=z_dim,
                                            computes_std=args.computes_std).cuda(), device_ids=range(1))
                    # model = nn.DataParallel(Joint_VAE_Dupont(z_dim=z_dim, hyper_params= hyp_params, hyper_params_disc=hyper_params_disc,
                    #                 computes_std=args.computes_std).cuda(), device_ids=range(1))
                else:
                    from models.joint_b_vae_archs import Joint_VAE_Ricky_MLP, Joint_VAE_Burgess
                    #model = nn.DataParallel(Beta_VAE_Ricky_MLP(z_dim=z_dim, gamma= args.gamma, n_cls=3).cuda(), device_ids=range(1))
                    self.model = torch.nn.DataParallel(Joint_VAE_Burgess(z_dim=z_dim, hyper_params= hyp_params, hyper_params_disc=hyper_params_disc,
                                            computes_std=args.computes_std).cuda(), device_ids=range(1))
                self.model.load_state_dict(torch.load('./{}_model_{}.pk'.format(args.model,args.dataset)))
    
                dim = 64
            else:
                from models.joint_b_vae_archs import Joint_VAE_ConvBN    
                self.model = torch.nn.DataParallel(Joint_VAE_ConvBN(z_dim=z_dim, hyper_params= hyp_params, hyper_params_disc=hyper_params_disc,
                                        computes_std=args.computes_std, nc= args.number_channel, output_type= args.output_type).cuda(), device_ids=range(1))
                objective ='H'
                dim = 64
                self.model.load_state_dict(torch.load('./{}_model_{}.pk'.format(args.model,args.dataset)))


    def compute_factor_VAE_mertic(self):
        accuracy, acc_test = F_VAE_metric(self.model,self.args)
        print("Factor-VAE mertic for {} dataset training is: {}".format(self.args.dataset, accuracy))
        print("Factor-VAE mertic for {} dataset test is: {}".format(self.args.dataset, acc_test))


    def compute_factor_VAE_mertic_disc(self):
        accuracy, acc_test = F_VAE_metric_disc(self.model,self.args)
        print("Factor-VAE mertic for {} dataset training is: {}".format(self.args.dataset, accuracy))
        print("Factor-VAE mertic for {} dataset test is: {}".format(self.args.dataset, acc_test))

    def compute_mutual_information_gap(self):
        if self.args.dataset == 'dsprites':
            metric, marginal_entropies, cond_entropies = mutual_info_metric_shapes(self.model, self.args)
        elif self.args.dataset == 'dsprites':
            metric, marginal_entropies, cond_entropies = mutual_info_metric_faces(self.model, self.args)
        else:
            metric = "Not Found!"    
        print("Mutual information for {} dataset is: {}".format(self.args.dataset, metric))
   

if __name__ == "__main__":
    args = args_parsing()
    evaluator = Evaluator(args)

    if args.model != "jointvae":
        evaluator.compute_factor_VAE_mertic()
        evaluator.compute_mutual_information_gap()
    
    else:
        evaluator.compute_factor_VAE_mertic_disc()
