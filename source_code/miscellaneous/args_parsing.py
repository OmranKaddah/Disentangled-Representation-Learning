"""main.py"""

import argparse
import numpy as np
import torch
import datetime
from lib.utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def args_parsing():

    parser = argparse.ArgumentParser(description='VAE')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--model', default='betavae', type=str, 
                        help='type of model to run or visualize its results: betavae or vade',
                        choices= ['betavae','vade','jointvae'])
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--initialization_itarations', default=2, type=int, help='number of pre-train epochs')
    parser.add_argument('--epochs', default=60, type=int, help='number of pre-train epochs')
    parser.add_argument('--max_iter', default=7e5, type=float, help='maximum training iteration')
    parser.add_argument('--analysis_after_epochs', default=1, type=int, help='analysis after number of epochs')
    parser.add_argument('--is_initialized', default=False, type=str2bool, help='Whether to GMM initalization for the clusters')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate of the model')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--output_type', default='binary', type=str, help='output type, whether it is continues or binary')
    parser.add_argument('--loss_fun', default='analytical', type=str, help='select loss function from list of choices \
                        for how the elbo is computed. \
                        montecarlo: monte carolo estimation for the objective funciton of VAE with one sample \
                        analytical: computes the objective function analytical for this option select how to control\
                                    the KLD term',
                        choices= ['montecarlo','analytical','TC_montecarlo']  )
    parser.add_argument('--arch', default='Burgess', type=str, help='select architecutre of encoder and decoder',
                        choices= ['mnist','Burgess','Higgins', 'ConvBN', 'Ricky_MLP','Dupont','BN','mnist_conv']  )
    parser.add_argument('--analytical', default='H', type=str, help='select loss function from list of choices \
                        for how the elbo is computed. \
                        H: Higgens loss analytical fucntion \
                        B: computes the objective function Burgiss analytical for this option select how to control\
                                    the KLD term',
                        choices= ['H','B']  )
    parser.add_argument('--kl_equality', default=False, type=str2bool, help='pushes embeddings to have unit with equal kl divergences')
    ## this command works only for main_vade.py and main_b_vae.py 
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    ###
    
    parser.add_argument('--computes_std', default=False, type=str2bool, help='whether the output in the laten \
                        dimension is log standard deviation or log variance, \
                        if set to true then it will be for std, otherwise var.')
    parser.add_argument('--number_channel', default=1, type=int, help='number of channels')
    parser.add_argument('--n_cls', default=3, type=int, help='number of clusters')
    parser.add_argument('--alpha', default=-5, type=float, help='gamma hyperparameter')
    parser.add_argument('--beta', default=6, type=float, help='gamma hyperparameter')
    parser.add_argument('--gamma', default=150, type=float, help='gamma hyperparameter')
    parser.add_argument('--capacity', default=40, type=float, help='capacity of continues of ')
    parser.add_argument('-a', '--reg_anneal', type=float, default=300000,
                        help="Number of annealing steps where gradually adding the regularisation.\
                             What is annealed is specific to each loss.")
    parser.add_argument('--x_num_param', default=1, type=float, help='x time number of parameters for encoder and decoder')
    parser.add_argument('--cluster', default=True, type=str2bool, help='with cluster')
    ### Factor VAE stuff
    parser.add_argument('--factorvae_tc', default=False, type=str2bool, help='adds total correlation term from FactorVAE paper')
    parser.add_argument('--lr_discriminator', default=1e-3, type=float, help='learning rate of the model')
    ####
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='dsprites', type=str, help='dataset name',
                        choices= ['3Dshapes','mnist','dsprites'])
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    #
    #  arguments only for joint-VAE on main_j_vae.py
    parser.add_argument('--con_latent_dims', default=6, type=int, help='dimension of the representation z')
    parser.add_argument('--disc_latent_dims', default=[4], type=int, help='dimension of the representation z')
    parser.add_argument('--alpha_disc', default=1, type=float, help='gamma hyperparameter')
    parser.add_argument('--beta_disc', default=1, type=float, help='gamma hyperparameter')
    parser.add_argument('--gamma_disc', default=150.0, type=float, help='gamma hyperparameter')
    parser.add_argument('--capacity_disc', default=1.1, type=float, help='gamma hyperparameter')
    parser.add_argument('--temprature', default=150.0, type=float, help='temprature of gumbel-softmax')
    ##################################################

    #parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    #parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    #parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')
    # parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--shuffle', default=True, type=str2bool, help='shuffle dataset')
    parser.add_argument('--output_dir', default='outputs_new', type=str, help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool, help='whether to save traverse results')

    args = parser.parse_args()

    return args
