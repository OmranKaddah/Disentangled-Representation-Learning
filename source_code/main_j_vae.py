import itertools
import os
import time
import datetime
import numpy as np
import torch
from sys import exit

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, kl_divergence
from tqdm import tqdm
from dataset import return_data
from miscellaneous.args_parsing import args_parsing
from lib.utils import save_args
from lib.ops import linear_annealing
################



print("Running with seed: " + str(torch.initial_seed()))
args = args_parsing()
if args.name != 'main':
    todays_date =  args.name
else:
    todays_date = datetime.datetime.today()
output_dir = os.path.join(args.output_dir, str(todays_date))
writer = SummaryWriter(os.path.join(output_dir,"tensorboard"))
training_objective = "train_{}".format(args.loss_fun)
objective = 'H'
hyp_params = [args.alpha, args.beta, args.gamma]
hyper_params_disc = [args.alpha, args.beta, args.gamma]
z_dim = [args.con_latent_dims, args.disc_latent_dims]
dim = args.image_size
mod = __import__('models.joint_b_vae_archs', fromlist=['Joint_VAE_{}'.format(args.arch)])
model = nn.DataParallel(getattr(mod, 'Joint_VAE_{}'.format(args.arch))(z_dim=args.z_dim, n_cls=args.n_cls,
                        hyper_params= hyp_params,hyper_params_disc=hyper_params_disc,
                        computes_std=args.computes_std, nc=args.number_channel, output_size=(dim,dim),
                        output_type= args.output_type).cuda(), device_ids=range(1))

data_loader = return_data(args)
#pre_train_optimizer = Adam(model.parameters(), lr=3e-5)
train_optimizer = Adam(model.parameters(), lr=args.lr,
                        betas=(args.beta1, args.beta2))

lr_scheduler = StepLR(train_optimizer, step_size=10, gamma=0.95)
if  os.path.exists('./{}_model_{}.pk'.format(args.model,args.dataset)):
    model.load_state_dict(torch.load('./{}_model_{}.pk'.format(args.model,args.dataset)))



def train_montecarlo(data_loader,ita, epoch, args):
    losses = 0
    N = len(data_loader)
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1

        train_optimizer.zero_grad()
        x = x.cuda()
        x_hat, mu, logvar, z = model(x)
        loss = model.module.vae_loss(x, x_hat, mu, logvar, z)

        loss.backward()
        train_optimizer.step()
        losses += float(loss)
        if ita >= args.max_iter:
            break
        
    if i == N -1:
        losses /= N
    

        print("AVG in itaration {}: loss:{}".format(ita, losses))
    return ita
def pre_train(data_loader,ita,args):
    losses = 0
    N = len(data_loader)
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1

        pre_train_optimizer.zero_grad()
        x = x.cuda()
        x_hat, mu, logvar, z = model(x)
        loss = model.module.vae_loss(x, x_hat, mu, logvar, z)

        loss.backward()
        pre_train_optimizer.step()
        losses += float(loss)
        if ita >= args.max_iter:
            break
        
    if i == N -1:
        losses /= N
    

        print("AVG in itaration {}: loss:{}".format(ita, losses))
    return ita
def train_analytical(data_loader,ita, epoch,args):

    losses = 0
    recons = 0
    KLD_batches = 0
    KLD_cat_batches =0
    #KLDs_batches = np.zeros(z_dim)

    N = len(data_loader)
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1
        train_optimizer.zero_grad()
        x = x.cuda()
        x_hat, mu, logvar, z, alphas, rep_as = model(x)
        model.module.C1 = torch.tensor(linear_annealing(0.0,args.capacity, ita, args.reg_anneal)).float().cuda()
        model.module.C2 = torch.tensor(linear_annealing(0.0,args.capacity_disc, ita, args.reg_anneal)).float().cuda()
        # model.module.C1 = linear_annealing(0.0,args.capacity, ita, args.reg_anneal)
        # model.module.C2 = linear_annealing(0.0,args.capacity_disc, ita, args.reg_anneal)
        loss, recon, KLD_total_mean, KLDs, KLD_catigorical_total_mean, KLDs_catigorical = \
                            model.module.losses(x, x_hat, mu, logvar, z, alphas, rep_as, objective)
        loss.backward()
        train_optimizer.step()
     
        losses += float(loss)
        recons += float(recon)
        KLD_batches += float(KLD_total_mean)
        KLD_cat_batches += float(KLD_catigorical_total_mean)
        KLDs = KLDs.detach().mean(0)
        batch_var_means = torch.std(mu.detach(),dim=0).pow(2)

        writer.add_scalar("Loss/ita",loss,ita)
        writer.add_scalar("recons/ita",recon,ita)
        writer.add_scalar("KLD/ita",KLD_total_mean,ita)
        writer.add_scalar("KLD_cat/ita",KLD_catigorical_total_mean,ita)
        dic = {}
        for j in range(KLDs.shape[0]):  
    
            dic['u_{}'.format(j)] = KLDs[j]

        writer.add_scalars("KLD_units/ita",dic, ita)
        dic.clear()
        dic = {}
        for io in range(len(KLDs_catigorical)):
             
            dic['disc_{}'.format(io)] = KLDs_catigorical[io]

        writer.add_scalars("KLD_cat/ita",dic, ita)
        dic.clear()
        dic = {}
        for j in range(batch_var_means.shape[0]):  
            dic['var_u_{}'.format(j)] = batch_var_means[j]
        writer.add_scalars("std_means/ita",dic, ita)
        dic.clear()

        if ita >=args.max_iter:
            break
    if i == N -1:
        losses /= N
        recons /= N
        KLD_batches /= N
        KLD_cat_batches /= N
        print("After an epoch in itaration_{} AVG: loss:{}, recons:{}, KLD:{}, KLD_cat{} ".format(ita, \
            losses, recons, KLD_batches, KLD_cat_batches))
       
    return ita

def train_TC_montecarlo(data_loader,ita, epoch, args):

    losses = 0
    recons = 0
    KLD_batches = 0
    mutual_info = 0
    totall_coorelation = 0
    regularzation = 0
    disc_mutual_info = 0
    disc_totall_coorelation = 0
    disc_regularzation = 0
    N = len(data_loader)
    dataset_size = N * args.batch_size
    AVG_KLD = torch.zeros(model.module.num_latent_dims)
    means = []
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1
        train_optimizer.zero_grad()
        x = x.cuda(async=True)

        x_hat, mu, logvar, z, alphas, rep_as = model(x)
        model.module.gamma = linear_annealing(0, 1, ita, args.reg_anneal)
        loss, recon, mi, tc, reg, mi_disc, tc_disc, reg_disc = \
            model.module.beta_tc_loss(x, x_hat, mu, logvar, z, alphas, rep_as, dataset_size)
     
        
        loss.backward()
        train_optimizer.step()

        
        losses += float(loss)
        recons += float(recon)
        mutual_info += float(mi)
        totall_coorelation += float(tc)
        regularzation += float(reg)
        disc_mutual_info += float(tc_disc)
        disc_totall_coorelation += float(tc_disc)
        disc_regularzation += float(reg_disc) 
        KLDs = model.module.kld_unit_guassians_per_sample(mu.clone().detach(), logvar.clone().detach())
        KLDs = KLDs.detach().mean(0)
        batch_var_means = torch.std(mu.detach(),dim=0).pow(2)

        if ita == args.max_iter or epoch == args.epochs -1:
            AVG_KLD += KLDs
            means.append(mu.clone().detach())
        ###discrete:

        if model.module.num_latent_dims_disc >0:
            KLDs_catigorical = []
            for alpha in alphas:
                unifom_params = torch.ones_like(alpha)/alpha.shape[1]
                kld = kl_divergence(Categorical(alpha.detach()),Categorical(unifom_params))
                KLDs_catigorical.append(kld.clone().detach().mean())
            
            dic ={}
            for disc_dim in range(len(alphas)):
                dic['disc_{}'.format(disc_dim)] = float(KLDs_catigorical[disc_dim])
            writer.add_scalars("KLDs_disc/ita",dic, ita)
        writer.add_scalar("Loss/ita",loss,ita)
        writer.add_scalar("recons/ita",recon,ita)
        writer.add_scalar("mutual_info/ita",mi,ita)
        writer.add_scalar("totall_coorelation/ita",tc,ita)
        writer.add_scalar("reg/ita",reg,ita)
        writer.add_scalar("mutual_info_disc/ita",mi_disc,ita)
        writer.add_scalar("totall_coorelation_disc/ita",tc_disc,ita)
        writer.add_scalar("reg_disc/ita",reg_disc,ita)
        dic = {}
        AVG_KLD /= dataset_size
        for j in range(KLDs.shape[0]):  
            dic['u_{}'.format(j)] = KLDs[j]

        writer.add_scalars("KLD_units/ita",dic, ita)
        dic.clear()
        dic = {}
        for j in range(batch_var_means.shape[0]):  
            dic['var_u_{}'.format(j)] = batch_var_means[j]
        writer.add_scalars("std_means/ita",dic, ita)
        dic.clear()
        if ita >=args.max_iter:
            break
    
    if i == N -1:
        losses /= N
        recons /= N
        mutual_info /= N
        totall_coorelation /=N
        regularzation /= N
        disc_mutual_info /=N
        disc_totall_coorelation /=N
        disc_regularzation /=N
        print("After an epoch in itaration_%0.2f AVG: loss:%0.2f, recons:%0.2f,mutual_info:%0.2f, totall_coorelation:%0.2f, regularzation%0.2f\
            Discrete = mutual_info:%0.2f, totall_coorelation:%0.2f, regularzation%0.2f"%
            (ita, losses, recons, mutual_info, totall_coorelation, regularzation, disc_mutual_info, disc_totall_coorelation, disc_regularzation))
    if ita ==args.max_iter or epoch == args.epochs -1:
        cat_means = torch.cat(means)
        VAR_means = torch.std(cat_means,dim=0).pow(2)
        torch.save({'AVG_KLDS': AVG_KLD/N,'VAR_means':VAR_means},"AVG_KLDs_VAR_means.pth")
       
    return ita


#___________________________________________________________________________
#PRETRAINING PHASE
#
is_pre_train = False
if is_pre_train:
    print("Started Pretraining Phase.")
    ita = 0
    pretrain_ita = 200000

    inc_per_ita = args.capacity / args.max_iter
    for epoch in range(args.epochs):
        print("epoch number{}".format(epoch))
    
        ita = pre_train(data_loader,ita, pretrain_ita)
        if ita >= pretrain_ita or epoch == 5000:
            break
        lr_scheduler.step(epoch)
    torch.save(model.state_dict(), './betavae_model_{}.pk'.format(args.dataset))
exit_prog = False
if exit_prog:  
    exit()
    
#__________________________________________________________________________
#TRAINING PHASE
#
print("Started Training Phase.")

ita = 0

for epoch in range(args.epochs):
    print("epoch number{}".format(epoch))
    ita = eval(training_objective)(data_loader, ita, epoch, args)
    if ita >= args.max_iter:
        break
    lr_scheduler.step(epoch)

    torch.save(model.state_dict(), './{}_model_{}.pk'.format(args.model,args.dataset))
    save_args({'args': args}, output_dir, 0)