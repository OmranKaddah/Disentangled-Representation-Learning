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
from tqdm import tqdm
from dataset import return_data
from miscellaneous.args_parsing import args_parsing
from lib.utils import save_args
from lib.ops import  permute_dims, linear_annealing
from models.decoders import Discriminator
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
objective = args.analytical
hyp_params = [args.alpha, args.beta, args.gamma]
dim = args.image_size
   
mod = __import__('models.beta_vae_archs', fromlist=['Beta_VAE_{}'.format(args.arch)])
model = nn.DataParallel(getattr(mod, 'Beta_VAE_{}'.format(args.arch))(z_dim=args.z_dim, n_cls=args.n_cls,
                            hyper_params= hyp_params,
                        computes_std=args.computes_std, nc=args.number_channel, 
                        output_type= args.output_type, x_num_param= args.x_num_param).cuda(), device_ids=range(1))

data_loader = return_data(args)
#pre_train_optimizer = Adam(model.parameters(), lr=3e-5)
train_optimizer = Adam(model.parameters(), lr=args.lr,
                        betas=(args.beta1, args.beta2))
if args.factorvae_tc:
    discriminator = Discriminator(args.z_dim).cuda()
    optim_D = Adam(discriminator.parameters(), lr=args.lr_discriminator)
    ONES = torch.ones(args.batch_size, dtype=torch.long).cuda()
    ZEROS = torch.zeros(args.batch_size, dtype=torch.long).cuda()
    training_objective += '_factorvae_tc'
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
    #KLDs_batches = np.zeros(args.z_dim)

    N = len(data_loader)
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1
        train_optimizer.zero_grad()
        x = x.cuda()
        x_hat, mu, logvar, z, param = model(x)
        model.module.capacity = torch.tensor(linear_annealing(0,args.capacity, ita, args.reg_anneal)).float().cuda().requires_grad_()

        loss, recon, KLD_total_mean, KLDs = model.module.losses(x, x_hat, mu, logvar, z, objective, equality=args.kl_equality)
        loss.backward()
        train_optimizer.step()
     
        losses += float(loss.detach())
        recons += float(recon.detach())
        KLD_batches += float(KLD_total_mean.detach())
        batch_var_means = torch.std(mu.detach(),dim=0).pow(2)
        KLDs = KLDs.detach().mean(0)
        writer.add_scalar("Loss/ita",loss.detach(),ita)
        writer.add_scalar("recons/ita",recon.detach(),ita)
        writer.add_scalar("KLD/ita",KLD_total_mean,ita)
        dic = {}
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
        KLD_batches /= N

        print("After an epoch in itaration_{} AVG: loss:{}, recons:{}, KLD:{}".format(ita, losses, recons, KLD_batches))
       
    return ita
def train_analytical_factorvae_tc(data_loader,ita, epoch,args):

    losses = 0
    recons = 0
    KLD_batches = 0
    #KLDs_batches = np.zeros(args.z_dim)

    N = len(data_loader)
    for i ,(x, x2, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1
        train_optimizer.zero_grad()
        x = x.cuda()
        x_hat, mu, logvar, z, param = model(x)
        model.module.capacity = torch.tensor(linear_annealing(0,args.capacity, ita, args.reg_anneal)).float().cuda()
        vae_tc_loss = 0
        if args.factorvae_tc:
            dz  = discriminator(z)

            vae_tc_loss = (dz[:, :1] - dz[:, 1:]).mean()

        loss, recon, KLD_total_mean, KLDs = model.module.losses(x, x_hat, mu, logvar, z, objective, vae_tc_loss)
        loss.backward(retain_graph=True)
        train_optimizer.step()


        x2 = x2.cuda()
        with torch.no_grad(): 
            params = model.module.encoder(x2).view(x2.size(0),args.z_dim,2)
            mu, logstd_var = params.select(-1,0), params.select(-1,1)
            z_prime = model.module.reparam(mu, logstd_var)
        
            z_pperm = permute_dims(z_prime).clone()

            params = model.module.encoder(x).view(x.size(0),args.z_dim,2)
            mu, logstd_var = params.select(-1,0), params.select(-1,1)
            dz_pr = model.module.reparam(mu, logstd_var).clone()
        
       
        D_z_pperm = discriminator(z_pperm)
        D_tc_loss = 0.5*(F.cross_entropy(dz_pr, ZEROS) + F.cross_entropy(D_z_pperm, ONES))

        optim_D.zero_grad()
        D_tc_loss.backward()
        optim_D.step()

        losses += float(loss.clone())
        recons += float(recon)
        KLD_batches += float(KLD_total_mean)

        KLDs = KLDs.detach().mean(0)

        writer.add_scalar("Loss/ita",loss,ita)
        writer.add_scalar("recons/ita",recon,ita)
        writer.add_scalar("KLD/ita",KLD_total_mean,ita)
        dic = {}
        for j in range(KLDs.shape[0]):  
    
            dic['u_{}'.format(j)] = KLDs[j]

        writer.add_scalars("KLD_units/ita",dic, ita)
        dic.clear()
        if ita >=args.max_iter:
            break
    if i == N -1:
        losses /= N
        recons /= N
        KLD_batches /= N

        print("After an epoch in itaration_{} AVG: loss:{}, recons:{}, KLD:{}".format(ita, losses, recons, KLD_batches))
       
    return ita

def train_TC_montecarlo(data_loader,ita, epoch, args):

    losses = 0
    recons = 0
    KLD_batches = 0
    mutual_info = 0
    totall_coorelation = 0
    regularzation = 0

    N = len(data_loader)
    dataset_size = N * args.batch_size
    AVG_KLD = torch.zeros(model.module.z_dim)
    means = []
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):

        ita += 1
        train_optimizer.zero_grad()
        x = x.cuda(async=True)

        x_hat, mu, logvar, z, params = model(x)
        model.module.gamma = linear_annealing(0, 1, ita, args.reg_anneal)
        loss, recon, mi, tc, reg = model.module.beta_tc_loss(x, x_hat, params, z, dataset_size)
     
        
        loss.backward()
        train_optimizer.step()

        
        losses += float(loss)
        recons += float(recon)
        mutual_info += float(mi)
        totall_coorelation += float(tc)
        regularzation += float(reg)
        KLDs = model.module.kld_unit_guassians_per_sample(mu.clone().detach(), logvar.clone().detach())
        KLDs = KLDs.detach().mean(0)
        batch_var_means = torch.std(mu.detach(),dim=0).pow(2)
        #var_means = torch.std(mu.detach(),dim=0).pow(2)
        if ita == args.max_iter or epoch == args.epochs -1:
            AVG_KLD += KLDs
            means.append(mu.clone().detach())
        
        writer.add_scalar("Loss/ita",loss,ita)
        writer.add_scalar("recons/ita",recon,ita)
        writer.add_scalar("mutual_info/ita",mi,ita)
        writer.add_scalar("totall_coorelation/ita",tc,ita)
        writer.add_scalar("reg/ita",reg,ita)
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

        print("After an epoch in itaration_{} AVG: loss:{}, recons:{},mutual_info:{}, totall_coorelation:{}, regularzation{}".format
                (ita, losses, recons, mutual_info, totall_coorelation, regularzation))
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
    #lr_scheduler.step(epoch)

    torch.save(model.state_dict(), './betavae_model_{}.pk'.format(args.dataset))

    save_args({'args': args}, output_dir, 0)
    if args.factorvae_tc:
        torch.save(discriminator.state_dict(), './discriminator.pk')