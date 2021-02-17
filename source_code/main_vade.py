import itertools
import os
import time
import datetime
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.mixture import GaussianMixture

from models.vade_archs import Vade2D, Vade_mnist
from models.beta_vae_archs import Beta_VAE_Burgess
from dataset import return_data, get_mnist
from miscellaneous.args_parsing import args_parsing
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
dim = args.image_size
   
mod = __import__('models.beta_vae_archs', fromlist=['Beta_VAE_{}'.format(args.arch)])
model = nn.DataParallel(getattr(mod, 'Beta_VAE_{}'.format(args.arch))(z_dim=args.z_dim, n_cls=args.n_cls,
                        output_size=(dim,dim),hyper_params= hyp_params,
                        computes_std=args.computes_std, nc=args.number_channel, 
                        output_type= args.output_type).cuda(), device_ids=range(1))

data_loader = return_data(args)
pre_train_optimizer = Adam(itertools.chain(model.module.encoder.parameters(), model.module.decoder.parameters()), lr=1e-3)
train_optimizer = Adam(model.module.parameters(), lr=2e-3)
lr_scheduler = StepLR(train_optimizer, step_size=10, gamma=0.95)
if  os.path.exists('./model_{}.pk'.format(args.dataset)):
    model.load_state_dict(torch.load('./model_{}.pk'.format(args.dataset)))
def pre_train(data_loader):
    loss = []
    for x, y in data_loader:
        pre_train_optimizer.zero_grad()
        x = x.view((-1, 1, dim, dim)).cuda()
        mu, _ = model.module.encoder(x)
        x_hat = model.module.decoder(mu)
        L = dim*dim * F.binary_cross_entropy_with_logits(x_hat, x, reduction='none').mean()
        loss.append(L)
        L.backward()
        pre_train_optimizer.step()
    return sum(loss)/len(data_loader)
if not args.is_initialized:
    for ita in range(args.initialization_itarations):

        loss = pre_train(data_loader)
        if ita % args.analysis_after_epochs == 0:
            # analyser.write_state(epoch, weights=False)
            latenet_embeddings = []
            Y = []
            for x,y in data_loader:       
                Mu, _ = model.module.cpu().float().encoder(x)
                latenet_embeddings.append(Mu.detach())
                Y.append(y)
            latenet_embeddings = torch.cat(latenet_embeddings)
            gmm = GaussianMixture(n_components=args.n_cls, covariance_type='diag', n_init=5)
            #pre = gmm.fit_predict(Mu.detach().cpu().numpy())
            pre = gmm.fit_predict(latenet_embeddings)
            cluster_acc = model.module.cluster_acc(pre,torch.cat(Y).numpy().astype(int))
            print("Loss: {}, cluster acc: {}".format(loss, cluster_acc[0]))
            model.cuda()

        end_time = time.time()

        print("Initialization phase, itaration number{}".format(ita))

    model.module.initialize_gmm_params(gmm)
    torch.save(model.state_dict(), './model_{}.pk'.format(args.dataset))
    model.cuda()
else:
    print("If you do not have an intialized model, then set --is_initialized False when running the model.module.")
    model.load_state_dict(torch.load('./model_{}.pk'.format(args.dataset)))



def train(data_loader,epoch):
    

   
    Y = []
    pc_given_zs = []
    
    total_bl = 0 #total batch loss
    total_recons = 0
    total_KLD = 0
    total_KLD_c = 0
    N = len(data_loader)
    for i ,(x, y) in tqdm(enumerate(data_loader, 0),total=len(data_loader),smoothing=0.9):
        #print("Inside the training loop")
        train_optimizer.zero_grad()
        x = x.cuda()
      
        x_hat, mu, logvar, z = model(x)
        #BCE, KLD, KLD_c, L_sparsity = model.module.module.losses(x, x_hat, mu, logvar, z)
        if args.dataset != 'dsprites':
            BCE, KLD, KLD_c, L_sparsity, pc_given_z = model.module.losses(x, x_hat, mu, logvar, z)
        else:
            BCE, KLD, KLD_c, L_sparsity, pc_given_z = model.module.losses_clustering(x, x_hat, mu, logvar, z) 
        pc_given_zs.append(pc_given_z)
        batch_loss = KLD + BCE + KLD_c + L_sparsity
        batch_loss.backward()
        train_optimizer.step()
        Y.append(y)
        total_bl += float(batch_loss)
        total_recons += float(BCE)
        total_KLD += float(KLD)
        total_KLD_c += float(KLD_c)

    if(args.dataset == 'l'): 
        y_pred = model.module.cpu().float().predict(DS["X"])
        cluster_acc = model.module.cluster_acc(y_pred,DS['Y'].numpy())
    else:
    

        # for x,y in data_loader:       
        #     y_pred = model.module.cup().flaot().predict(x)
        #     Y_pred.append(y_pred)
        #     Y.append(y)
        # cluster_acc = model.module.cluster_acc(np.concatenate(Y_pred),torch.cat(Y).numpy())
        pc_given_zs = torch.cat(pc_given_zs)
        pc_given_z_np = pc_given_zs.detach().cpu().numpy()
        Y_pred = np.argmax(pc_given_z_np, axis=1)
        cluster_acc = model.module.cluster_acc(Y_pred,torch.cat(Y).numpy().astype(int))
    if i >=1:
        print("last losses BCE:{}, KLD:{}, KLDc:{}, cluster acc{}".format(total_recons/N
                , total_KLD/N, total_KLD_c/N, cluster_acc[0]))
        writer.add_scalar("Loss/ita",batch_loss,epoch)
        writer.add_scalar("BCE/ita",BCE,epoch)
        writer.add_scalar("KLD/ita",KLD,epoch)
        writer.add_scalar("KLD_catigorical/ita",KLD_c,epoch)
        writer.add_scalar("L_sparsity/ita",L_sparsity,epoch)
    model.cuda()
for epoch in range(args.epochs):
    print("epoch number{}".format(epoch))
    start_time = time.time()
    
    train(data_loader,epoch)
    lr_scheduler.step(epoch)

    
    KLDs = model.module.kld_unit_guassians_per_cluster()
    for i in range(KLDs.shape[0]):
        dic = {}
        for j in range(KLDs.shape[1]):
            dic['u_{}'.format(j)] = KLDs[i,j]
        writer.add_scalars("KLD cluster {}/ita".format(i),dic, epoch)
    dic.clear()
    for i in range(KLDs.shape[0]):
        dic = {}
        for j in range(KLDs.shape[1]):

            dic['var_{}'.format(j)] = torch.exp(model.module.logvar_c[i,j])
        writer.add_scalars("Var cluster {}/ita".format(i),dic, epoch)
    dic.clear()
    #if epoch % args.analysis_after_epochs == 0:
        #analyser.write_state(epoch, weights=False)
    end_time = time.time()
    #print(analyser.update_str(epoch=epoch, epoch_time=(end_time - start_time)))
    torch.save(model.state_dict(), './model_{}.pk'.format(args.dataset))
# data_loader = iter(data_loader)
# images, labels = data_loader.next()
# writer.add_graph(model,images)