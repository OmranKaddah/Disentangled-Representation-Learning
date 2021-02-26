import torch
import sys
import torchsummary

args = torch.load("outputs_new/{}/args-0000.pth".format(sys.argv[1]))['args']
mod = __import__('models.beta_vae_archs', fromlist=['Beta_VAE_{}'.format(args.arch)])
hyp_params = [args.alpha, args.beta, args.gamma]
model = torch.nn.DataParallel(getattr(mod, 'Beta_VAE_{}'.format(args.arch))(z_dim=args.z_dim, n_cls=args.n_cls,
                            hyper_params= hyp_params,
                        computes_std=args.computes_std, nc=args.number_channel, 
                        output_type= args.output_type, x_num_param= args.x_num_param).cuda(), device_ids=range(1))
try:
    torchsummary.summary(model,(1, 28, 28))
except:
    print("Something Went wrong")

pp=0
for p in list(model.parameters()):
    nn=1
    for s in list(p.size()):
        nn = nn*s
    pp += nn
print("Number of parameters", pp)
print('--name ', args.name)
print('--model ', args.model)
print('--batch_size ', args.batch_size)

print('--epochs ', args.epochs)
print('--lr ', args.lr)
print('--z_dim ', args.z_dim)

print('--output_type ', args.output_type)
print('--loss_fun ', args.loss_fun)
print('--arch ', args.arch)
print('--analytical ', args.analytical)
print('--alpha ', args.alpha)
print('--beta ', args.beta)
print('--gamma ', args.gamma)
print('--capacity ', args.capacity)
print('--reg_anneal ', args.reg_anneal)
print('--kl_equality ', args.kl_equality)

print('--dataset ', args.dataset)
print('--image_size ', args.image_size)

print('--computes_std ', args.computes_std)
print('--number_channel ', args.number_channel)

print('--x_num_param ', args.x_num_param)
### Factor VAE stuff
if args.factorvae_tc:
    print('--factorvae_tc ', args.factorvae_tc)
    print('--lr_discriminator ', args.lr_discriminator)
    ####

if args.model == "jointvae":
    #
    #  arguments only for joint-VAE on main_j_vae.py
    print('--con_latent_dims ', args.con_latent_dims)
    print('--disc_latent_dims ', args.disc_latent_dims )
    print('--alpha_disc ', args.alpha_disc)
    print('--beta_disc ', args.beta_disc)
    print('--gamma_disc ', args.gamma_disc)
    print('--capacity_disc ', args.capacity_disc)
    print('--temprature ', args.temprature)
    ##################################################

