import math
import os
import torch
from lib import dist
from tqdm import tqdm
from dataset import return_data
from lib.functions import compute_disc_embds_var
from torch.distributions import kl_divergence, Categorical
from  lib.dist import Normal
import lib.ops as ops
def estimate_entropies(qz_samples, qz_params, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """
    q_dist = Normal()
    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, torch.randperm(qz_samples.size(1))[:n_samples].cuda())
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()


    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - torch.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        pbar.update(batch_size)
    pbar.close()

    entropies /= float(S)

    return entropies


def mutual_info_metric_shapes(vae, args):
    dataset_loader = return_data(args)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.module.z_dim                    # number of latent variables
    nparams = 2

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs,_ in dataset_loader:
        batch_size = xs.size(0)
        xs = xs.view(batch_size, 1, 64, 64).cuda()
        qz_params[n:n + batch_size] = vae.module.encoder(xs).view(batch_size, vae.module.z_dim, nparams).data
        n += batch_size

    if not vae.module.computes_std:
        qz_params[:,:,1] = qz_params[:,:,1] *0.5
    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda()
    mu, logstd_var = qz_params.select(-1,0), qz_params.select(-1,1)

    qz_samples = vae.module.reparam(mu, logstd_var)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams))

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)

    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams))

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams))

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams))

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams))

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_faces(vae, args):
    dataset_loader = return_data(args)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.module.z_dim                    # number of latent variables
    nparams = 2


    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs, _ in dataset_loader:
        batch_size = xs.size(0).cuda()
        xs = xs.view(batch_size, 1, 64, 64).cuda()
        qz_params[n:n + batch_size] = vae.module.encoder.forward(xs).view(batch_size, vae.module.z_dim, nparams).data
        n += batch_size

    if not vae.module.computes_std:
        qz_params[:,:,1] = qz_params[:,:,1]  *0.5
    qz_params = qz_params.view(50, 21, 11, 11, K, nparams).cuda()
    mu, logstd_var = qz_params.select(-1,0), params.select(-1,1)

    qz_samples = vae.module.reparam(mu, logstd_var)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams))

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(3, K)

    print('Estimating conditional entropies for azimuth.')
    for i in range(21):
        qz_samples_pose_az = qz_samples[:, i, :, :, :].contiguous()
        qz_params_pose_az = qz_params[:, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_az.view(N // 21, K).transpose(0, 1),
            qz_params_pose_az.view(N // 21, K, nparams))

        cond_entropies[0] += cond_entropies_i.cpu() / 21

    print('Estimating conditional entropies for elevation.')
    for i in range(11):
        qz_samples_pose_el = qz_samples[:, :, i, :, :].contiguous()
        qz_params_pose_el = qz_params[:, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_el.view(N // 11, K).transpose(0, 1),
            qz_params_pose_el.view(N // 11, K, nparams))

        cond_entropies[1] += cond_entropies_i.cpu() / 11

    print('Estimating conditional entropies for lighting.')
    for i in range(11):
        qz_samples_lighting = qz_samples[:, :, :, i, :].contiguous()
        qz_params_lighting = qz_params[:, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_lighting.view(N // 11, K).transpose(0, 1),
            qz_params_lighting.view(N // 11, K, nparams))

        cond_entropies[2] += cond_entropies_i.cpu() / 11

    metric = compute_metric_faces(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies
def get_samples_F_VAE_metric(model, args):
    dataset_loader = return_data(args)

    N = len(dataset_loader.dataset)  # number of data samples
    K = args.z_dim                    # number of latent variables



    nparams = 2
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs,_ in dataset_loader:
            batch_size = xs.shape[0]
            qz_params[n:n + batch_size] = model.module.encoder(xs.cuda()).view(batch_size, model.module.z_dim, nparams).data
            n += batch_size


    mu, logstd_var = qz_params.select(-1,0), qz_params.select(-1,1)
    z = model.module.reparam(mu, logstd_var)
    KLDs =  model.module.kld_unit_guassians_per_sample(mu,logstd_var).mean(0)
    

    # discarding latent dimensions with small KLD
    idx = torch.where(KLDs>1e-2)[0]
    mu = mu[:,idx]
    K = mu.shape[1]
    list_samples = []
    if args.dataset =='dsprites':
        mu = mu.view(3, 6, 40, 32, 32, K)
        # for factor in range(3):
        #     if factor == 0:
        #         shape_fixed = mu[0,:,:,:,:].view(1,6*40*32*32,K)
        #     else:
        #         shape_fixed = torch.cat([shape_fixed, mu[factor,:,:,:,:].view(1,6*40*32*32,K)],dim=0)
        fixed = torch.randint(0,3,(1,))
        shape_fixed = mu[fixed,:,:,:,:].view(6*40*32*32,K)
        list_samples.append(shape_fixed)
        del shape_fixed

        fixed = torch.randint(0,6,(1,))
        scale_fixed = mu[:,fixed,:,:,:].view(3*40*32*32,K)
        list_samples.append(scale_fixed)
        del scale_fixed

        fixed = torch.randint(0,40,(1,))
        orientation_fixed = mu[:,:,fixed,:,:].view(3*6*32*32,K)
        list_samples.append(orientation_fixed)
        del orientation_fixed

        fixed = torch.randint(0,32,(1,))
        posx_fixed= mu[:,:,:,fixed,:].view(3*6*40*32,K)
        list_samples.append(posx_fixed)
        del posx_fixed

        fixed = torch.randint(0,32,(1,))
        posy_fixed = mu[:,:,:,:,fixed].view(3*6*40*32,K)
        list_samples.append(posy_fixed)
        del posy_fixed
    else:
        pass

    return list_samples

def get_samples_F_VAE_metric_v2(model, L, num_votes, args, used_smaples = None):
    """This is the one that is used in the paper
    """
    dataset_loader = return_data(args)

    N = len(dataset_loader.dataset)  # number of data samples
    K = args.z_dim                    # number of latent variables

        
    nparams = 2
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs,_ in dataset_loader:
            batch_size = xs.shape[0]
            qz_params[n:n + batch_size] = model.module.encoder(xs.cuda()).view(batch_size, model.module.z_dim, nparams).data
            n += batch_size


    mu, logstd_var = qz_params.select(-1,0), qz_params.select(-1,1)
    z = model.module.reparam(mu, logstd_var)
    KLDs =  model.module.kld_unit_guassians_per_sample(mu,logstd_var).mean(0)
    
    # discarding latent dimensions with small KLD
    
    # idx = torch.where(KLDs>1e-2)[0]
    global_var = torch.var(mu,axis = 0)
    idx = torch.where(global_var>5e-2)[0]
    mu = mu[:,idx]
    K = mu.shape[1]
    list_samples = []
    global_var = global_var[idx]
    if args.dataset =='dsprites':
        if used_smaples == None:
            used_smaples = []
            factors = [3, 6, 40, 32, 32] 
            for f in factors:
                used_smaples.append([ 0 for _ in range(f)])


        # 5 is the number of generative factors
        num_votes_per_factor = num_votes / 5
        num_samples_per_factor = int(num_votes_per_factor * L)
        mu = mu.view(3, 6, 40, 32, 32, K)
        # for factor in range(3):
        #     if factor == 0:
        #         shape_fixed = mu[0,:,:,:,:].view(1,6*40*32*32,K)
        #     else:
        #         shape_fixed = torch.cat([shape_fixed, mu[factor,:,:,:,:].view(1,6*40*32*32,K)],dim=0)
        shape_fixed = torch.zeros((num_samples_per_factor,K))
        for idx in range(0,num_samples_per_factor,L):
            fixed = torch.randint(0,3,(1,))

            shape_fixed[idx:idx+L] = mu[fixed,:,:,:,:].view(6*40*32*32,K)[torch.randint(0,6*40*32*32,(L,)),:]

        list_samples.append(shape_fixed)
        del shape_fixed

        scale_fixed = torch.zeros((num_samples_per_factor,K))
        for idx in range(0,num_samples_per_factor,L):
            fixed = torch.randint(0,6,(1,))
 
            scale_fixed[idx:idx+L] = mu[:,fixed,:,:,:].view(3*40*32*32,K)[torch.randint(0,3*40*32*32,(L,)),:]
        list_samples.append(scale_fixed)
        del scale_fixed

        orientation_fixed = torch.zeros((num_samples_per_factor,K))
        for idx in range(0,num_samples_per_factor,L):
            fixed = torch.randint(0,40,(1,))
     
            orientation_fixed[idx:idx+L] = mu[:,:,fixed,:,:].view(3*6*32*32,K)[torch.randint(0,3*6*32*32,(L,)),:]
   
        list_samples.append(orientation_fixed)
        del orientation_fixed

        posx_fixed = torch.zeros((num_samples_per_factor,K))
        for idx in range(0,num_samples_per_factor,L):
            fixed = torch.randint(0,32,(1,))
     
            posx_fixed[idx:idx+L] = mu[:,:,:,fixed,:].view(3*6*40*32,K)[torch.randint(0,3*6*40*32,(L,)),:]

        list_samples.append(posx_fixed)
        del posx_fixed

        posy_fixed = torch.zeros((num_samples_per_factor,K))
        for idx in range(0,num_samples_per_factor,L):

            idx = used_smaples[4][fixed]
            posy_fixed[idx:idx+L] = mu[:,:,:,:,fixed].view(3*6*40*32,K)[torch.randint(0,3*6*40*32,(L,)),:]
    
        list_samples.append(posy_fixed)
        del posy_fixed
    else:
        pass

    return list_samples, global_var, used_smaples
def get_samples_F_VAE_metric_v3(model, L, num_votes, num_test_votes, args):
    """This is the one that is used in the paper
    """
    dataset_loader = return_data(args)

    N = len(dataset_loader.dataset)  # number of data samples
    K = args.z_dim                    # number of latent variables

        
    nparams = 2
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    with torch.no_grad():
        for xs,_ in dataset_loader:
            batch_size = xs.shape[0]
            qz_params[n:n + batch_size] = model.module.encoder(xs.cuda()).view(batch_size, model.module.z_dim, nparams).data
            n += batch_size


    mu, logstd_var = qz_params.select(-1,0), qz_params.select(-1,1)
    z = model.module.reparam(mu, logstd_var)
    KLDs =  model.module.kld_unit_guassians_per_sample(mu,logstd_var).mean(0)
    
    # discarding latent dimensions with small KLD
    
    idx = torch.where(KLDs>1)[0]
    print('Mean KL-diveregence of units')
    # print(KLDs)
    

    std = torch.std(mu,axis=0)
    mu = mu / std
    # global_var = torch.var(mu,axis = 0)
    # print("variances: ")
    # print(global_var)
    # idx = torch.where(global_var>0.005)[0]
    mu = mu[:,idx]
    K = mu.shape[1]
    print('There are :{} active unit'.format(K))
    list_samples = []
    list_test_samples = []
    # global_var = global_var[idx]
   
    if args.dataset =='dsprites':
  
        # 5 is the number of generative factors
        num_votes_per_factor = num_votes / 5
        num_samples_per_factor = int(num_votes_per_factor * L)

        num_test_votes_per_factor = num_test_votes / 5
        num_test_samples_per_factor = int(num_test_votes_per_factor * L)

        mu = mu.view(3, 6, 40, 32, 32, K)

        #####################
        # SHAPE FIXED
        #####################
        unused = [] # the unused indices
        for _ in range(3):
            unused.append(ops.choice(0,6*40*32*32))

        shape_fixed, unused = ops.get_fixed_factor_samples(mu, 0, num_samples_per_factor, K, L, 3, unused)
        list_samples.append(shape_fixed)
        del shape_fixed
        shape_fixed = torch.zeros((num_test_samples_per_factor,K))
        shape_fixed, _ = ops.get_fixed_factor_samples(mu, 0, num_test_samples_per_factor, K, L, 3, unused)
        list_test_samples.append(shape_fixed)
        del shape_fixed
        ###############################

        ################################
        # SCALE FIXED
        ################################
        unused = [] # the unused indices
        for _ in range(6):
            unused.append(ops.choice(0,3*40*32*32))

        scale_fixed, unused = ops.get_fixed_factor_samples(mu, 1, num_samples_per_factor, K, L, 6, unused)
        list_samples.append(scale_fixed)
        del scale_fixed
        scale_fixed, _ = ops.get_fixed_factor_samples(mu, 1, num_test_samples_per_factor, K, L, 6, unused)
        list_test_samples.append(scale_fixed)
        del scale_fixed
        ################################

        ################################
        # ORIENTATION FIXED
        ################################
        unused = [] # the unused indices
        for _ in range(40):
            unused.append(ops.choice(0,3*6*32*32))
        orientation_fixed, unused = ops.get_fixed_factor_samples(mu, 2, num_samples_per_factor, K, L, 40, unused)
        list_samples.append(orientation_fixed)
        del orientation_fixed
        orientation_fixed, _ = ops.get_fixed_factor_samples(mu, 2, num_test_samples_per_factor, K, L, 40, unused)
        list_test_samples.append(orientation_fixed)
        del orientation_fixed

        #################################

        #################################
        # COORDINATE ON X-AXIS FIXED
        #################################
        unused = [] # the unused indices
        for _ in range(32):
            unused.append(ops.choice(0,3*6*40*32))

        posx_fixed, unused = ops.get_fixed_factor_samples(mu, 3, num_samples_per_factor, K, L, 32, unused)
        list_samples.append(posx_fixed)
        del posx_fixed
        posx_fixed, _ = ops.get_fixed_factor_samples(mu, 3, num_test_samples_per_factor, K, L, 32, unused)
        list_test_samples.append(posx_fixed)
        del posx_fixed
        ###############################

        #################################
        # COORDINATE ON Y-AXIS FIXED
        #################################
        unused = [] # the unused indices
        for _ in range(32):
            unused.append(ops.choice(0,3*6*40*32))

        posy_fixed, unused = ops.get_fixed_factor_samples(mu, 4, num_samples_per_factor, K, L, 32, unused)
        list_samples.append(posy_fixed)
        del posy_fixed
        posy_fixed, _ = ops.get_fixed_factor_samples(mu, 4, num_test_samples_per_factor, K, L, 32, unused)
        list_test_samples.append(posy_fixed)
        del posy_fixed
        ###############################

    else:
        num_votes_per_factor = num_votes / 6
        num_samples_per_factor = int(num_votes_per_factor * L)

        num_test_votes_per_factor = num_test_votes / 6
        num_test_samples_per_factor = int(num_test_votes_per_factor * L)
        mu = mu.view(10, 10, 10, 8, 4, 15, K)
        factors_variations = [10, 10, 10, 8, 4, 15]
        total_multi = 10 * 10 * 10 * 8 * 4 * 15
        num_rest = [ total_multi // vari for vari in factors_variations]
        for fac_id, vari in enumerate(factors_variations):
            unused = [] # the unused indices
            for _ in range(vari):
                unused.append(ops.choice(0,num_rest[fac_id]))
            genfac_fixed, unused = ops.get_fixed_factor_samples(mu, fac_id, num_samples_per_factor, K, L, vari, unused)
            list_samples.append(genfac_fixed)
            del genfac_fixed
            genfac_fixed, _ = ops.get_fixed_factor_samples(mu, fac_id, num_test_samples_per_factor, K, L, vari, unused)
            list_test_samples.append(genfac_fixed)
            del genfac_fixed


    return list_samples, list_test_samples




def _compute_votes_matrix(list_samples, votes_per_factor, L):
    for factor_id ,samples in enumerate(list_samples):
        N = samples.shape[0]

        for idx in range(0,N,L):
            end_batch = idx + L
            if end_batch >= N:
                end_batch = N
            
            embds_var = torch.var(samples[idx:end_batch],dim=0)
            argmin = torch.argmin(embds_var)
            # list_argmin.append(torch.tensor(argmin).view(1,).clone())
            # list_ground_truth.append(torch.tensor(factor_id).view(1,))
            votes_per_factor[factor_id,argmin] += 1
    return votes_per_factor


def F_VAE_metric(model, args, L=100):

    #list_samples = get_samples_F_VAE_metric(model,args)
    list_samples, list_test_samples = get_samples_F_VAE_metric_v3(model,L, 800, 800,args)
    #each element in the list is samples generated with one factor-of-generation fixed
    num_genfactors =len(list_samples)
    n_latent = list_samples[0].shape[1]
    votes_per_factor = torch.zeros((num_genfactors,n_latent))
    classifier = torch.zeros(n_latent)
    # total_votes =0
    votes_per_factor = _compute_votes_matrix(list_samples, votes_per_factor, L)
            # total_votes += 1


    classifier = torch.argmax(votes_per_factor,dim=0)

    # argmins = torch.cat(list_argmin)
    # ground_truths = torch.cat(list_ground_truth,0)

    # #accuracy
    # acc = torch.sum(classifier[argmins] == ground_truths) / float(total_votes)
    acc = float(votes_per_factor[classifier,torch.arange(n_latent)].sum()) / float(votes_per_factor.sum())

    ## test_set evaluation

    votes_per_factor = torch.zeros((num_genfactors,n_latent))    
    votes_per_factor = _compute_votes_matrix(list_test_samples, votes_per_factor, L)
    acc_test = float(votes_per_factor[classifier,torch.arange(n_latent)].sum()) / float(votes_per_factor.sum())
    return acc, acc_test


def get_samples_F_VAE_metric_disc(model, L, num_votes, num_test_votes, args):
    dataset_loader = return_data(args)

    N = len(dataset_loader.dataset)  # number of data samples
    K = 2*model.module.num_latent_dims +  \
        model.module.num_latent_dims_disc          # number of latent variables



    nparams = 2
    qz_params = torch.Tensor(N, K).cuda()

    n = 0
    with torch.no_grad():

        #self.model.eval()
        means = []
        for xs,_ in dataset_loader:
            batch_size = xs.shape[0]
            qz_params[n:n + batch_size] = model.module.encoder(xs.cuda())
            n += batch_size


    mu, logstd_var = qz_params[:, 0: model.module.z_dim[0]], qz_params[:,model.module.z_dim[0]:2*model.module.z_dim[0]]

    d = 2*model.module.z_dim[0] #starting index of discrete vars
    alphas = []
    rep_as = 0
    if len(model.module.z_dim[1])>0:

        for dims in model.module.z_dim[1]:
            alphas.append(torch.softmax(qz_params[:,d:d+dims],dim=-1).cuda())
            d += dims
        rep_as =[]
        for alpha in alphas:
            rep_as.append(dist.Gumbel_Softmax.sample(alpha,0.67,False))
        rep_as = model.module.reparam_disc(alphas)
 
        accepted_disc = []
        accepted_disc_dims = []
        for index_disc, alpha in enumerate(alphas):
            unifom_params = torch.ones_like(alpha)/alpha.shape[1]
            kld = kl_divergence(Categorical(alpha),Categorical(unifom_params))
            if kld.mean() >1e-2:
                # normalize
                alpha = alpha / torch.sqrt( compute_disc_embds_var(alpha,[alpha.shape[1]])[0].float())


                accepted_disc.append(alpha)
                accepted_disc_dims.append(alpha.shape[-1])
 
    KLDs =  model.module.kld_unit_guassians_per_sample(mu,logstd_var)
    KLDs = KLDs.sum(0) / N
    # discarding latent dimensions with small KLD
    idx = torch.where(KLDs>5e-2)[0]
    mu = mu[:,idx]
    num_con_dims = mu.shape[1]
    list_num_con_disc = [ alpha.shape[-1] for alpha in accepted_disc]
    list_samples = []
    list_test_samples = []
    #append the accepted disc latent dims
    if len(accepted_disc) >1:
        accepted_disc = torch.cat(accepted_disc, dim=-1)
    elif len(accepted_disc) == 1:
        accepted_disc = accepted_disc[0]
    

    mu = mu / torch.std(mu, axis = 0)
    #discrete embds have been already normilized 
    if len(accepted_disc) != 0:
        mu  = torch.cat([mu.cuda(),accepted_disc.cuda()], dim=-1)
    K = mu.shape[1]
    
    
    if args.dataset =='dsprites':
        # 5 is the number of generative factors
        num_votes_per_factor = num_votes / 5
        num_samples_per_factor = int(num_votes_per_factor * L)

        num_test_votes_per_factor = num_test_votes / 5
        num_test_samples_per_factor = int(num_test_votes_per_factor * L)
        mu = mu.view(3, 6, 40, 32, 32, K)

        #####################
        # SHAPE FIXED
        #####################
        unused = [] # the unused indices
        for _ in range(3):
            unused.append(ops.choice(0,6*40*32*32))

        shape_fixed, unused = ops.get_fixed_factor_samples(mu, 0, num_samples_per_factor, K, L, 3, unused)
        list_samples.append(shape_fixed)
        del shape_fixed
        shape_fixed = torch.zeros((num_test_samples_per_factor,K))
        shape_fixed, _ = ops.get_fixed_factor_samples(mu, 0, num_test_samples_per_factor, K, L, 3, unused)
        list_test_samples.append(shape_fixed)
        del shape_fixed
        ###############################

        ################################
        # SCALE FIXED
        ################################
        unused = [] # the unused indices
        for _ in range(6):
            unused.append(ops.choice(0,3*40*32*32))

        scale_fixed, unused = ops.get_fixed_factor_samples(mu, 1, num_samples_per_factor, K, L, 6, unused)
        list_samples.append(scale_fixed)
        del scale_fixed
        scale_fixed, _ = ops.get_fixed_factor_samples(mu, 1, num_test_samples_per_factor, K, L, 6, unused)
        list_test_samples.append(scale_fixed)
        del scale_fixed
        ################################

        ################################
        # ORIENTATION FIXED
        ################################
        unused = [] # the unused indices
        for _ in range(40):
            unused.append(ops.choice(0,3*6*32*32))
        orientation_fixed, unused = ops.get_fixed_factor_samples(mu, 2, num_samples_per_factor, K, L, 40, unused)
        list_samples.append(orientation_fixed)
        del orientation_fixed
        orientation_fixed, _ = ops.get_fixed_factor_samples(mu, 2, num_test_samples_per_factor, K, L, 40, unused)
        list_test_samples.append(orientation_fixed)
        del orientation_fixed

        #################################

        #################################
        # COORDINATE ON X-AXIS FIXED
        #################################
        unused = [] # the unused indices
        for _ in range(32):
            unused.append(ops.choice(0,3*6*40*32))

        posx_fixed, unused = ops.get_fixed_factor_samples(mu, 3, num_samples_per_factor, K, L, 32, unused)
        list_samples.append(posx_fixed)
        del posx_fixed
        posx_fixed, _ = ops.get_fixed_factor_samples(mu, 3, num_test_samples_per_factor, K, L, 32, unused)
        list_test_samples.append(posx_fixed)
        del posx_fixed
        ###############################

        #################################
        # COORDINATE ON Y-AXIS FIXED
        #################################
        unused = [] # the unused indices
        for _ in range(32):
            unused.append(ops.choice(0,3*6*40*32))

        posy_fixed, unused = ops.get_fixed_factor_samples(mu, 4, num_samples_per_factor, K, L, 32, unused)
        list_samples.append(posy_fixed)
        del posy_fixed
        posy_fixed, _ = ops.get_fixed_factor_samples(mu, 4, num_test_samples_per_factor, K, L, 32, unused)
        list_test_samples.append(posy_fixed)
        del posy_fixed
        ###############################
    else:
        num_votes_per_factor = num_votes / 6
        num_samples_per_factor = int(num_votes_per_factor * L)

        num_test_votes_per_factor = num_test_votes / 6
        num_test_samples_per_factor = int(num_test_votes_per_factor * L)
        mu = mu.view(10, 10, 10, 8, 4, 15, K)
        factors_variations = [10, 10, 10, 8, 4, 15]
        total_multi = 10 * 10 * 10 * 8 * 4 * 15
        num_rest = [ total_multi // vari for vari in factors_variations]
        for fac_id, vari in enumerate(factors_variations):
            unused = [] # the unused indices
            for _ in range(vari):
                unused.append(ops.choice(0,num_rest[fac_id]))
            genfac_fixed, unused = ops.get_fixed_factor_samples(mu, fac_id, num_samples_per_factor, K, L, vari, unused)
            list_samples.append(genfac_fixed)
            del genfac_fixed
            genfac_fixed, _ = ops.get_fixed_factor_samples(mu, fac_id, num_test_samples_per_factor, K, L, vari, unused)
            list_test_samples.append(genfac_fixed)
            del genfac_fixed

    return list_samples, list_test_samples,num_con_dims, accepted_disc_dims


        



def F_VAE_metric_disc(model, args,L=100):

    list_samples, list_test_samples, num_con_dims, accepted_disc_dims = get_samples_F_VAE_metric_disc(model,L, 800, 800,args)

    #each element in the list is samples generated with one factor-of-generation fixed
    num_genfactors =len(list_samples)
    n_latent = list_samples[0].shape[1]
    votes_per_factor = torch.zeros((num_genfactors,n_latent))
    classifier = torch.zeros(n_latent)

    classifier = []

    for factor_id ,samples in enumerate(list_samples):
        N = samples.shape[0]
      
        for idx in range(0,N,L):
            end_batch = idx + L
            if end_batch >= N:
                end_batch = N
            
            embds_var = torch.var(samples[idx:end_batch,:num_con_dims],dim=0)
            if len(accepted_disc_dims) > 0:
                embds_var_disc = compute_disc_embds_var(samples[idx:end_batch,num_con_dims:], accepted_disc_dims)
                if len(embds_var_disc) > 1:
                    embds_var_disc = torch.cat(embds_var_disc).view(-1,)
                else: 
                    embds_var_disc = embds_var_disc[0].view(-1,)

                embds_var = torch.cat([embds_var,embds_var_disc])


            argmin = torch.argmin(embds_var)
            votes_per_factor[factor_id,argmin] += 1



    classifier = torch.argmax(votes_per_factor,dim=0)
    acc = float(votes_per_factor[classifier,torch.arange(n_latent)].sum()) / float(votes_per_factor.sum())

    for factor_id ,samples in enumerate(list_test_samples):
        N = samples.shape[0]
      
        for idx in range(0,N,L):
            end_batch = idx + L
            if end_batch >= N:
                end_batch = N
            
            embds_var = torch.var(samples[idx:end_batch,:num_con_dims],dim=0)

            if len(accepted_disc_dims) > 0:
                embds_var_disc = compute_disc_embds_var(samples[idx:end_batch,num_con_dims:], accepted_disc_dims)
                if len(embds_var_disc) > 1:
                    embds_var_disc = torch.cat(embds_var_disc).view(-1,)
                else: 
                    embds_var_disc = embds_var_disc[0].view(-1,)

                embds_var = torch.cat([embds_var,embds_var_disc])
            argmin = torch.argmin(embds_var)
            votes_per_factor[factor_id,argmin] += 1

    acc_test = float(votes_per_factor[classifier,torch.arange(n_latent)].sum()) / float(votes_per_factor.sum())

    return float(acc), acc_test

def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies):
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = MIG(mi_normed)
    return metric


def compute_metric_faces(marginal_entropies, cond_entropies):
    factor_entropies = [21, 11, 11]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = MIG(mi_normed)
    return metric


    

