
import torch

def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def compute_disc_embds_var(embds, list_disc_dims):
    """computes variances for latent discrete embeddings

    Parameters
    ----------
    embds : Tensor floats [N, sum(list_disc_dims)]
        Discreet embeddings, where each unit of index # has size has size 
        list_disc_dims[#]
    list_disc_dims : list of ints
        Contains size of each discrete unit

    Outputs:

    varaince : list of floats
        variance of disc unit
    """
    N  = embds.shape[1]
    num_comparisons = 2*N *(N-1)
    list_vars = []
    start_index = 0
    for end_index in list_disc_dims:
        disc_embd = embds[:,start_index:end_index]
        active_indecies = torch.argmax(disc_embd,dim=-1)
        active_indecies_repeated = active_indecies.expand(-1)
        active_indecies_repeated_tr = active_indecies_repeated_tr.transpose(0,1)
        variance = (active_indecies_repeated_tr != active_indecies_repeated).sum(-1).sum(0) / num_comparisons
        list_vars.append(variance)
        start_index = end_index
    
    return list_vars

def permute_dims(z):



    B = z.shape[0]
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def choice(first, last):
    perm = torch.randperm(last)
    perm += first
    return perm

def get_fixed_factor_samples(dataset, factor_idx, num_samples, K, L, num_variations, n_used_idx):
    fixed_factor = torch.zeros((num_samples, K))
    for i in range(0,num_samples,L):
        fixed = int(torch.randint(0,num_variations,(1,)))
        
        if i+L> num_samples:
            end = num_samples
            idx = n_used_idx[fixed][:num_samples - i]
        else:
            end = i + L
            idx = n_used_idx[fixed][:L]
        
        fixed_factor[i:end] = torch.reshape(dataset.select(factor_idx,fixed), (-1,K))[idx,:]
        n_used_idx[fixed] = n_used_idx[fixed][L:] 
    return fixed_factor, n_used_idx
def equal_kl_loss(kl_units):
    """
    A loss that pushes KL diveregence of units to be equal, by computing all possible permutations 
    differences between unis, and then divide by the number of permutations

    Parameters:

    kl_units: Tensor shape=[1,D]

    return:
    
    loss: float
    """
    D  = kl_units.shape[0]
    # num_comparisons = 2*D *(D-1)
    list_vars = []
    start_index = 0


    units_repeated = kl_units.repeat(D,1)
    units_repeated_tr = units_repeated.transpose(0,1)
    loss = torch.abs(units_repeated - units_repeated_tr).sum(-1).sum(0) / float(2)


    return loss