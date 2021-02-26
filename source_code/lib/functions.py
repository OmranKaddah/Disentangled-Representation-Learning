import torch
from torch.autograd import Function


class STHeaviside(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(x.size()).type_as(x)
        y[x >= 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

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
    N  = embds.shape[0]
    num_comparisons = 2*64 *(64-1)
    list_vars = []
    start_index = 0
    for end_index in list_disc_dims:
        disc_embd = embds[:,start_index:start_index+end_index]
        variance =0
        for i in range(0, N, 64):
            if i+64 > N:
                end = N
            else:
                end = i+64
                
            active_indecies = torch.argmax(disc_embd[i:end],dim=-1)
            active_indecies_repeated = active_indecies.repeat(active_indecies.shape[0],1)
            active_indecies_repeated_tr = active_indecies_repeated.transpose(0,1)
            variance += (active_indecies_repeated_tr != active_indecies_repeated).sum(-1).sum(0) / float(num_comparisons)
        # active_indecies = torch.argmax(disc_embd,dim=-1)
        # active_indecies_repeated = active_indecies.repeat(N,1)
        # active_indecies_repeated_tr = active_indecies_repeated.transpose(0,1)
        # variance = (active_indecies_repeated_tr != active_indecies_repeated).sum(-1).sum(0) / num_comparisons
        list_vars.append(variance)
        start_index = end_index
    
    return list_vars

def permute_dims(z):

    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)