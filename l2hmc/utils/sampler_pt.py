import torch
import torch.nn as nn
import numpy as np
import pdb

torchType = torch.float32
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def propose(x, dynamics, init_v=None, aux=None, do_mh_step=False, log_jac=False, trainable=True, temperature=None, device='cpu'):
    if dynamics.hmc:
        Lx, Lv, px = dynamics.forward(x, init_v=init_v, aux=aux)
        return Lx, Lv, px, [tf_accept(x, Lx, px)]
    else:
        # sample mask for forward/backward
        if temperature is not None:
            dynamics.temperature = temperature
        mask = (2 * torch.tensor(np.random.rand(x.shape[0], 1))).type(torch.int32).type(torchType).to(device)
        x_clone = x.data
        # pdb.set_trace()
        Lx1, Lv1, px1 = dynamics.forward(x, aux=aux, log_jac=log_jac)
        Lx2, Lv2, px2 = dynamics.backward(x_clone, aux=aux, log_jac=log_jac)

    Lx = mask * Lx1 + (1 - mask) * Lx2  # by this we imitate the random choice of d (direction)

    # orig  1 - orig
    #  0       1
    #  1       0

    Lv = None
    if init_v is not None:
        Lv = mask * Lv1 + (1 - mask) * Lv2

    px = torch.squeeze(mask, dim=1) * px1 + torch.squeeze(1 - mask, dim=1) * px2

    outputs = []

    if do_mh_step:
        outputs.append(tf_accept(x, Lx, px, device))

    return Lx, Lv, px, outputs  # new coordinates, new momenta, new acceptance probability and outputs for coodinates,
    # taking MH acceptance in account

def tf_accept(x, Lx, px, device='cpu'):
    mask = (px - torch.tensor(np.random.rand(*list(px.shape)), device=device, dtype=torchType) >= 0.)[:, None]
    mask = torch.cat([mask, mask], dim=-1)
    return torch.where(mask, Lx, x).detach()

def chain_operator(init_x, dynamics, nb_steps, aux=None, init_v=None, do_mh_step=False, device='cpu'):
    if not init_v:
        init_v = torch.tensor(np.random.randn(*list(init_x.shape)), device=device, dtype=torchType)
    final_x, final_v, log_jac, t = init_x, init_v, torch.zeros(init_x.shape[0]), torch.tensor(0.)

    while t < nb_steps.type(torchType):
        final_x, final_v, px, _ = propose(final_x, dynamics, init_v=final_v, aux=aux, log_jac=True, do_mh_step=False)
        log_jac += px
        t += 1

    p_accept = dynamics.p_accept(init_x, init_v, final_x, final_v, log_jac, aux=aux)
    outputs = []

    if do_mh_step:
        outputs.append(tf_accept(init_x, final_x, p_accept))
    return final_x, final_v, p_accept, outputs
