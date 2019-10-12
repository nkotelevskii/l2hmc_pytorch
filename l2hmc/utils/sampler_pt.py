import torch
import torch.nn as nn
import numpy as np

torchType = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def propose(x, dynamics, init_v=None, aux=None, do_mh_step=False, log_jac=False, trainable=True, temperature=None):
    if dynamics.hmc:
        Lx, Lv, px = dynamics.forward(x, init_v=init_v, aux=aux)
        return Lx, Lv, px, [tf_accept(x, Lx, px)]
    else:
        if trainable:
            dynamics.VNet.train()
            dynamics.XNet.train()
        else:
            dynamics.VNet.eval()
            dynamics.XNet.eval()
        # sample mask for forward/backward
        if temperature is not None:
            dynamics.temperature = temperature
        mask = (2 * torch.rand((x.shape[0], 1))).type(torch.int32).type(torchType).to(device)
        Lx1, Lv1, px1 = dynamics.forward(x, aux=aux, log_jac=log_jac)
        Lx2, Lv2, px2 = dynamics.backward(x, aux=aux, log_jac=log_jac)

    Lx = mask * Lx1 + (1 - mask) * Lx2  # by this we imitate the random choice of d (direction)

    # orig  1 - orig
    #  0       1
    #  1       0
    #  2      -1  ???

    Lv = None
    if init_v is not None:
        Lv = mask * Lv1 + (1 - mask) * Lv2

    px = torch.squeeze(mask, dim=1) * px1 + torch.squeeze(1 - mask, dim=1) * px2

    outputs = []

    if do_mh_step:
        outputs.append(tf_accept(x, Lx, px))

    return Lx, Lv, px, outputs  # new coordinates, new momenta, new acceptance probability and outputs for coodinates,
    # taking MH acceptance in account


def tf_accept(x, Lx, px):
    mask = (px - torch.rand_like(px).to(device) >= 0.)[:, None]
    mask = torch.cat([mask, mask], dim=-1)
    return torch.where(mask, Lx, x)


def chain_operator(init_x, dynamics, nb_steps, aux=None, init_v=None, do_mh_step=False):
    if not init_v:
        init_v = torch.randn(init_x.shape)

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
