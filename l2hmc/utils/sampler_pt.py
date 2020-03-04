import torch
import torch.nn as nn
import numpy as np
import pdb

torchType = torch.float32


def propose(x, dynamics, init_v=None, aux=None, do_mh_step=False, return_log_jac=False, temperature=None, device='cpu', our_alg=False, use_barker=False):
    if dynamics.hmc:
        Lx, Lv, log_px, _ = dynamics.forward(x, init_v=init_v, aux=aux)
        return Lx, Lv, log_px, [tf_accept(x, Lx, log_px, device)]
    else:
        # sample mask for forward/backward
        if temperature is not None:
            dynamics.temperature = temperature
        mask = (2 * torch.tensor(np.random.rand(x.shape[0], 1))).type(torch.int32).type(torchType).to(device)
        # pdb.set_trace()
        Lx1, Lv1, log_px1, log_jac_f = dynamics.forward(x.clone(), init_v=init_v, aux=aux, return_log_jac=return_log_jac, use_barker=use_barker)
        Lx2, Lv2, log_px2, log_jac_b = dynamics.backward(x.clone(), init_v=init_v, aux=aux, return_log_jac=return_log_jac, use_barker=use_barker)

    # orig  1 - orig
    #  0       1
    #  1       0
    # pdb.set_trace()
    Lx = mask * Lx1 + (1 - mask) * Lx2  # by this we imitate the random choice of d (direction)
    Lv = mask * Lv1 + (1 - mask) * Lv2
    log_jac = torch.squeeze(mask, dim=1) * log_jac_f + torch.squeeze((1 - mask), dim=1) * log_jac_b

    if use_barker:
        log_px = []
        log_px.append(torch.squeeze(mask, dim=1) * log_px1[0] + torch.squeeze(1 - mask, dim=1) * log_px2[0])
        log_px.append(torch.squeeze(mask, dim=1) * log_px1[1] + torch.squeeze(1 - mask, dim=1) * log_px2[1])
    else:
        log_px = torch.squeeze(mask, dim=1) * log_px1 + torch.squeeze(1 - mask, dim=1) * log_px2


    outputs = []
    directions = None
    if our_alg:
        if do_mh_step:
            new_Lx, new_log_px, new_log_jac, directions = tf_accept(x=x, Lx=Lx, log_px=log_px, device=device,
                                                        our_alg=our_alg, use_barker=use_barker, log_jac=log_jac)
            log_jac = new_log_jac
            log_px = new_log_px
            outputs.append(new_Lx)
    else:
        if do_mh_step:
            new_Lx, directions = tf_accept(x=x, Lx=Lx, log_px=log_px, device=device, use_barker=use_barker)
            outputs.append(new_Lx)

    return Lx, Lv, log_px, outputs, log_jac, directions  # new coordinates, new momenta, new acceptance probability, outputs for coodinates, log_jac
    # taking MH acceptance in account

def tf_accept(x, Lx, log_px, device='cpu', our_alg=False, use_barker=False, log_jac=None):
    if our_alg:
        if use_barker:
            # pdb.set_trace()
            logprobs = torch.log(torch.tensor(np.random.rand(*list(log_px[0].shape)), device=device,
                                          dtype=torchType))
            mask = logprobs <= log_px[0]
            new_log_px = torch.where(mask, log_px[0], -log_px[1])
            new_log_jac = torch.where(mask, log_jac, torch.zeros_like(log_jac))
            mask = mask[:, None]
        else:
            logprobs = torch.log(torch.tensor(np.random.rand(*list(log_px.shape)), device=device,
                                          dtype=torchType))
            mask = (logprobs <= log_px)[:, None]
            new_log_px = torch.where(mask, log_px, torch.log(1. - log_px.exp()))
            new_log_jac = torch.where(mask, log_jac, torch.zeros_like(log_jac))

        new_Lx = torch.where(mask, Lx, x)
        return new_Lx, new_log_px, new_log_jac, mask.squeeze()
    else:
        if use_barker:
            logprobs = torch.log(torch.tensor(np.random.rand(*list(log_px[0].shape)), device=device,
                                              dtype=torchType))
            mask = (logprobs <= log_px[0])[:, None]
        else:
            logprobs = torch.log(torch.tensor(np.random.rand(*list(log_px.shape)), device=device, dtype=torchType))
            mask = (logprobs <= log_px)[:, None]
        return torch.where(mask, Lx, x).detach(), mask.squeeze()

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
