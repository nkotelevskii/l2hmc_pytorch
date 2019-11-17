import torch
import torch.nn as nn
import numpy as np
import pdb

def safe_exp(x, name=None):
    return torch.exp(x)
    # return tf.check_numerics(tf.exp(x), message='%s is NaN' % name)


torchType = torch.float32
numpyType = np.float32
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Dynamics(nn.Module):
    """
    The main class that describes modidied HMC dynamics
    """
    def __init__(self,
                 x_dim,
                 energy_function,
                 T=25,
                 eps=0.1,
                 hmc=False,
                 net_factory=None,
                 eps_trainable=True,
                 use_temperature=False,
                 device='cpu'):
        super(Dynamics, self).__init__()
        self.device = device
        self.x_dim = x_dim  # dimensionality of input x
        self.use_temperature = use_temperature  # whether we use temperature or not
        self.temperature = torch.tensor(5.0, dtype=torchType, device=device)  # temperature value
        if not hmc:
            self.alpha = nn.Parameter(torch.tensor(eps, dtype=torchType, device=device),
                                    requires_grad=eps_trainable)
        else:
            self.alpha = torch.tensor(eps, dtype=torchType, device=device)

        self._fn = energy_function  # -log(p(variable)) is what is energy function
        self.hmc = hmc  # whether we use hmc or not

        # if HMC we just return all zeros
        if hmc:
            # z = lambda x, *args, **kwargs: tf.zeros_like(x)
            self.XNet = lambda inp: [torch.zeros_like(inp[0], device=device) for t in range(3)]
            self.VNet = lambda inp: [torch.zeros_like(inp[0], device=device) for t in range(3)]
        else:
            self.XNet = net_factory(x_dim, scope='XNet', factor=2.0).to(device)  # net_factory is just a NN
            self.VNet = net_factory(x_dim, scope='VNet', factor=1.0).to(device)

        self.T = T
        self._init_mask()
        self.T = torch.tensor(T, dtype=torchType, device=device)  # number of iteration for forward/backward procedure during training
        if self.use_temperature:
            self.Temp = torch.tensor(self.temperature, dtype=torchType, device=device)
        else:
            self.Temp = torch.tensor(1.0, dtype=torchType, device=device)

    def _init_mask(self):  # just set a half of components to zero, and a half to one
        # why it is necessarily to make these mask for all T at one time and to store all them?
        mask_per_step = []
        device = self.device
        for t in range(self.T):
            ind = np.random.permutation(torch.arange(self.x_dim))[:int(self.x_dim / 2)]
            m = np.zeros((self.x_dim,))
            m[ind] = 1
            mask_per_step.append(m)
        self.mask = torch.tensor(np.stack(mask_per_step), dtype=torchType, device=device)

    def _get_mask(self, step):
        m = self.mask[step.type(torch.int32), ...]  # , torch.tensor(step, dtype=torchType))
        return m, 1. - m

    def _format_time(self, t, tile=1):
        trig_t = torch.cat([
            torch.cos(2 * np.pi * t / self.T)[..., None],
            torch.sin(2 * np.pi * t / self.T)[..., None],
        ])
        out = trig_t[None].repeat(tile, 1)
        assert out.shape == (tile, 2), 'in _format_time'
        return out  # outputs tensor of size tile x 2

    def kinetic(self, v):
        return 0.5 * torch.sum(v**2, dim=1)

    def _forward_step(self, x, v, step, aux=None):
        # transformation which corresponds for d=+1
        # pdb.set_trace()
        eps = self.alpha
        t = self._format_time(step, tile=x.shape[0])  # time has size x.shape[0] x 2
        grad1 = self.grad_energy(x, aux=aux)  # gets gradient of sum of energy function at points x wrt x
        # if it is required to use temperature, energy function is scaled by it

        S1 = self.VNet([x, grad1, t, aux])  # this network is for momentum

        # here we get final outputs of our networks
        sv1 = 0.5 * eps * S1[0]  # Sv
        tv1 = S1[1]  # Tv
        fv1 = eps * S1[2]  # Qv

        v_h = v * safe_exp(sv1) + 0.5 * eps * (
                    -(safe_exp(fv1) * grad1) + tv1)  # should be - in front of tv1 according
        #  to formulas from the article! (but may be does not matter since tv1 is learnable)

        m, mb = self._get_mask(step)  # m and 1 - m

        # m, mb = self._gen_mask(x)
        X1 = self.XNet([v_h, m * x, t, aux])  # input is current momentum (output of the previous network v_h,
        # a half of current coordinates, time moment t)

        sx1 = (eps * X1[0])  # Sx
        tx1 = X1[1]  # Tx
        fx1 = eps * X1[2]  # Qx

        y = m * x + mb * (x * safe_exp(sx1) + eps * (safe_exp(fx1) * v_h + tx1))

        X2 = self.XNet([v_h, mb * y, t, aux])

        sx2 = (eps * X2[0])
        tx2 = X2[1]
        fx2 = eps * X2[2]

        x_o = mb * y + m * (y * safe_exp(sx2) + eps * (safe_exp(fx2) * v_h + tx2))

        grad2 = self.grad_energy(x_o, aux=aux)

        S2 = self.VNet([x_o, grad2, t, aux])  # last momentum update
        sv2 = (0.5 * eps * S2[0])
        tv2 = S2[1]
        fv2 = eps * S2[2]

        v_o = v_h * safe_exp(sv2) + 0.5 * eps * (-(safe_exp(fv2) * grad2) + tv2)
        log_jac_contrib = torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

        #  x_o - output of coordinates
        #  v_o - output of momentum
        #  log_jac_contrib - logarithm of Jacobian of forward transformation
        return x_o, v_o, log_jac_contrib

    def _backward_step(self, x_o, v_o, step, aux=None):
        # transformation which corresponds for d=-1
        eps = self.alpha
        t = self._format_time(step, tile=x_o.shape[0])
        grad1 = self.grad_energy(x_o, aux=aux)

        S1 = self.VNet([x_o, grad1, t, aux])

        sv2 = (-0.5 * eps * S1[0])
        tv2 = S1[1]
        fv2 = eps * S1[2]

        v_h = safe_exp(sv2) * (v_o - 0.5 * eps * (-safe_exp(fv2) * grad1 + tv2))

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, mb * x_o, t, aux])

        sx2 = (-eps * X1[0])
        tx2 = X1[1]
        fx2 = eps * X1[2]

        y = mb * x_o + m * safe_exp(sx2) * (x_o - eps * (safe_exp(fx2) * v_h + tx2))

        X2 = self.XNet([v_h, m * y, t, aux])

        sx1 = (-eps * X2[0])
        tx1 = X2[1]
        fx1 = eps * X2[2]

        x = m * y + mb * safe_exp(sx1) * (y - eps * (safe_exp(fx1) * v_h + tx1))

        grad2 = self.grad_energy(x, aux=aux)
        S2 = self.VNet([x, grad2, t, aux])

        sv1 = (-0.5 * eps * S2[0])
        tv1 = S2[1]
        fv1 = eps * S2[2]

        v = safe_exp(sv1) * (v_h - 0.5 * eps * (-safe_exp(fv1) * grad2 + tv1))

        return x, v, torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

    def energy(self, x, aux=None):
        if aux is not None:
            return self._fn(x, aux=aux) / self.Temp
        else:
            return self._fn(x) / self.Temp

    def hamiltonian(self, x, v, aux=None):
        return self.energy(x, aux=aux) + self.kinetic(v)

    def grad_energy(self, x, aux=None):
        flag = x.requires_grad
        with torch.set_grad_enabled(True):
            if not flag:
                x = x.data.requires_grad_(True)
            energy = self.energy(x, aux=aux)
            out = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
            if not flag:
                out = out.data
        return out

    def _gen_mask(self, x):
        b = np.zeros(self.x_dim)
        for i in range(self.x_dim):
            if i % 2 == 0:
                b[i] = 1
        b = b.astype('bool')
        nb = np.logical_not(b)

        return b.astype(numpyType), nb.astype(numpyType)

    def forward(self, x, init_v=None, aux=None, log_path=False, log_jac=False):
        # this function repeats _step_forward T times
        device = self.device
        if init_v is None:
            v = torch.tensor(np.random.randn(*list(x.shape)), dtype=torchType, device=device)
        else:
            v = init_v

        dN = x.shape[0]
        t = torch.tensor(0., dtype=torchType, device=device)
        j = torch.zeros(dN, dtype=torchType, device=device)

        x_init = x.data
        v_init = v.data

        while t < self.T:
            x, v, log_j = self._forward_step(x, v, t, aux=aux)
            j += log_j
            t += 1

        if log_jac:
            return x, v, j

        return x, v, self.p_accept(x_init, v_init, x, v, j, aux=aux)

    def backward(self, x, init_v=None, aux=None, log_jac=False):
        # this function repeats _step_backward T times
        device = self.device
        if init_v is None:
            v = torch.tensor(np.random.randn(*list(x.shape)), dtype=torchType, device=device)
        else:
            v = init_v

        dN = x.shape[0]
        t = torch.tensor(0., dtype=torchType, device=device)
        j = torch.zeros(dN, dtype=torchType, device=device)

        x_init = x.data #.clone() #.data
        v_init = v.data #.clone() #.data

        while t < self.T:
            x, v, log_j = self._backward_step(x, v, self.T - t - 1, aux=aux)
            j += log_j
            t += 1

        if log_jac:
            return x, v, j

        return x, v, self.p_accept(x_init, v_init, x, v, j, aux=aux)

    def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
        device = self.device
        e_old = self.hamiltonian(x0, v0, aux=aux)
        e_new = self.hamiltonian(x1, v1, aux=aux)

        v = e_old - e_new + log_jac
        other = torch.zeros_like(v)
        p = torch.exp(torch.min(v, other))

        return torch.where(torch.isfinite(p), p, torch.zeros_like(p))
