import torch
import torch.nn as nn
import numpy as np


def safe_exp(x, name=None):
    return torch.exp(x)
    # return tf.check_numerics(tf.exp(x), message='%s is NaN' % name)


torchType = torch.float32
numpyType = np.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
                 use_temperature=False):
        super().__init__()

        self.x_dim = x_dim  # dimensionality of input x
        self.use_temperature = use_temperature  # whether we use temperature or not
        self.temperature = torch.tensor(5.0, dtype=torchType).to(device)  # temperature value

        if not hmc:
            alpha = torch.tensor(eps, requires_grad=eps_trainable).to(device)
        else:
            alpha = torch.tensor(eps, dtype=torchType).to(device)

        self.eps = alpha  # safe_exp(alpha, name='alpha') here I skipped this overaccurate
        # thing with safe_exp
        self._fn = energy_function  # -log(p(variable)) is what is energy function
        self.hmc = hmc  # whether we use hmc or not
        self.T = T
        self._init_mask()
        self.T = torch.tensor(T, dtype=torchType).to(device)  # number of iteration for forward/backward procedure during training

        # if HMC we just return all zeros
        if hmc:
            # z = lambda x, *args, **kwargs: tf.zeros_like(x)
            self.XNet = lambda inp: [torch.zeros_like(inp[0]).to(device) for t in range(3)]
            self.VNet = lambda inp: [torch.zeros_like(inp[0]).to(device) for t in range(3)]
        else:
            self.XNet = net_factory(x_dim, scope='XNet', factor=2.0).to(device)  # net_factory is just a NN
            self.VNet = net_factory(x_dim, scope='VNet', factor=1.0).to(device)

    def _init_mask(self):  # just set a half of components to zero, and a half to one
        # why it is necessarily to make these mask for all T at one time and to store all them?
        mask_per_step = []
        for t in range(self.T):
            ind = np.random.permutation(np.arange(self.x_dim))[:int(self.x_dim / 2)]
            m = np.zeros((self.x_dim,))
            m[ind] = 1
            mask_per_step.append(m)
        self.mask = torch.tensor(np.stack(mask_per_step), dtype=torchType).to(device)

    def _get_mask(self, step):
        m = self.mask[step.type(torch.int32), ...]  # , torch.tensor(step, dtype=torchType))
        return m, 1. - m

    def _format_time(self, t, tile=1):
        trig_t = torch.tensor([
            torch.cos(2 * np.pi * t / self.T),
            torch.sin(2 * np.pi * t / self.T),
        ])
        out = trig_t[None].repeat(tile, 1)
        assert out.shape == (tile, 2), 'in _format_time'
        out = out.to(device)
        return out  # outputs tensor of size tile x 2

    def kinetic(self, v):
        return 0.5 * torch.sum(v**2, dim=1)

    def _forward_step(self, x, v, step, aux=None):
        # transformation which corresponds for d=+1
        t = self._format_time(step, tile=x.shape[0])  # time has size x.shape[0] x 2
        grad1 = self.grad_energy(x, aux=aux)  # gets gradient of sum of energy function at points x wrt x
        # if it is required to use temperature, energy function is scaled by it
        S1 = self.VNet([x, grad1, t, aux])  # this network is for momentum

        # here we get final outputs of our networks
        sv1 = 0.5 * self.eps * S1[0]  # Sv
        tv1 = S1[1]  # Tv
        fv1 = self.eps * S1[2]  # Qv

        v_h = v * safe_exp(sv1) + 0.5 * self.eps * (
                    -(safe_exp(fv1) * grad1) + tv1)  # should be - in front of tv1 according
        #  to formulas from the article! (but may be does not matter since tv1 is learnable)

        m, mb = self._get_mask(step)  # m and 1 - m

        # m, mb = self._gen_mask(x)
        X1 = self.XNet([v_h, m * x, t, aux])  # input is current momentum (output of the previous network v_h,
        # a half of current coordinates, time moment t)

        sx1 = (self.eps * X1[0])  # Sx
        tx1 = X1[1]  # Tx
        fx1 = self.eps * X1[2]  # Qx

        y = m * x + mb * (x * safe_exp(sx1) + self.eps * (safe_exp(fx1) * v_h + tx1))

        X2 = self.XNet([v_h, mb * y, t, aux])

        sx2 = (self.eps * X2[0])
        tx2 = X2[1]
        fx2 = self.eps * X2[2]

        x_o = mb * y + m * (y * safe_exp(sx2) + self.eps * (safe_exp(fx2) * v_h + tx2))

        grad2 = self.grad_energy(x_o, aux=aux)

        S2 = self.VNet([x_o, grad2, t, aux])  # last momentum update
        sv2 = (0.5 * self.eps * S2[0])
        tv2 = S2[1]
        fv2 = self.eps * S2[2]

        v_o = v_h * safe_exp(sv2) + 0.5 * self.eps * (-(safe_exp(fv2) * grad2) + tv2)
        log_jac_contrib = torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

        #  x_o - output of coordinates
        #  v_0 - output of momentums
        #  log_jac_contrib - logarithm of Jacobian of forward transformation


        return x_o, v_o, log_jac_contrib

    def _backward_step(self, x_o, v_o, step, aux=None):
        # transformation which corresponds for d=-1

        t = self._format_time(step, tile=x_o.shape[0])
        grad1 = self.grad_energy(x_o, aux=aux)

        S1 = self.VNet([x_o, grad1, t, aux])

        sv2 = (-0.5 * self.eps * S1[0])
        tv2 = S1[1]
        fv2 = self.eps * S1[2]

        v_h = safe_exp(sv2) * (v_o - 0.5 * self.eps * (-safe_exp(fv2) * grad1 + tv2))

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, mb * x_o, t, aux])

        sx2 = (-self.eps * X1[0])
        tx2 = X1[1]
        fx2 = self.eps * X1[2]

        y = mb * x_o + m * safe_exp(sx2) * (x_o - self.eps * (safe_exp(fx2) * v_h + tx2))

        X2 = self.XNet([v_h, m * y, t, aux])

        sx1 = (-self.eps * X2[0])
        tx1 = X2[1]
        fx1 = self.eps * X2[2]

        x = m * y + mb * safe_exp(sx1) * (y - self.eps * (safe_exp(fx1) * v_h + tx1))

        grad2 = self.grad_energy(x, aux=aux)

        S2 = self.VNet([x, grad2, t, aux])

        sv1 = (-0.5 * self.eps * S2[0])
        tv1 = S2[1]
        fv1 = self.eps * S2[2]

        v = safe_exp(sv1) * (v_h - 0.5 * self.eps * (-safe_exp(fv1) * grad2 + tv1))

        return x, v, torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

    def energy(self, x, aux=None):
        if self.use_temperature:
            T = self.temperature
        else:
            T = torch.tensor(1.0, dtype=torchType).to(device)

        if aux is not None:
            return self._fn(x, aux=aux) / T
        else:
            return self._fn(x) / T

    def hamiltonian(self, x, v, aux=None):
        # x = x.detach()
        # v = v.detach()
        return self.energy(x, aux=aux) + self.kinetic(v)

    def grad_energy(self, x, aux=None):
        new_x = x #.detach()
        new_x.requires_grad_(True)
        energy = self.energy(new_x, aux=aux)
        sum_energy = torch.sum(energy)
        out = torch.autograd.grad(sum_energy, new_x)[0].detach().requires_grad_(False)
        # sum_energy.backward()
        # out[out != out] = 0.0
        # out = x
        # out = new_x.grad
        return out

    def _gen_mask(self, x):
        # dX = x.get_shape().as_list()[1]
        b = np.zeros(self.x_dim)
        for i in range(self.x_dim):
            if i % 2 == 0:
                b[i] = 1
        b = b.astype('bool')
        nb = np.logical_not(b)

        return b.astype(numpyType), nb.astype(numpyType)

    def forward(self, x, init_v=None, aux=None, log_path=False, log_jac=False):

        # this function repeats _step_forward T times

        if init_v is None:
            v = torch.randn_like(x)
        else:
            v = init_v

        dN = x.shape[0]
        t = torch.tensor(0., dtype=torchType).to(device)
        j = torch.zeros(dN, ).to(device)

        X_buf = x.detach() #.clone() #.data
        v_buf = v.detach() #.clone() #.data

        while t < self.T:
            X_buf, v_buf, log_j = self._forward_step(X_buf, v_buf, t, aux=aux)
            j += log_j
            t += 1

        # X_buf.detach_()
        # v_buf.detach_()

        if log_jac:
            return X_buf, v_buf, j

        return X_buf, v_buf, self.p_accept(x, v, X_buf, v_buf, j, aux=aux)

    def backward(self, x, init_v=None, aux=None, log_jac=False):

        # this function repeats _step_backward T times

        if init_v is None:
            v = torch.randn_like(x)
        else:
            v = init_v

        dN = x.shape[0]
        t = torch.tensor(0., dtype=torchType).to(device)
        j = torch.zeros(dN, ).to(device)

        X_buf = x.detach() #.clone() #.data
        v_buf = v.detach() #.clone() #.data

        while t < self.T:
            X_buf, v_buf, log_j = self._backward_step(X_buf, v_buf, self.T - t - 1, aux=aux)
            j += log_j
            t += 1

        # X_buf.detach_()
        # v_buf.detach_()

        if log_jac:
            return X_buf, v_buf, j

        return X_buf, v_buf, self.p_accept(x, v, X_buf, v_buf, j, aux=aux)

    def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
        e_old = self.hamiltonian(x0, v0, aux=aux)
        e_new = self.hamiltonian(x1, v1, aux=aux)

        v = e_old - e_new + log_jac
        other = torch.tensor(0.0).to(device)
        p = torch.exp(torch.min(v, other))

        return torch.where(torch.isfinite(p), p, torch.zeros_like(p))
