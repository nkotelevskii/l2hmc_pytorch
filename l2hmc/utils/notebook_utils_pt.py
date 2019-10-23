import numpy as np
from dynamics_pt import Dynamics
from sampler_pt import propose
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def get_hmc_samples(x_dim, eps, energy_function, T=10, steps=200, samples=None):
    hmc_dynamics = Dynamics(x_dim, energy_function, T=T, eps=eps, hmc=True)
    if samples is None:
        samples = gaussian.get_samples(n=200)

    final_samples = []

    for t in tqdm(range(steps)):
        final_samples.append(samples.cpu().numpy())  # do we need .item() ?
        _, _, _, samples = propose(samples, hmc_dynamics, do_mh_step=True)
        samples = samples[0].detach()

    return np.array(final_samples)

def plot_gaussian_contours(mus, covs, colors=['blue', 'red'], spacing=5,
        x_lims=[-4,4], y_lims=[-3,3], res=100):

    X = np.linspace(x_lims[0], x_lims[1], res)
    Y = np.linspace(y_lims[0], y_lims[1], res)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for i in range(len(mus)):
        mu = mus[i]
        cov = covs[i]
        F = multivariate_normal(mu, cov)
        Z = F.pdf(pos)
        plt.contour(X, Y, Z, spacing, colors=colors[0])

    return plt