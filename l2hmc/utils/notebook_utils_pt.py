import numpy as np
from dynamics_pt import Dynamics
from sampler_pt import propose
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def get_hmc_samples(x_dim, eps, energy_function, T=10, steps=200, samples=None):
    hmc_dynamics = Dynamics(x_dim, energy_function, T=T, eps=eps, hmc=True).eval()
    if samples is None:
        samples = gaussian.get_samples(n=200)

    final_samples = []

    for t in tqdm(range(steps)):
        final_samples.append(samples.cpu().detach().numpy())  # do we need .item() ?
        _, _, _, samples = propose(samples, hmc_dynamics, do_mh_step=True)
        samples = samples[0]

    return np.array(final_samples)
