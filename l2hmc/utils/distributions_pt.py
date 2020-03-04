import collections
import torch
import numpy as np
import torch.nn as nn
from scipy.stats import multivariate_normal, ortho_group
import pdb
torchType = torch.float32
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def quadratic_gaussian(x, mu, S):
    matrix = torch.mm(torch.mm(x - mu, S), (x - mu).T)
    matrix *= 0.5
    return torch.diag(matrix)


class Gaussian(nn.Module):
    def __init__(self, mu, sigma, device='cpu'):
        super(Gaussian, self).__init__()
        self.device = device
        self.mu = mu.type(torchType)
        self.sigma = sigma.type(torchType)
        self.target_distr = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu,
                                                                                covariance_matrix=self.sigma)

    def get_energy_function(self):
        # def fn(x, *args, **kwargs):
        #     return quadratic_gaussian(x.type(torchType), self.mu, self.i_sigma)
        def fn(x):
            return -self.target_distr.log_prob(x)
        return fn

    def get_samples(self, n):
        '''
        Sampling is broken in numpy for d > 10
        '''
        return self.target_distr.sample((n, )).cpu().detach().numpy()

    def log_density(self, X):
        # pdb.set_trace()
        return self.target_distr.log_prob(X)


class GMM(nn.Module):
    def __init__(self, mus, sigmas, pis, device='cpu'):
        super(GMM, self).__init__()
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.p = pis[0]  # probability of the first gaussian (1-p for the second)
        self.log_pis = [torch.tensor(np.log(self.p), dtype=torch.float32, device=device),
                        torch.tensor(np.log(1 - self.p), dtype=torch.float32,
                                     device=device)]  # LOGS! probabilities of Gaussians
        self.locs = mus  # list of locations for each of these gaussians
        self.covs = sigmas  # list of covariance matrices for each of these gaussians
        self.dists = [torch.distributions.MultivariateNormal(loc=self.locs[0], covariance_matrix=self.covs[0]),
                      torch.distributions.MultivariateNormal(loc=self.locs[1], covariance_matrix=self.covs[
                          1])]  # list of distributions for each of them

    def get_energy_function(self):
        def fn(x):
            return -self.log_density(x)
        return fn

    def get_samples(self, n):
        n_first = int(n * self.p)
        n_second = n - n_first
        samples_1 = self.dists[0].sample((n_first,))
        samples_2 = self.dists[1].sample((n_second,))
        samples = torch.cat([samples_1, samples_2])
        return samples.cpu().detach().numpy()

    def log_density(self, X):
        log_p_1 = (self.log_pis[0] + self.dists[0].log_prob(X)).view(-1, 1)
        log_p_2 = (self.log_pis[1] + self.dists[1].log_prob(X)).view(-1, 1)
        log_p_1_2 = torch.cat([log_p_1, log_p_2], dim=-1)
        log_density = torch.logsumexp(log_p_1_2, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density
