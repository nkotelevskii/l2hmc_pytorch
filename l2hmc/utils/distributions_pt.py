import collections
import torch
import numpy as np
import torch.nn as nn
from scipy.stats import multivariate_normal, ortho_group

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
        self.i_sigma = torch.tensor(np.linalg.inv(sigma.cpu()), device=device, dtype=torchType)

    def get_energy_function(self):
        def fn(x, *args, **kwargs):
            return quadratic_gaussian(x.type(torchType), self.mu, self.i_sigma)

        return fn

    def get_logdensity(self, x):
        return -self.get_energy_function()(x)

    def get_samples(self, n):
        '''
        Sampling is broken in numpy for d > 10
        '''
        C = np.linalg.cholesky(self.sigma.cpu())
        X = np.random.randn(n, self.sigma.shape[0])
        return X.dot(C.T)

    def log_density(self, X):
        return multivariate_normal(mean=self.mu, cov=self.sigma).logpdf(X)


class GMM(nn.Module):
    def __init__(self, mus, sigmas, pis, device='cpu'):
        super(GMM, self).__init__()
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.device = device
        self.mus = mus
        self.sigmas = sigmas
        self.pis = pis

        self.nb_mixtures = len(pis)

        self.k = mus[0].shape[0]

        self.i_sigmas = []
        self.constants = []

        for i, sigma in enumerate(sigmas):
            self.i_sigmas.append(torch.tensor(np.linalg.inv(sigma).astype('float32'), device=device, dtype=torchType))
            det = np.sqrt((2 * np.pi) ** self.k * np.linalg.det(sigma)).astype('float32')
            self.constants.append(torch.tensor((pis[i] / det).astype('float32'), device=device, dtype=torchType))

        self.mus = torch.tensor(mus, device=device, dtype=torchType)
        self.sigmas = torch.tensor(sigmas, device=device, dtype=torchType)
        self.pis = torch.tensor(pis, device=device, dtype=torchType)

    def get_energy_function(self):
        def fn(x):
            V = torch.cat([
                (-quadratic_gaussian(x, self.mus[i], self.i_sigmas[i])
                               + torch.log(self.constants[i])).unsqueeze(1)
                for i in range(self.nb_mixtures)
            ], dim=1)
            # print('Nans in x:', (x != x).sum())
            # print('V exp sum max', V.exp().sum(1).max())
            # print('V exp sum min', V.exp().sum(1).min())
            # out = V.exp().sum(dim=1)
            # out = -torch.log(out)
            # print('Nans in get_energy_function: ', (out != out).sum())
            # print('Max get_energy_function:', out.max())
            # print('Min get_energy_function:', out.min())
            out = -torch.logsumexp(V, dim=1)
            return out
        return fn

    def get_logdensity(self, x):
        return -self.get_energy_function()(x)

    def get_samples(self, n):
        categorical = np.random.choice(self.nb_mixtures, size=(n,), p=self.pis.cpu().detach().numpy())
        counter_samples = collections.Counter(categorical)

        samples = []

        for k, v in counter_samples.items():
            samples.append(np.random.multivariate_normal(self.mus[k].cpu().detach().numpy(),
                                                         self.sigmas[k].cpu().detach().numpy(),
                                                         size=(v,)))

        samples = np.concatenate(samples, axis=0)

        np.random.shuffle(samples)
        # samples = torch.tensor(samples).to(device).type(torchType)

        return samples

    def log_density(self, X):
        return np.log(sum([self.pis[i] * multivariate_normal(mean=self.mus[i], cov=self.sigmas[i]).pdf(X) for i in
                           range(self.nb_mixtures)]))
