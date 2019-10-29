import torch
import numpy as np
from scipy.stats import multivariate_normal


def accept(x_i, x_p, p):
    assert x_i.shape == x_p.shape

    dN, dX = x_i.shape

    u = np.random.uniform(size=(dN,))

    m = (p - u >= 0).astype('int32')[:, None]

    return x_i * (1 - m) + x_p * m


def autocovariance(X, tau=0):
    dT, dN, dX = np.shape(X)
    s = 0.

    for t in range(dT - tau):
        x1 = X[t, :, :]
        x2 = X[t + tau, :, :]

        s += np.sum(x1 * x2) / dN

    return s / (dT - tau)

def autocovariance_1(X, tau=0):
    dT, dN, dX = np.shape(X)
    s = 0.

    for t in range(dT - tau):
        x1 = X[t, :, :]
        x2 = X[t + tau, :, :]

        s += np.sum((x1 - x1.mean(0)) * (x2 - x2.mean(0))) / dN

    return s / (dT - tau)

def autocovariance_2(X, tau=0):
    dT, dN, dX = np.shape(X)
    s = 0.
    for t in range(dT - tau):
        x1 = X[t, :, :]
        x2 = X[t + tau, :, :]
        mean = X.mean(0)
        s += np.sum((x1 - mean) * (x2 - mean)) / dN

    return s / (dT - tau)


def acl_spectrum(X, scale):
    n = X.shape[0]
    return np.array([autocovariance_2(X / scale, tau=t) for t in range(n - 1)])


def ESS(A):
    A = A * (A > 0.05)
    return 1. / (1. + 2 * np.sum(A[1:]))
