import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torchType = torch.float32

class Dense(nn.Module):
    """Dense module"""

    def __init__(self, in_, out_, scope='dense', factor=1.0):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_, out_)
        self.fc.bias = nn.Parameter(torch.zeros(out_, dtype=torchType))
        self.fc.weight = nn.Parameter(0.0001 * torch.ones((out_, in_), dtype=torchType))

    def forward(self, x):
        return self.fc(x)


class Parallel(nn.Module):
    """Parallel module"""

    def __init__(self, x_dim, factor):
        super(Parallel, self).__init__()
        self.seq1 = nn.Sequential(
            Dense(10, x_dim, scope='linear_s', factor=0.001),
            ScaleTanh(x_dim, scope='scale_s'))
        self.fc_par_1 = Dense(10, x_dim, scope='linear_t', factor=0.001)

        self.seq2 = nn.Sequential(
            Dense(10, x_dim, scope='linear_f', factor=0.001),
            ScaleTanh(x_dim, scope='scale_f'))

    def forward(self, x):
        return [self.seq1(x), self.fc_par_1(x), self.seq2(x)]


class ScaleTanh(nn.Module):
    """Scaled tanh (lambda * tanh)"""

    def __init__(self, in_, scope='scaled_tanh'):
        super(ScaleTanh, self).__init__()
        self.param = nn.Parameter(torch.zeros(in_, dtype=torchType))
        self.tanh = nn.Tanh()

    def forward(self, x):
        scale = torch.exp(self.param)
        return scale * self.tanh(x)


class Zip(nn.Module):
    """Zip module"""
    def __init__(self, x_dim, factor):
        super(Zip, self).__init__()
        self.fc_zip_1 = Dense(x_dim, 10, scope='embed_1', factor=1.0 / 3)
        self.fc_zip_2 = Dense(x_dim, 10, scope='embed_2', factor=factor * 1.0 / 3)
        self.fc_zip_3 = Dense(2, 10, scope='embed_3', factor=1.0 / 3)

    def forward(self, x):
        assert len(x) == 3
        return [self.fc_zip_1(x[0]), self.fc_zip_2(x[1]), self.fc_zip_3(x[2])]

class Net_old(nn.Module):
    """Multilayer perceptron"""
    def __init__(self, x_dim, factor, scope=None):
        super(Net_old, self).__init__()
        self.embed = Zip(x_dim, factor)
        self.fc1 = Dense(10, 10, scope='linear_1')
        self.fc2 = Parallel(x_dim, factor)
        # pdb.set_trace()

    def forward(self, x, scope=None, factor=None):
        if (x[-1] is None):
            x = x[:-1]
        x = self.embed(x)
        x = F.relu(sum(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net(nn.Module):
    """Multilayer perceptron"""
    def __init__(self, x_dim, factor, scope=None):
        super(Net, self).__init__()
        self.fc_zip_1 = nn.Linear(x_dim, 10)
        nn.init.kaiming_normal_(self.fc_zip_1.weight, mode='fan_in', a=1./3.)
        nn.init.constant_(self.fc_zip_1.bias, 0.)

        self.fc_zip_2 = nn.Linear(x_dim, 10)
        nn.init.kaiming_normal_(self.fc_zip_2.weight, mode='fan_in', a=1./3.)
        nn.init.constant_(self.fc_zip_2.bias, 0.)

        self.fc_zip_3 = nn.Linear(2, 10)
        nn.init.kaiming_normal_(self.fc_zip_3.weight, mode='fan_in', a=1./3.)
        nn.init.constant_(self.fc_zip_3.bias, 0.)

        self.linear = nn.Linear(10, 10)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', a=1.)
        nn.init.constant_(self.linear.bias, 0.)

        self.fc_par_1 = nn.Linear(10, x_dim)
        nn.init.kaiming_normal_(self.fc_par_1.weight, mode='fan_in', a=0.001)
        nn.init.constant_(self.fc_par_1.bias, 0.)

        self.scale_1 = nn.Parameter(torch.zeros((1, x_dim)))

        self.fc_par_2 = nn.Linear(10, x_dim)
        nn.init.kaiming_normal_(self.fc_par_2.weight, mode='fan_in', a=0.001)
        nn.init.constant_(self.fc_par_2.bias, 0.)

        self.fc_par_3 = nn.Linear(10, x_dim)
        nn.init.kaiming_normal_(self.fc_par_3.weight, mode='fan_in', a=0.001)
        nn.init.constant_(self.fc_par_3.bias, 0.)

        self.scale_3 = nn.Parameter(torch.zeros((1, x_dim)))


    def forward(self, x, scope=None, factor=None):
        if (x[-1] is None):
            x = x[:-1]

        h1 = self.fc_zip_1(x[0])[None]
        h2 = self.fc_zip_2(x[1])[None]
        h3 = self.fc_zip_3(x[2])[None]

        h_sum = torch.relu(torch.sum(torch.cat([h1, h2, h3], dim=0), dim=0))

        h4 = torch.relu(self.linear(h_sum))

        out1 = torch.exp(self.scale_1) * torch.tanh(self.fc_par_1(h4))
        out2 = self.fc_par_2(h4)
        out3 = torch.exp(self.scale_3) * torch.tanh(self.fc_par_3(h4))

        return out1, out2, out3
