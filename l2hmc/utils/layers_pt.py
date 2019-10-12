import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torchType = torch.float32

class Dense(nn.Module):
    """Dense module"""

    def __init__(self, in_, out_, scope='dense', factor=1.0):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_, out_)
        weights = lambda tensor: self.init_weights(tensor, factor)
        self.fc.apply(weights).to(device)

    def init_weights(self, tensor, factor):
        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(torch.tensor(0., dtype=torchType))
            nn.init.kaiming_normal_(tensor.weight.data, mode='fan_in', a=factor * 2)

    def forward(self, x):
        # print('x.shape in Dense', x.shape)
        return self.fc(x)


class Parallel(nn.Module):
    """Parallel module"""

    def __init__(self, layers=[]):
        super(Parallel, self).__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        return [layer(x) for layer in self.layers]


class ScaleTanh(nn.Module):
    """Scaled tanh (lambda * tanh)"""

    def __init__(self, in_, scope='scaled_tanh'):
        super(ScaleTanh, self).__init__()
        self.scale = torch.exp(torch.zeros(in_, dtype=torchType).to(device))
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.scale * self.tanh(x)


class Zip(nn.Module):
    """Zip module"""

    def __init__(self, layers=[]):
        super(Zip, self).__init__()
        self.layers = layers

    def forward(self, x):
        assert len(x) == len(self.layers)
        n = len(self.layers)
        return [self.layers[i](x[i]) for i in range(n)]


class Net(nn.Module):
    """Multilayer perceptron"""

    def __init__(self, x_dim, factor, scope=None):
        super(Net, self).__init__()

        self.embed = nn.Sequential(Zip([
            Dense(x_dim, 10, scope='embed_1', factor=1.0 / 3),
            Dense(x_dim, 10, scope='embed_2', factor=factor * 1.0 / 3),
            Dense(2, 10, scope='embed_3', factor=1.0 / 3)
        ]))

        self.fc1 = Dense(10, 10, scope='linear_1')

        self.fc2 = Parallel([
            nn.Sequential(
                Dense(10, x_dim, scope='linear_s', factor=0.001),
                ScaleTanh(x_dim, scope='scale_s')),
            Dense(10, x_dim, scope='linear_t', factor=0.001),
            nn.Sequential(
                Dense(10, x_dim, scope='linear_f', factor=0.001),
                ScaleTanh(x_dim, scope='scale_f'))
        ])

    def forward(self, x, scope=None, factor=None):
        if (x[-1] is None):
            x = x[:-1]
        x = self.embed(x)
        x = F.relu(sum(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
