import torch as T
import torch.nn as nn
import numpy as np

class DqnModel(nn.Module):
    """
    Neural Network, choosing actions
    """
    def __init__(self, n_in, n_out):
        super(DqnModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(n_in)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(), 
            nn.Linear(512, n_out)
        )
        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.dtype = T.cuda.FloatTensor if T.cuda.is_available() else T.FloatTensor #here

    def _get_conv_out(self, shape):
        o = self.conv(T.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.type(self.dtype) 
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class DuelingDqnModel(nn.Module):
    def __init__(self, n_in, n_out):
        super(DuelingDqnModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_in[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(n_in)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_out)
        )
        self.conv.apply(self.init_weights)
        self.value_stream.apply(self.init_weights)
        self.advantage_stream.apply(self.init_weights)

        self.dtype = T.cuda.FloatTensor if T.cuda.is_available() else T.FloatTensor

    def forward(self, x):
        x = x.type(self.dtype)
        conv_out = self.conv(x).view(x.size()[0], -1)
        values = self.value_stream(conv_out)
        advantages = self.advantage_stream(conv_out)
        q_vals = values + (advantages - advantages.mean())
        return q_vals
        
    def _get_conv_out(self, shape):
        o = self.conv(T.zeros(1, *shape))
        return int(np.prod(o.size()))

    def init_weights(self, m):
        if type(m) ==  nn.Linear:
            T.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
