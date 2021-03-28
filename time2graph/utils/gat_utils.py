import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, reshape=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.reshape = reshape

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        nbatch, N = h.size()[0], h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(nbatch, N * N, -1),
                             h.repeat(1, N, 1)], dim=2).view(nbatch, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.reshape:
            h_prime = h_prime.view(nbatch, -1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(GATDataloader, self).__init__(*args, **kwargs)


class GATDataset(object):
    def __init__(self, feat, adj, y=None):
        if y is not None:
            self.data = [(feat[k], adj[k], y[k]) for k in range(len(y))]
        else:
            self.data = [(feat[k], adj[k]) for k in range(len(adj))]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
